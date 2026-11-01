import Foundation
import XCTest
import CoreML
import CryptoKit
import TruongNoiserCore
@testable import TiledInferenceEngine

/// Compares actual Swift Core ML predictions with a separately retained official FP32 oracle.
/// Simulator GPU errors or the observed invalid all-zero result are unavailable observations,
/// never fidelity passes; the production routing must exclude those requests.
final class CGNetOfficialReferenceTests: XCTestCase, @unchecked Sendable {
    private struct Fixture: Decodable {
        struct Case: Decodable {
            struct Sample: Decodable { let index: Int; let value: Float }
            let name: String
            let orientation: Int
            let inputSHA256: String
            let samples: [Sample]
        }
        let size: Int
        let officialCommit: String
        let checkpointSHA256: String
        let cases: [Case]
    }

    func testCPUOnlyMatchesIndependentOfficialEdgeAndOrientationSamples() async throws { try await compare(.cpuOnly) }
    func testCPUAndNeuralEngineMatchesIndependentOfficialSamples() async throws { try await compare(.cpuAndNeuralEngine) }
    func testRawCPUAndGPURequestMatchesReferenceOrRecordsSimulatorUnavailability() async throws { try await compare(.cpuAndGPU) }
    func testRawAllRequestMatchesReferenceOrRecordsSimulatorUnavailability() async throws { try await compare(.all) }

    private func compare(_ policy: MLComputeUnits) async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let registry = CoreMLModelRegistry(initialComputeUnits: .cpuOnly,
            customSearchDirectories: [root.appendingPathComponent("Targets/TruongNoiserApp/Models")],
            availabilityPolicy: .requireProductionModels)
        let discovered = await registry.discoveredProductionModelURLs()
        for size in [256, 512] {
            let fixtureURL = try XCTUnwrap(Bundle.module.url(forResource: "cgnet-runtime-reference-\(size)",
                withExtension: "json", subdirectory: "Fixtures"))
            let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtureURL))
            XCTAssertEqual(fixture.size, size)
            XCTAssertEqual(fixture.officialCommit, "0daa98442b928274ff50a1b5e594a2b53c94879d")
            XCTAssertEqual(fixture.checkpointSHA256, "d7daaece4350f2e3d505155d1712d22929fec06b463532e076de217996a3462d")
            XCTAssertEqual(fixture.cases.count, 6)
            let identifier: CoreMLModelIdentifier = size == 256 ? .cgnet256 : .cgnet512
            let url = try XCTUnwrap(discovered[identifier])
            let configuration = MLModelConfiguration(); configuration.computeUnits = policy
            let model: MLModel
            do { model = try MLModel(contentsOf: url, configuration: configuration) }
            catch { try unavailableOrThrow(error, policy: policy); throw error }
            for item in fixture.cases {
                let orientation = try XCTUnwrap(TileOrientation(rawValue: item.orientation))
                let supplied = try TileOrientationEnsemble.transformed(baseTensor(item.name, size: size),
                    size: size, orientation: orientation)
                let bytes = supplied.withUnsafeBytes { Data($0) }
                XCTAssertEqual(SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(), item.inputSHA256,
                    "Input must be byte-identical to the independent oracle")
                let tensor = try MLMultiArray(shape: [1, 3, size, size].map { NSNumber(value: $0) }, dataType: .float16)
                XCTAssertEqual(tensor.strides.map(\.intValue), [3 * size * size, size * size, size, 1])
                supplied.withUnsafeBufferPointer { values in
                    tensor.dataPointer.assumingMemoryBound(to: Float16.self).update(from: values.baseAddress!, count: values.count)
                }
                let input = try MLDictionaryFeatureProvider(dictionary: ["input_tile": MLFeatureValue(multiArray: tensor)])
                let prediction: MLFeatureProvider
                do { prediction = try predictSynchronously(model, input: input) }
                catch { try unavailableOrThrow(error, policy: policy); throw error }
                let output = try XCTUnwrap(prediction.featureValue(for: "output_tile")?.multiArrayValue)
                XCTAssertEqual(output.shape.map(\.intValue), [1, 3, size, size])
                XCTAssertEqual(output.dataType, .float16)
                guard output.shape.map(\.intValue) == [1, 3, size, size], output.dataType == .float16 else {
                    throw CocoaError(.coderInvalidValue)
                }
                let strides = output.strides.map(\.intValue)
                let pointer = output.dataPointer.assumingMemoryBound(to: Float16.self)
                func value(_ index: Int) -> Float {
                    let channel = index / (size * size), pixel = index % (size * size)
                    return Float(pointer[channel * strides[1] + (pixel / size) * strides[2] + (pixel % size) * strides[3]])
                }
                #if targetEnvironment(simulator)
                var allZero = true
                #endif
                for index in supplied.indices {
                    let sample = value(index)
                    #if targetEnvironment(simulator)
                    if sample != 0 { allZero = false }
                    #endif
                    if !sample.isFinite || !(-0.25...1.25).contains(sample) {
                        XCTFail("Invalid raw production prediction at \(index), size\(size), policy\(policy.rawValue)")
                        return
                    }
                }
                #if targetEnvironment(simulator)
                if (policy == .cpuAndGPU || policy == .all), allZero, item.name == "gradient" {
                    // This runtime returns a zero tensor without an NSError for these raw GPU
                    // requests. Preserve the failed numerical observation and require the app's
                    // safe routing; do not count the raw request as a reference-fidelity pass.
                    for mode: BackboneModelKind in [.cgnet, .cgnetEnsemble] {
                        XCTAssertEqual(registry.effectiveComputeUnits(for: policy, backbone: mode), .cpuOnly)
                    }
                    XCTAssertThrowsError(try CoreMLModelRegistry.validateCGNetBackendOutput(
                        [Float16](repeating: 0, count: supplied.count), modelID: identifier,
                        policy: .requireProductionModels, tileIndex: 0))
                    let observed: [String: Any] = ["status": "RAW_INVALID_ZERO_TENSOR",
                        "scope": "iOS27 Simulator raw request; not phone behavior or a fidelity pass",
                        "computeRequestedRawValue": policy.rawValue, "size": size,
                        "case": item.name, "inputSHA256": item.inputSHA256,
                        "expectedMaximumSample": item.samples.map(\.value).max() ?? 0,
                        "productionCompute": "cpuOnly"]
                    let attachment = XCTAttachment(data: try JSONSerialization.data(withJSONObject: observed,
                        options: [.prettyPrinted, .sortedKeys]), uniformTypeIdentifier: "public.json")
                    attachment.name = "CGNet-unusable-Simulator-GPU-policy-\(policy.rawValue)"
                    attachment.lifetime = .keepAlways; add(attachment)
                    print("CGNET_POLICY_OBSERVATION", String(decoding: attachmentData(observed), as: UTF8.self))
                    throw XCTSkip("Raw Simulator GPU request returned an invalid zero tensor; no fidelity pass. Production routes both CGNet modes to CPU-only.")
                }
                #endif
                for sample in item.samples {
                    XCTAssertEqual(value(sample.index), sample.value, accuracy: 0.002,
                        "Official reference: \(size)/\(item.name)/orientation\(item.orientation)/policy\(policy.rawValue)")
                }
            }
        }
    }

    #if targetEnvironment(simulator)
    private func attachmentData(_ value: [String: Any]) -> Data {
        (try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])) ?? Data()
    }
    #endif

    private func predictSynchronously(_ model: MLModel, input: MLFeatureProvider) throws -> MLFeatureProvider {
        // Select the same synchronous Core ML entry point used by the production registry.
        // The SDK also exposes an async overload with this argument list.
        try model.prediction(from: input, options: MLPredictionOptions())
    }

    private func unavailableOrThrow(_ error: Error, policy: MLComputeUnits) throws {
        #if targetEnvironment(simulator)
        let failure = error as NSError
        if (policy == .cpuAndGPU || policy == .all), failure.domain == MLModelErrorDomain, failure.code == -14 {
            throw XCTSkip("Raw Simulator GPU execution plan unavailable: \(failure). Production CGNet requests CPU-only on Simulator.")
        }
        #endif
        throw error
    }

    private func baseTensor(_ name: String, size: Int) throws -> [Float16] {
        var tensor = [Float16](repeating: 0, count: 3 * size * size)
        for y in 0..<size {
            for x in 0..<size {
                let offset = y * size + x
                switch name {
                case "gradient":
                    let red = Float16(Float(x) / Float(size - 1))
                    tensor[offset] = red
                    tensor[size * size + offset] = Float16(Float(y) / Float(size - 1))
                    tensor[2 * size * size + offset] = red * Float16(0.5)
                case "fine-color-stripes":
                    let stripe = Float16(x % 2)
                    tensor[offset] = Float16(0.45) + Float16(0.1) * stripe
                    tensor[size * size + offset] = Float16(0.55) - Float16(0.1) * stripe
                    tensor[2 * size * size + offset] = Float16(0.5) + Float16(0.025) * stripe
                default: throw CocoaError(.coderInvalidValue)
                }
            }
        }
        return tensor
    }
}
