import Foundation
import XCTest
import CoreML
import TruongNoiserCore
@testable import TiledInferenceEngine

/// Routing requests are policy, not evidence that Core ML placed an operator on an accelerator.
/// Job-entry tests deliberately stop at a missing source, exercising model binding without loading
/// weights, decoding images or doing inference. Real model execution has a separate regression suite.
final class CGNetComputePolicyTests: XCTestCase, @unchecked Sendable {
    private let requests: [MLComputeUnits] = [.all, .cpuAndGPU, .cpuAndNeuralEngine, .cpuOnly]

    func testRequiredModelRoutingKeepsNAFNetPolicyAndAllowsNativeCGNetForegroundRequests() {
        let registry = CoreMLModelRegistry(availabilityPolicy: .requireProductionModels)
        for request in requests {
            XCTAssertEqual(registry.effectiveComputeUnits(for: request, backbone: .nafnetNormal),
                request == .cpuOnly ? .cpuOnly : .cpuAndNeuralEngine)
            for mode: BackboneModelKind in [.cgnet, .cgnetEnsemble] {
                XCTAssertEqual(registry.effectiveComputeUnits(for: request, backbone: mode), cgNetUnits(request))
            }
        }
    }

    func testExplicitTestPolicyRemainsPassthroughForEveryFamilyAndRequest() {
        let registry = CoreMLModelRegistry(availabilityPolicy: .allowDeterministicFallback)
        for mode in BackboneModelKind.allCases {
            for request in requests {
                XCTAssertEqual(registry.effectiveComputeUnits(for: request, backbone: mode), request)
            }
        }
    }

    func testRawRequestSurvivesNAFNetPolicyCollapsingAndSequentialCGNetModeSwitches() async {
        let registry = CoreMLModelRegistry(initialComputeUnits: .all, availabilityPolicy: .requireProductionModels)
        await assertRegistry(registry, raw: .all, effective: .cpuAndNeuralEngine)
        await registry.setComputeUnits(.cpuAndGPU, backbone: .nafnetNormal)
        // Both foreground requests resolve to CPU+NE for NAFNet, but their raw values must remain
        // distinct so switching to CGNet can recover the user's actual request.
        await assertRegistry(registry, raw: .cpuAndGPU, effective: .cpuAndNeuralEngine)
        let raw = await registry.currentRequestedUnits()
        await registry.setComputeUnits(raw, backbone: .cgnet)
        await assertRegistry(registry, raw: .cpuAndGPU, effective: cgNetUnits(.cpuAndGPU))
        await registry.setComputeUnits(.all, backbone: .cgnetEnsemble)
        await assertRegistry(registry, raw: .all, effective: cgNetUnits(.all))
        await registry.setComputeUnits(.cpuAndNeuralEngine, backbone: .cgnetEnsemble)
        await assertRegistry(registry, raw: .cpuAndNeuralEngine, effective: cgNetUnits(.cpuAndNeuralEngine))
        for mode in BackboneModelKind.allCases {
            await registry.setComputeUnits(.cpuOnly, backbone: mode)
            await assertRegistry(registry, raw: .cpuOnly, effective: .cpuOnly)
        }
    }

    func testExecutorRebindsRawForegroundRequestWhenSequentialJobsSwitchModelFamilies() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let registry = CoreMLModelRegistry(initialComputeUnits: .cpuAndGPU, availabilityPolicy: .requireProductionModels)
        let executor = try makeExecutor(directory: directory, registry: registry)
        for mode: BackboneModelKind in [.nafnetNormal, .cgnet, .nafnetNormal, .cgnetEnsemble, .nafnetNormal] {
            try await enterJobUntilMissingSource(executor, directory: directory, mode: mode)
            let expected: MLComputeUnits = mode == .nafnetNormal ? .cpuAndNeuralEngine : cgNetUnits(.cpuAndGPU)
            await assertRegistry(registry, raw: .cpuAndGPU, effective: expected)
            let description = await executor.currentRequestedComputeDescriptionForTesting()
            XCTAssertEqual(description, describe(expected))
        }
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty,
            "Routing checks must not create a canvas or checkpoint for a missing source")
    }

    func testBackgroundAndExplicitCPUOnlyRemainSafeAcrossFamilyChangesAndForegroundRestoresCGNet() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let registry = CoreMLModelRegistry(initialComputeUnits: .all, availabilityPolicy: .requireProductionModels)
        let executor = try makeExecutor(directory: directory, registry: registry)
        try await enterJobUntilMissingSource(executor, directory: directory, mode: .cgnet)
        await assertRegistry(registry, raw: .all, effective: cgNetUnits(.all))
        await executor.setBackgroundExecutionMode(true)
        await assertRegistry(registry, raw: .cpuAndNeuralEngine, effective: cgNetUnits(.cpuAndNeuralEngine))
        for mode: BackboneModelKind in [.nafnetNormal, .cgnetEnsemble] {
            try await enterJobUntilMissingSource(executor, directory: directory, mode: mode)
            let expected: MLComputeUnits = mode == .nafnetNormal ? .cpuAndNeuralEngine : cgNetUnits(.cpuAndNeuralEngine)
            await assertRegistry(registry, raw: .cpuAndNeuralEngine, effective: expected)
            XCTAssertFalse(expected == .all || expected == .cpuAndGPU, "Background requests must never allow GPU work")
        }
        await executor.setBackgroundExecutionMode(false)
        await assertRegistry(registry, raw: .all, effective: cgNetUnits(.all))
        await executor.switchExecutionTier(.cpuOnly)
        for mode: BackboneModelKind in [.cgnet, .nafnetNormal, .cgnetEnsemble] {
            try await enterJobUntilMissingSource(executor, directory: directory, mode: mode)
            await assertRegistry(registry, raw: .cpuOnly, effective: .cpuOnly)
        }
        // A new explicit foreground request supersedes CPU-only; model switching by itself did not.
        await executor.switchExecutionTier(.gpuAndANE)
        await assertRegistry(registry, raw: .all, effective: cgNetUnits(.all))
    }

    func testQueuedSceneTransitionsDrainInOrderAndNewestRequestWins() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let registry = CoreMLModelRegistry(initialComputeUnits: .all, availabilityPolicy: .requireProductionModels)
        let executor = try makeExecutor(directory: directory, registry: registry)
        try await enterJobUntilMissingSource(executor, directory: directory, mode: .cgnetEnsemble)
        let background = await executor.beginBackgroundExecutionMode(true)
        let foreground = await executor.beginBackgroundExecutionMode(false)
        await background.value
        await foreground.value
        await assertRegistry(registry, raw: .all, effective: cgNetUnits(.all))
        var description = await executor.currentRequestedComputeDescriptionForTesting()
        XCTAssertEqual(description, describe(cgNetUnits(.all)))
        let foregroundAgain = await executor.beginBackgroundExecutionMode(false)
        let backgroundAgain = await executor.beginBackgroundExecutionMode(true)
        await foregroundAgain.value
        await backgroundAgain.value
        await assertRegistry(registry, raw: .cpuAndNeuralEngine, effective: cgNetUnits(.cpuAndNeuralEngine))
        description = await executor.currentRequestedComputeDescriptionForTesting()
        XCTAssertEqual(description, describe(cgNetUnits(.cpuAndNeuralEngine)))
    }

    func testResumeRoutesRecordedEnsembleRatherThanNextJobModelAndUnknownNewJobsCannotChangePolicy() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let registry = CoreMLModelRegistry(initialComputeUnits: .all, availabilityPolicy: .requireProductionModels)
        let executor = try makeExecutor(directory: directory, registry: registry)
        let original = DenoiseJobConfig(sourceImageURL: directory.appendingPathComponent("missing.png"), backbone: .cgnetEnsemble)
        let checkpoint = JobCheckpointState(jobID: original.id, config: original,
            tileConfig: TileConfiguration(tier: .tierB_ConstrainedMemory, backbone: original.backbone),
            sourceWidth: 64, sourceHeight: 48, paddedSourceWidth: 64, paddedSourceHeight: 48,
            outputWidth: 64, outputHeight: 48, globalMaxLuminance: 1, globalMaxLinearComponent: 1,
            didRunDenoise: true, sourceMMapFileName: "source.bin", outputMMapFileName: "output.bin",
            metadataPacketFileName: "metadata.json", lastCompletedTileIndex: -1, totalTiles: 1,
            normalizationPeakVersion: JobCheckpointState.currentNormalizationPeakVersion)
        var nextSelection = original; nextSelection.backbone = .nafnetNormal
        do {
            _ = try await executor.executeOrResumeJob(config: nextSelection, existingCheckpoint: checkpoint)
            XCTFail("The intentionally absent source sidecar must stop this routing-only test")
        } catch let error as ScalerError {
            guard case .fileNotFound = error else { throw error }
        }
        await assertRegistry(registry, raw: .all, effective: cgNetUnits(.all))
        var unknown = original; unknown.backbone = BackboneModelKind(rawValue: "future-opaque-denoiser")
        do {
            _ = try await executor.executeOrResumeJob(config: unknown)
            XCTFail("Unknown models must be refused before changing execution state")
        } catch let error as ScalerError {
            guard case .modelInferenceFailed = error else { throw error }
        }
        await assertRegistry(registry, raw: .all, effective: cgNetUnits(.all))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
    }

    func testNewerRequestRevisionRefusesStaleEntryDiagnosticPublicationEvenIfPolicyMatches() {
        let policy = RequestedComputePolicy("initial")
        let foreground = policy.setFromRequest("foreground")
        _ = policy.setFromRequest("background")
        policy.publishIfRevisionUnchanged("stale foreground snapshot", expectedRevision: foreground)
        XCTAssertEqual(policy.description, "background")
        let current = policy.currentRevision
        policy.publishIfRevisionUnchanged("confirmed background", expectedRevision: current)
        XCTAssertEqual(policy.description, "confirmed background")
        // Equal policy strings still represent a newer request and must invalidate an old read.
        _ = policy.setFromRequest("confirmed background")
        policy.publishIfRevisionUnchanged("stale registry value", expectedRevision: current)
        XCTAssertEqual(policy.description, "confirmed background")
    }

    private func enterJobUntilMissingSource(_ executor: TiledPipelineExecutor, directory: URL,
        mode: BackboneModelKind) async throws {
        let missing = directory.appendingPathComponent("missing-\(UUID()).png")
        let config = DenoiseJobConfig(sourceImageURL: missing, backbone: mode)
        do {
            _ = try await executor.executeOrResumeJob(config: config, overrideTier: .tierB_ConstrainedMemory)
            XCTFail("An intentionally missing source must prevent inference")
        } catch let error as ScalerError {
            guard case .fileNotFound(let url) = error else { throw error }
            XCTAssertEqual(url, missing)
        }
    }

    private func assertRegistry(_ registry: CoreMLModelRegistry, raw: MLComputeUnits, effective: MLComputeUnits,
        file: StaticString = #filePath, line: UInt = #line) async {
        let actualRaw = await registry.currentRequestedUnits()
        let actualEffective = await registry.currentUnits()
        XCTAssertEqual(actualRaw, raw, file: file, line: line)
        XCTAssertEqual(actualEffective, effective, file: file, line: line)
    }

    private func cgNetUnits(_ requested: MLComputeUnits) -> MLComputeUnits {
        #if targetEnvironment(simulator)
        return .cpuOnly
        #else
        return requested
        #endif
    }

    private func describe(_ units: MLComputeUnits) -> String {
        switch units {
        case .all: return "all (CPU+GPU+ANE requested)"
        case .cpuAndGPU: return "cpuAndGPU requested"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine requested"
        case .cpuOnly: return "cpuOnly"
        @unknown default: return "unknown"
        }
    }

    private func makeExecutor(directory: URL, registry: CoreMLModelRegistry) throws -> TiledPipelineExecutor {
        TiledPipelineExecutor(workingDirectory: directory, modelRegistry: registry,
            checkpointStore: try JobCheckpointStore(workingDirectory: directory))
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("CGNetComputePolicy-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
