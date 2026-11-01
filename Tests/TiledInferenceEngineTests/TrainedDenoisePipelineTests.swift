import Foundation
import XCTest
import CoreML
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import CoreIngestion
import CoreExport
import MMapCanvas
import MetalColorEngine
import TruongNoiserCore
@testable import TiledInferenceEngine

private final class DenoiseProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [PipelineProgressUpdate] = []
    func record(_ value: PipelineProgressUpdate) { lock.lock(); storage.append(value); lock.unlock() }
    var updates: [PipelineProgressUpdate] { lock.lock(); defer { lock.unlock() }; return storage }
}

private struct DenoiseTransactionInterruption: Error, Sendable {}

final class TrainedDenoisePipelineTests: XCTestCase, @unchecked Sendable {
    func testBothRealTrainedShapesProduceFiniteNonIdentityNativeResolutionResults() async throws {
        for tier in [DeviceTier.tierB_ConstrainedMemory, .tierA_HighMemory] {
            let directory = try temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            let input = try officialFixture()
            let config = DenoiseJobConfig(sourceImageURL: input)
            let recorder = DenoiseProgressRecorder()
            let registry = productionRegistry()
            let discovered = await registry.discoveredProductionModelURLs()
            XCTAssertEqual(Set(discovered.keys), Set(CoreMLModelIdentifier.allCases))
            let identifier = CoreMLModelIdentifier.denoiser(tier: tier)
            let modelURL = try XCTUnwrap(discovered[identifier])
            XCTAssertEqual(modelURL.pathExtension, "mlmodelc", "Exercise the compiled production asset, not a fallback")
            let executor = try makeExecutor(directory: directory, registry: registry)
            let artifacts = try await executor.executeOrResumeJob(config: config, overrideTier: tier,
                                                                   progressHandler: recorder.record)
            XCTAssertEqual(artifacts.sourceBuffer.width, 256)
            XCTAssertEqual(artifacts.sourceBuffer.height, 256)
            XCTAssertEqual(artifacts.outputBuffer.width, 256)
            XCTAssertEqual(artifacts.outputBuffer.height, 256)
            XCTAssertEqual(artifacts.checkpoint.config.backbone, .nafnetNormal)
            XCTAssertTrue(artifacts.checkpoint.didRunDenoise)
            XCTAssertEqual(artifacts.checkpoint.lastCompletedTileIndex, artifacts.checkpoint.totalTiles - 1)
            XCTAssertGreaterThan(try meanAbsoluteChange(artifacts), 0.0001, "Real noisy-photo inference must not be identity")
            let commits = recorder.updates.filter { $0.phase == .tileCommitted }
            XCTAssertEqual(commits.count, artifacts.checkpoint.totalTiles)
            XCTAssertTrue(commits.allSatisfy { $0.modelName == identifier.rawValue && $0.didRunDenoise == true })
            XCTAssertTrue(commits.contains { ($0.timings?.modelSeconds ?? 0) > 0 })
            XCTAssertEqual(recorder.updates.last?.phase, .completed)

            let destination = directory.appendingPathComponent("real-result.tiff")
            let export = try FormatAwareExporter.exportImageDetailed(artifacts: artifacts,
                destinationURL: destination, overrideFormat: .tiff16BitLZW)
            XCTAssertEqual(export.actualScaleFactor, 1)
            let reopened = try UniversalStripDecoder.decodeToMMap(imageURL: destination,
                destinationMMapURL: directory.appendingPathComponent("reopened.bin"))
            XCTAssertEqual(reopened.decodedWidth, 256)
            XCTAssertEqual(reopened.decodedHeight, 256)
            XCTAssertTrue(reopened.metadata.isAlreadyDenoisedByNoiser)
            XCTAssertFalse(reopened.metadata.isAlreadyUpscaledBySuite)
            XCTAssertEqual(reopened.metadata.processingHistory?.steps.last?.app, "com.iantruong.truongnoiser")
            XCTAssertEqual(reopened.metadata.processingHistory?.steps.last?.operation, "denoise")
            XCTAssertEqual(reopened.metadata.processingHistory?.steps.last?.inputWidth, 256)
            XCTAssertEqual(reopened.metadata.processingHistory?.steps.last?.outputWidth, 256)
        }
    }

    func testRequiredMissingModelErrorsWhileExplicitTestFallbackIsIdentity() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let tensor = [Float16](repeating: 0.3, count: 256 * 256 * 3)
        let required = CoreMLModelRegistry(initialComputeUnits: .cpuOnly, availabilityPolicy: .requireProductionModels,
                                           searchDirectoriesOverride: [directory])
        do {
            _ = try await predict(required, tensor: tensor)
            XCTFail("A production job must not substitute identity for missing denoise weights")
        } catch let error as ScalerError {
            guard case .modelNotFound(let message) = error else { throw error }
            XCTAssertTrue(message.contains("NAFNet_w64_256"))
            XCTAssertTrue(message.contains(directory.path), "The missing-model failure must identify searched locations")
        }
        let explicitTestOnly = CoreMLModelRegistry(initialComputeUnits: .cpuOnly,
            availabilityPolicy: .allowDeterministicFallback, searchDirectoriesOverride: [directory])
        let fallback = try await predict(explicitTestOnly, tensor: tensor)
        XCTAssertEqual(fallback, tensor)
        XCTAssertEqual(required.availabilityPolicy, .requireProductionModels)
    }

    func testProductionRejectsMalformedAndNonFiniteInputBeforeModelDiscovery() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let registry = CoreMLModelRegistry(initialComputeUnits: .cpuOnly, availabilityPolicy: .requireProductionModels,
                                           searchDirectoriesOverride: [directory])
        var tensor = [Float16](repeating: 0.3, count: 256 * 256 * 3)
        for bad in [Float16.nan, .infinity, -0.1, 1.1] {
            tensor[900] = bad
            do {
                _ = try await predict(registry, tensor: tensor)
                XCTFail("Invalid production training-domain input must be refused")
            } catch let error as ScalerError {
                guard case .modelInferenceFailed = error else { throw error }
            }
        }
        do {
            _ = try await predict(registry, tensor: [0.3])
            XCTFail("A malformed tensor must be refused")
        } catch let error as ScalerError {
            guard case .modelInferenceFailed = error else { throw error }
        }
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
    }

    func testPauseResumeKeepsRecordedModelStrengthsAndMatchesAnUninterruptedRun() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let input = try makeRepeatedOfficialImage(width: 480, height: 320, directory: directory)
        let work = directory.appendingPathComponent("paused", isDirectory: true)
        let config = DenoiseJobConfig(sourceImageURL: input, lumaStrength: 0.55, chromaStrength: 0.85)
        let store = try JobCheckpointStore(workingDirectory: work)
        let executor = TiledPipelineExecutor(workingDirectory: work, modelRegistry: productionRegistry(), checkpointStore: store)
        await executor.requestPauseAfterTileForTesting(0)
        do {
            _ = try await executor.executeOrResumeJob(config: config, overrideTier: .tierB_ConstrainedMemory)
            XCTFail("The explicit pause must stop at its committed tile boundary")
        } catch let error as ScalerError {
            guard case .jobPausedForBackgroundExpiration(let completed) = error else { throw error }
            XCTAssertEqual(completed, 0)
        }
        let saved = try await store.load(jobID: config.id)
        XCTAssertEqual(saved.lastCompletedTileIndex, 0)
        XCTAssertGreaterThan(saved.totalTiles, 1)
        XCTAssertTrue(try journalNames(work).isEmpty)
        // A newly chosen model/recipe is not permission to relabel an already committed canvas.
        var otherSelection = config
        otherSelection.backbone = .restormerAstro
        otherSelection.lumaStrength = 0.1
        otherSelection.chromaStrength = 0.2
        let resumed = try await makeExecutor(directory: work).executeOrResumeJob(config: otherSelection,
            existingCheckpoint: saved, overrideTier: .tierA_HighMemory)
        XCTAssertEqual(resumed.checkpoint.config, config)
        XCTAssertEqual(resumed.checkpoint.tileConfig.tier, .tierB_ConstrainedMemory)
        XCTAssertEqual(resumed.checkpoint.lastCompletedTileIndex, saved.totalTiles - 1)
        let baseline = try await makeExecutor(directory: directory.appendingPathComponent("baseline"))
            .executeOrResumeJob(config: config, overrideTier: .tierB_ConstrainedMemory)
        XCTAssertEqual(snapshot(resumed.outputBuffer), snapshot(baseline.outputBuffer))
    }

    func testIncompatibleCheckpointIsRefusedBeforeAnySavedBytesChange() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let input = try makeRepeatedOfficialImage(width: 480, height: 320, directory: directory)
        let work = directory.appendingPathComponent("job", isDirectory: true)
        let config = DenoiseJobConfig(sourceImageURL: input)
        let store = try JobCheckpointStore(workingDirectory: work)
        let executor = TiledPipelineExecutor(workingDirectory: work, modelRegistry: productionRegistry(), checkpointStore: store)
        await executor.requestPauseAfterTileForTesting(0)
        do { _ = try await executor.executeOrResumeJob(config: config, overrideTier: .tierB_ConstrainedMemory) }
        catch let error as ScalerError {
            guard case .jobPausedForBackgroundExpiration = error else { throw error }
        }
        let saved = try await store.load(jobID: config.id)
        let before = try fileSnapshot(work)
        for change in ["normalization", "model", "scale"] {
            var json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(saved)) as? [String: Any])
            if change == "normalization" { json["normalizationPeakVersion"] = 2 }
            else {
                var cfg = try XCTUnwrap(json["config"] as? [String: Any])
                if change == "model" { cfg["backbone"] = "unknown-future-model" }
                else { cfg["scaleFactor"] = 2 }
                json["config"] = cfg
            }
            let incompatible = try JSONDecoder().decode(JobCheckpointState.self,
                from: JSONSerialization.data(withJSONObject: json))
            do {
                _ = try await makeExecutor(directory: work).executeOrResumeJob(config: config,
                                                                               existingCheckpoint: incompatible)
                XCTFail("Incompatible \(change) must be rejected before opening or migrating the canvas")
            } catch let error as ScalerError {
                guard case .modelInferenceFailed = error else { throw error }
            }
            XCTAssertEqual(try fileSnapshot(work), before)
        }
    }

    func testInterruptedRealTileReplaysUndoJournalExactlyOnce() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let input = try makeRepeatedOfficialImage(width: 480, height: 320, directory: directory)
        let config = DenoiseJobConfig(sourceImageURL: input)
        let work = directory.appendingPathComponent("interrupted", isDirectory: true)
        let store = try JobCheckpointStore(workingDirectory: work)
        let executor = TiledPipelineExecutor(workingDirectory: work, modelRegistry: productionRegistry(), checkpointStore: store)
        await executor.setTransactionInterruptionHookForTesting { point in
            if point == .afterCanvasFlushBeforeCheckpoint { throw DenoiseTransactionInterruption() }
        }
        do {
            _ = try await executor.executeOrResumeJob(config: config, overrideTier: .tierB_ConstrainedMemory)
            XCTFail("The injected transaction interruption must abort")
        } catch is DenoiseTransactionInterruption { }
        let saved = try await store.load(jobID: config.id)
        XCTAssertEqual(saved.lastCompletedTileIndex, -1)
        XCTAssertEqual(try journalNames(work).count, 2)
        let resumed = try await makeExecutor(directory: work).executeOrResumeJob(config: config, existingCheckpoint: saved)
        XCTAssertTrue(try journalNames(work).isEmpty)
        let baseline = try await makeExecutor(directory: directory.appendingPathComponent("baseline"))
            .executeOrResumeJob(config: config, overrideTier: .tierB_ConstrainedMemory)
        XCTAssertEqual(snapshot(resumed.outputBuffer), snapshot(baseline.outputBuffer))
    }

    func testOptInTwelveMegapixelMacPipelineBenchmark() async throws {
        guard ProcessInfo.processInfo.environment["TRUONGNOISER_RUN_12MP_BENCHMARK"] == "1" else {
            throw XCTSkip("Opt-in: repeated official sample, 3024×4032 Mac pipeline benchmark")
        }
        #if os(macOS)
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let input = try makeRepeatedOfficialImage(width: 3024, height: 4032, directory: directory)
        let model = ProcessInfo.processInfo.environment["TRUONGNOISER_BENCHMARK_MODEL"]
            .map(BackboneModelKind.init(rawValue:)) ?? .nafnetNormal
        guard model.isSupported else { throw ScalerError.modelNotFound(modelName: model.rawValue) }
        let config = DenoiseJobConfig(sourceImageURL: input, backbone: model)
        let recorder = DenoiseProgressRecorder()
        let start = ContinuousClock.now
        let registry = CoreMLModelRegistry(initialComputeUnits: .all,
            customSearchDirectories: [repositoryRoot().appendingPathComponent("Targets/TruongNoiserApp/Models")],
            availabilityPolicy: .requireProductionModels)
        let artifacts = try await makeExecutor(directory: directory.appendingPathComponent("work"), registry: registry)
            .executeOrResumeJob(config: config, overrideTier: .tierA_HighMemory, progressHandler: recorder.record)
        let processingSeconds = seconds(start.duration(to: .now))
        let exportStart = ContinuousClock.now
        let result = try FormatAwareExporter.exportImageDetailed(artifacts: artifacts,
            destinationURL: directory.appendingPathComponent("benchmark.tiff"), overrideFormat: .tiff16BitLZW)
        let exportSeconds = seconds(exportStart.duration(to: .now))
        XCTAssertEqual(artifacts.outputBuffer.width, 3024)
        XCTAssertEqual(artifacts.outputBuffer.height, 4032)
        XCTAssertEqual(result.actualScaleFactor, 1)
        let updates = recorder.updates
        let commits = updates.filter { $0.phase == .tileCommitted }
        XCTAssertEqual(commits.count, artifacts.checkpoint.totalTiles)
        let frozenModel = artifacts.checkpoint.config.backbone
        let executionName = try CoreMLModelIdentifier.executionName(backbone: frozenModel,
            tier: artifacts.checkpoint.tileConfig.tier)
        let tileReports: [[String: Any]] = commits.map { update in
            ["tile": update.currentTileIndex ?? -1, "elapsedSeconds": update.elapsedSeconds ?? 0,
             "timings": timingsDictionary(update.timings)]
        }
        let report: [String: Any] = [
            "scope": "Mac production required-model pipeline, repeated official noisy.png; not a real DNG or phone benchmark",
            "source": "Official NAFNet noisy demo repeated without resizing to 3024×4032",
            "sourceWidth": 3024, "sourceHeight": 4032, "outputWidth": 3024, "outputHeight": 4032,
            "model": executionName,
            "mode": frozenModel.rawValue,
            "predictionsPerTile": frozenModel.ensemblePassCount,
            "requestedCompute": commits.last?.requestedCompute ?? "unknown",
            "actualANEPlacement": "NOT_MEASURED", "phoneSeconds": "NOT_MEASURED", "physicalImageQuality": "NOT_MEASURED",
            "lumaStrength": config.lumaStrength, "chromaStrength": config.chromaStrength,
            "processingSeconds": processingSeconds, "exportSeconds": exportSeconds,
            "totalSeconds": processingSeconds + exportSeconds, "tiles": artifacts.checkpoint.totalTiles,
            "timingSamples": tileReports,
            "phaseEvents": updates.map { ["phase": $0.phase.rawValue, "elapsedSeconds": $0.elapsedSeconds ?? 0] as [String: Any] }
        ]
        let defaultPath = repositoryRoot().appendingPathComponent(
            "build/model-validation/" + executionName.lowercased() + "-12mp-pipeline.json").path
        let path = ProcessInfo.processInfo.environment["TRUONGNOISER_BENCHMARK_REPORT"] ?? defaultPath
        let reportURL = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: reportURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: reportURL, options: .atomic)
        print("\(frozenModel.rawValue) 12MP repeated-sample Mac benchmark: \(processingSeconds)s pipeline + \(exportSeconds)s TIFF; phone unmeasured")
        #else
        throw XCTSkip("This opt-in benchmark records Mac measurements only")
        #endif
    }

    private func productionRegistry() -> CoreMLModelRegistry {
        CoreMLModelRegistry(initialComputeUnits: .cpuOnly,
            customSearchDirectories: [repositoryRoot().appendingPathComponent("Targets/TruongNoiserApp/Models")],
            availabilityPolicy: .requireProductionModels)
    }

    private func makeExecutor(directory: URL, registry: CoreMLModelRegistry? = nil) throws -> TiledPipelineExecutor {
        TiledPipelineExecutor(workingDirectory: directory, modelRegistry: registry ?? productionRegistry(),
                              checkpointStore: try JobCheckpointStore(workingDirectory: directory))
    }

    private func predict(_ registry: CoreMLModelRegistry, tensor: [Float16]) async throws -> [Float16] {
        try await registry.predict(modelID: .nafnet256, inputPlanarFP16: tensor,
            inputHeight: 256, inputWidth: 256, outputHeight: 256, outputWidth: 256, tileIndex: 0)
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("NoiserPipeline-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func repositoryRoot() -> URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    private func officialFixture() throws -> URL {
        try XCTUnwrap(Bundle.module.url(forResource: "noisy", withExtension: "png", subdirectory: "Fixtures"))
    }

    private func makeRepeatedOfficialImage(width: Int, height: Int, directory: URL) throws -> URL {
        let imageSource = try XCTUnwrap(CGImageSourceCreateWithURL(try officialFixture() as CFURL, nil))
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(imageSource, 0, nil))
        let colorSpace = try XCTUnwrap(CGColorSpace(name: CGColorSpace.sRGB))
        let context = try XCTUnwrap(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.interpolationQuality = .none
        for y in stride(from: 0, to: height, by: image.height) {
            for x in stride(from: 0, to: width, by: image.width) {
                context.draw(image, in: CGRect(x: x, y: y, width: image.width, height: image.height))
            }
        }
        let destinationURL = directory.appendingPathComponent("repeated-official-sample.png")
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(destinationURL as CFURL,
            UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, try XCTUnwrap(context.makeImage()), nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return destinationURL
    }

    private func meanAbsoluteChange(_ artifacts: CompletedPipelineArtifacts) throws -> Double {
        var sum = 0.0
        for y in 0..<artifacts.outputBuffer.height {
            for x in 0..<artifacts.outputBuffer.width {
                let original = artifacts.sourceBuffer.unsafePixelPointer(x: x, y: y)
                let result = artifacts.outputBuffer.unsafePixelPointer(x: x, y: y)
                for channel in 0..<4 { XCTAssertTrue(result[channel].isFinite) }
                XCTAssertEqual(result[3], original[3])
                for channel in 0..<3 { sum += Double(abs(Float(result[channel]) - Float(original[channel]))) }
            }
        }
        return sum / Double(artifacts.outputBuffer.width * artifacts.outputBuffer.height * 3)
    }

    private func snapshot(_ buffer: MMapBuffer) -> [UInt16] {
        (0..<buffer.height).flatMap { y in
            (0..<buffer.width).flatMap { x in (0..<4).map { buffer.unsafePixelPointer(x: x, y: y)[$0].bitPattern } }
        }
    }

    private func journalNames(_ directory: URL) throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: directory.path).filter { $0.hasPrefix("journal_") }
    }

    private func fileSnapshot(_ directory: URL) throws -> [String: Data] {
        let urls = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.isRegularFileKey])
        return try Dictionary(uniqueKeysWithValues: urls.filter { try $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true }
            .map { ($0.lastPathComponent, try Data(contentsOf: $0)) })
    }

    private func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    private func timingsDictionary(_ timing: PipelineTileTimings?) -> [String: Double] {
        guard let timing else { return [:] }
        let candidates: [(String, Double?)] = [
            ("extractionSeconds", timing.extractionSeconds), ("modelLoadSeconds", timing.modelLoadSeconds),
            ("modelSeconds", timing.modelSeconds), ("stabilitySeconds", timing.stabilitySeconds),
            ("reconstructionSeconds", timing.reconstructionSeconds), ("journalSeconds", timing.journalSeconds),
            ("blendSeconds", timing.blendSeconds), ("flushSeconds", timing.flushSeconds),
            ("checkpointSeconds", timing.checkpointSeconds), ("cleanupSeconds", timing.cleanupSeconds)
        ]
        return Dictionary(uniqueKeysWithValues: candidates.compactMap { name, value in value.map { (name, $0) } })
    }
}
