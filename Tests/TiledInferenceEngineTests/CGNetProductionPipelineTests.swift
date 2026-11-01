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

private final class CGNetPipelineEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var progress: [PipelineProgressUpdate] = []
    private var passes: [Int] = []
    func record(_ update: PipelineProgressUpdate) { lock.withLock { progress.append(update) } }
    func recordPass(_ pass: Int) { lock.withLock { passes.append(pass) } }
    var updates: [PipelineProgressUpdate] { lock.withLock { progress } }
    var completedPasses: [Int] { lock.withLock { passes } }
}

/// These tests execute shipped trained weights, not the deterministic fallback. The noisy-photo
/// fixture is upstream's demo, not a clean ground truth or a Lightroom/DxO comparison. Mac CPU
/// request timings and tile-shape differences cannot establish phone quality or ANE placement.
final class CGNetProductionPipelineTests: XCTestCase, @unchecked Sendable {
    func testBothCGNetModesAndCompiledShapesRunNativeResolutionAndExportActualModelHistory() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let input = try photographicFixture(width: 96, height: 72, directory: directory)
        let originalBytes = try Data(contentsOf: input)
        let registry = productionRegistry()
        let discovered = await registry.discoveredProductionModelURLs()
        var measurements: [[String: Any]] = []
        for mode: BackboneModelKind in [.cgnet, .cgnetEnsemble] {
            for tier: DeviceTier in [.tierB_ConstrainedMemory, .tierA_HighMemory] {
                let identifier = try CoreMLModelIdentifier.denoiser(backbone: mode, tier: tier)
                XCTAssertEqual(try XCTUnwrap(discovered[identifier]).pathExtension, "mlmodelc")
                let name = try CoreMLModelIdentifier.executionName(backbone: mode, tier: tier)
                let work = directory.appendingPathComponent(name, isDirectory: true)
                let events = CGNetPipelineEvents()
                let executor = try makeExecutor(directory: work, registry: registry)
                await executor.setEnsemblePassCompletionHookForTesting { events.recordPass($0) }
                let config = DenoiseJobConfig(sourceImageURL: input, backbone: mode,
                    lumaStrength: 0.8, chromaStrength: 1)
                let started = ContinuousClock.now
                let artifacts = try await executor.executeOrResumeJob(config: config, overrideTier: tier,
                    progressHandler: events.record)
                let elapsed = seconds(started.duration(to: .now))
                XCTAssertEqual(artifacts.checkpoint.config, config)
                XCTAssertEqual(artifacts.checkpoint.tileConfig, TileConfiguration(tier: tier, backbone: mode))
                XCTAssertEqual(artifacts.checkpoint.tileConfig.inputOverlap, 32)
                XCTAssertEqual(artifacts.sourceBuffer.width, 96)
                XCTAssertEqual(artifacts.sourceBuffer.height, 72)
                XCTAssertEqual(artifacts.outputBuffer.width, 96)
                XCTAssertEqual(artifacts.outputBuffer.height, 72)
                XCTAssertEqual(artifacts.checkpoint.totalTiles, 1)
                XCTAssertEqual(artifacts.checkpoint.lastCompletedTileIndex, 0)
                XCTAssertTrue(artifacts.checkpoint.didRunDenoise)
                let change = assertFiniteNativeResult(artifacts)
                XCTAssertGreaterThan(change, 0.00001, "A noisy photograph must use real, non-identity inference")
                let commits = events.updates.filter { $0.phase == .tileCommitted }
                XCTAssertEqual(commits.count, 1)
                XCTAssertEqual(commits.first?.modelName, name)
                XCTAssertGreaterThan(commits.first?.timings?.modelSeconds ?? 0, 0)
                XCTAssertEqual(events.updates.last?.phase, .completed)
                XCTAssertEqual(events.completedPasses, mode == .cgnetEnsemble ? Array(1...8) : [])
                for (format, ext) in [(ExportFormat.tiff16BitLZW, "tiff"), (.linearDNG16Bit, "dng")] {
                    let destination = work.appendingPathComponent("result." + ext)
                    let result = try FormatAwareExporter.exportImageDetailed(artifacts: artifacts,
                        destinationURL: destination, overrideFormat: format)
                    XCTAssertEqual(result.actualScaleFactor, 1)
                    let reopened = try UniversalStripDecoder.decodeToMMap(imageURL: destination,
                        destinationMMapURL: work.appendingPathComponent("readback-" + ext + ".bin"))
                    XCTAssertEqual(reopened.decodedWidth, 96)
                    XCTAssertEqual(reopened.decodedHeight, 72)
                    XCTAssertTrue(reopened.metadata.isAlreadyDenoisedByNoiser)
                    XCTAssertFalse(reopened.metadata.isAlreadyUpscaledBySuite)
                    try assertHistory(reopened.metadata, artifacts: artifacts, prior: [])
                    assertPixels(reopened.sourceBuffer, equalTo: artifacts.outputBuffer, accuracy: 0.004)
                    try retainImageIfRequested(destination, name: name + "." + ext)
                }
                measurements.append(["mode": mode.rawValue, "asset": identifier.rawValue,
                    "inputSize": identifier.inputTileSize, "predictionsPerTile": mode.ensemblePassCount,
                    "tiles": artifacts.checkpoint.totalTiles, "pipelineSeconds": elapsed,
                    "meanAbsoluteLinearRGBChange": change])
            }
        }
        XCTAssertEqual(try Data(contentsOf: input), originalBytes)
        try writeReport("cgnet-native-modes", values: ["runs": measurements,
            "fixture": "96×72 unscaled crop of the official noisy photographic demo"])
    }

    func testBothCGNetModesRequireWeightsAndNeverCommitIdentityForAMissingAsset() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let input = try photographicFixture(width: 48, height: 32, directory: directory)
        let absent = directory.appendingPathComponent("no-models", isDirectory: true)
        try FileManager.default.createDirectory(at: absent, withIntermediateDirectories: true)
        for mode: BackboneModelKind in [.cgnet, .cgnetEnsemble] {
            let registry = CoreMLModelRegistry(initialComputeUnits: .cpuOnly,
                availabilityPolicy: .requireProductionModels, searchDirectoriesOverride: [absent])
            let work = directory.appendingPathComponent(mode.rawValue, isDirectory: true)
            let store = try JobCheckpointStore(workingDirectory: work)
            let executor = TiledPipelineExecutor(workingDirectory: work, modelRegistry: registry, checkpointStore: store)
            let config = DenoiseJobConfig(sourceImageURL: input, backbone: mode)
            let events = CGNetPipelineEvents()
            do {
                _ = try await executor.executeOrResumeJob(config: config, overrideTier: .tierB_ConstrainedMemory,
                    progressHandler: events.record)
                XCTFail("Required CGNet weights may not become an identity prediction")
            } catch let error as ScalerError {
                guard case .modelNotFound(let message) = error else { throw error }
                XCTAssertTrue(message.contains("CGNet_SIDD_256"))
            }
            let checkpoint = try await store.load(jobID: config.id)
            XCTAssertEqual(checkpoint.lastCompletedTileIndex, -1)
            XCTAssertFalse(events.updates.contains { $0.phase == .tileCommitted })
            XCTAssertTrue(try journalNames(work).isEmpty)
            let output = try Data(contentsOf: work.appendingPathComponent(checkpoint.outputMMapFileName))
            XCTAssertTrue(output.allSatisfy { $0 == 0 })
        }
    }

    func testActualCGNetHDRAlphaAndColorSurviveProcessingTIFFReopenAndPriorHistory() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let input = try hdrAlphaFixture(directory: directory)
        let originalBytes = try Data(contentsOf: input)
        let prior: [ImageProcessingHistory.Step] = [
            .init(id: UUID(), app: "com.iantruong.truongscaler", operation: "upscale", model: "EDSR_x2", scale: 2,
                inputWidth: 16, inputHeight: 12, outputWidth: 32, outputHeight: 24),
            .init(id: UUID(), app: "com.iantruong.truongnoiser", operation: "denoise", model: "opaque-legacy-denoiser",
                inputWidth: 32, inputHeight: 24, outputWidth: 32, outputHeight: 24)
        ]
        let registry = productionRegistry()
        var measurements: [[String: Any]] = []
        for mode: BackboneModelKind in [.cgnet, .cgnetEnsemble] {
            let work = directory.appendingPathComponent(mode.rawValue, isDirectory: true)
            let config = DenoiseJobConfig(sourceImageURL: input, backbone: mode,
                lumaStrength: 0.35, chromaStrength: 0.65, sessionState: .init(
                    isDenoised: true, isUpscaled: true, upscaleFactor: 2,
                    isDenoisedByNoiser: true, processingHistory: .init(steps: prior)))
            let artifacts = try await makeExecutor(directory: work, registry: registry)
                .executeOrResumeJob(config: config, overrideTier: .tierB_ConstrainedMemory)
            XCTAssertEqual(artifacts.outputBuffer.width, 32)
            XCTAssertEqual(artifacts.outputBuffer.height, 24)
            XCTAssertTrue(artifacts.metadata.hasAlpha)
            XCTAssertGreaterThan(artifacts.checkpoint.globalMaxLinearComponent, 1)
            let change = assertFiniteNativeResult(artifacts)
            XCTAssertGreaterThan(change, 0.000001)
            XCTAssertGreaterThan(componentPeak(artifacts.outputBuffer), 1, "Denoising must not clip HDR to SDR")
            // The fixture contains a saturated red, semitransparent HDR patch. Its hue must not
            // collapse to neutral after the developed-sRGB residual is restored into linear P3.
            let red = artifacts.outputBuffer.unsafePixelPointer(x: 1, y: 1)
            XCTAssertGreaterThan(Float(red[0]), Float(red[1]) * 1.5)
            XCTAssertGreaterThan(Float(red[0]), Float(red[2]) * 2)
            let sourceBeforeExport = snapshot(artifacts.sourceBuffer)
            let outputBeforeExport = snapshot(artifacts.outputBuffer)
            let destination = work.appendingPathComponent("hdr-alpha.tiff")
            let export = try FormatAwareExporter.exportImageDetailed(artifacts: artifacts,
                destinationURL: destination, overrideFormat: .tiff16BitLZW)
            XCTAssertTrue(export.isHDR)
            XCTAssertEqual(export.actualScaleFactor, 1)
            let reopened = try UniversalStripDecoder.decodeToMMap(imageURL: destination,
                destinationMMapURL: work.appendingPathComponent("readback.bin"))
            XCTAssertEqual(reopened.decodedWidth, 32)
            XCTAssertEqual(reopened.decodedHeight, 24)
            XCTAssertTrue(reopened.metadata.hasAlpha)
            XCTAssertTrue(reopened.metadata.isAlreadyDenoisedByNoiser)
            XCTAssertTrue(reopened.metadata.isAlreadyUpscaledBySuite)
            XCTAssertEqual(reopened.metadata.upscaledByAppFactor, 2, "Keep the earlier Scaler factor, not this job's1×")
            try assertHistory(reopened.metadata, artifacts: artifacts, prior: prior)
            assertPixels(reopened.sourceBuffer, equalTo: artifacts.outputBuffer, accuracy: 0.008)
            XCTAssertEqual(snapshot(artifacts.sourceBuffer), sourceBeforeExport)
            XCTAssertEqual(snapshot(artifacts.outputBuffer), outputBeforeExport)
            try retainImageIfRequested(destination, name: mode.rawValue + "-hdr-alpha.tiff")
            measurements.append(["mode": mode.rawValue, "meanAbsoluteLinearRGBChange": change,
                "sourcePeak": componentPeak(artifacts.sourceBuffer), "outputPeak": componentPeak(artifacts.outputBuffer),
                "historyEvents": reopened.metadata.processingHistory?.steps.count ?? 0])
        }
        XCTAssertEqual(try Data(contentsOf: input), originalBytes)
        try writeReport("cgnet-hdr-alpha", values: ["runs": measurements,
            "fixture": "Synthetic linear-P3 HDR and alpha using photographic sample colors; not a RAW quality ranking"])
    }

    func testTiled256Versus512MeasuresPhotographicDifferencesAndFeatherBoundarySeams() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let input = try photographicFixture(width: 480, height: 160, directory: directory)
        let registry = productionRegistry()
        var measurements: [[String: Any]] = []
        for mode: BackboneModelKind in [.cgnet, .cgnetEnsemble] {
            let config = DenoiseJobConfig(sourceImageURL: input, backbone: mode)
            let tiled = try await makeExecutor(directory: directory.appendingPathComponent(mode.rawValue + "-256"), registry: registry)
                .executeOrResumeJob(config: config, overrideTier: .tierB_ConstrainedMemory)
            let whole = try await makeExecutor(directory: directory.appendingPathComponent(mode.rawValue + "-512"), registry: registry)
                .executeOrResumeJob(config: config, overrideTier: .tierA_HighMemory)
            XCTAssertEqual(tiled.checkpoint.totalTiles, 2)
            XCTAssertEqual(whole.checkpoint.totalTiles, 1)
            _ = assertFiniteNativeResult(tiled)
            _ = assertFiniteNativeResult(whole)
            let differences = differenceMetrics(tiled.outputBuffer, whole.outputBuffer)
            XCTAssertLessThan(differences.rmse, 0.025, "Shapes may differ, but must not yield a large full-image disagreement")
            XCTAssertLessThan(differences.maximum, 0.25, "A bad boundary or outlier must not hide behind the mean")
            // Compare gradients of the shape-difference field at the start/end of the32px feather.
            // Subtracting the512 result removes genuine photographed edges from the seam metric.
            let boundary = residualGradient(tiled.outputBuffer, whole.outputBuffer, columns: [224, 255, 256])
            let controls = residualGradient(tiled.outputBuffer, whole.outputBuffer, columns: [208, 216, 264, 272])
            XCTAssertLessThan(boundary, max(0.003, controls * 4), "Tile boundaries introduce excessive new discontinuity")
            let localMAE = differenceMetrics(tiled.outputBuffer, whole.outputBuffer, xRange: 220..<260).mae
            XCTAssertLessThan(localMAE, max(0.01, differences.mae * 4))
            measurements.append(["mode": mode.rawValue, "sourceWidth": 480, "sourceHeight": 160,
                "smallShapeTiles": 2, "largeShapeTiles": 1, "meanAbsoluteDifference": differences.mae,
                "rmsDifference": differences.rmse, "maximumAbsoluteDifference": differences.maximum,
                "boundaryResidualGradient": boundary, "controlResidualGradient": controls,
                "featherBandMeanAbsoluteDifference": localMAE])
            for (suffix, artifacts) in [("256", tiled), ("512", whole)] {
                if retainImages {
                    let destination = directory.appendingPathComponent(mode.rawValue + "-seams-" + suffix + ".tiff")
                    _ = try FormatAwareExporter.exportImageDetailed(artifacts: artifacts,
                        destinationURL: destination, overrideFormat: .tiff16BitLZW)
                    try retainImageIfRequested(destination, name: destination.lastPathComponent)
                }
            }
        }
        try writeReport("cgnet-tile-seams", values: ["runs": measurements,
            "fixture": "Official noisy photographic demo repeated horizontally without resizing to480×160",
            "interpretation": "Measured tile-shape/context differences and seam regressions; not full-tensor equality or a quality score against clean ground truth"])
    }

    func testStopAfterOneOrientationCommitsNothingThenResumeMatchesUninterruptedAndStrengthsApplyOnce() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let input = try photographicFixture(width: 64, height: 48, directory: directory)
        let config = DenoiseJobConfig(sourceImageURL: input, backbone: .cgnetEnsemble,
            lumaStrength: 0.35, chromaStrength: 0.65)
        let registry = productionRegistry()
        let work = directory.appendingPathComponent("paused", isDirectory: true)
        let store = try JobCheckpointStore(workingDirectory: work)
        let executor = TiledPipelineExecutor(workingDirectory: work, modelRegistry: registry, checkpointStore: store)
        let events = CGNetPipelineEvents()
        await executor.setEnsemblePassCompletionHookForTesting { [weak executor] pass in
            events.recordPass(pass)
            if pass == 1 { await executor?.requestGracefulPause() }
        }
        do {
            _ = try await executor.executeOrResumeJob(config: config, overrideTier: .tierB_ConstrainedMemory,
                progressHandler: events.record)
            XCTFail("Stop must be observed between orientations, before an incomplete tile is committed")
        } catch let error as ScalerError {
            guard case .jobPausedForBackgroundExpiration(let completed) = error else { throw error }
            XCTAssertEqual(completed, -1)
        }
        await executor.setEnsemblePassCompletionHookForTesting(nil)
        XCTAssertEqual(events.completedPasses, [1])
        XCTAssertFalse(events.updates.contains { $0.phase == .tileCommitted || $0.phase == .committingTile })
        XCTAssertEqual(events.updates.last?.phase, .paused)
        let saved = try await store.load(jobID: config.id)
        XCTAssertEqual(saved.lastCompletedTileIndex, -1)
        XCTAssertEqual(saved.totalTiles, 1)
        XCTAssertEqual(saved.config, config)
        XCTAssertTrue(try journalNames(work).isEmpty)
        let canvas = try Data(contentsOf: work.appendingPathComponent(saved.outputMMapFileName))
        XCTAssertEqual(canvas.count, 64 * 48 * 8)
        XCTAssertTrue(canvas.allSatisfy { $0 == 0 }, "An incomplete ensemble must never reach the durable output canvas")
        var nextSelection = config
        nextSelection.backbone = .cgnet
        nextSelection.lumaStrength = 0.9; nextSelection.chromaStrength = 0.1
        let resumed = try await executor.executeOrResumeJob(config: nextSelection,
            existingCheckpoint: saved, overrideTier: .tierA_HighMemory)
        XCTAssertEqual(resumed.checkpoint.config, config)
        XCTAssertEqual(resumed.checkpoint.tileConfig.tier, .tierB_ConstrainedMemory)
        let uninterrupted = try await makeExecutor(directory: directory.appendingPathComponent("baseline"), registry: registry)
            .executeOrResumeJob(config: config, overrideTier: .tierB_ConstrainedMemory)
        XCTAssertEqual(snapshot(resumed.outputBuffer), snapshot(uninterrupted.outputBuffer),
            "Discarded partial averages may not change the resumed result")
        XCTAssertTrue(try journalNames(work).isEmpty)
        let fullConfig = DenoiseJobConfig(sourceImageURL: input, backbone: .cgnetEnsemble,
            lumaStrength: 1, chromaStrength: 1)
        let full = try await makeExecutor(directory: directory.appendingPathComponent("full-strength"), registry: registry)
            .executeOrResumeJob(config: fullConfig, overrideTier: .tierB_ConstrainedMemory)
        var maximumMixError: Float = 0
        for y in 0..<48 {
            for x in 0..<64 {
                let source = full.sourceBuffer.unsafePixelPointer(x: x, y: y)
                let result = full.outputBuffer.unsafePixelPointer(x: x, y: y)
                let mixed = resumed.outputBuffer.unsafePixelPointer(x: x, y: y)
                let delta = SIMD3<Float>(Float(result[0]) - Float(source[0]),
                    Float(result[1]) - Float(source[1]), Float(result[2]) - Float(source[2]))
                let dy = 0.22897456 * delta.x + 0.69173852 * delta.y + 0.07928691 * delta.z
                for c in 0..<3 {
                    let expected = max(0, Float(source[c]) + 0.65 * delta[c] + (0.35 - 0.65) * dy)
                    maximumMixError = max(maximumMixError, abs(Float(mixed[c]) - expected))
                }
                XCTAssertEqual(mixed[3], source[3])
            }
        }
        XCTAssertLessThan(maximumMixError, 0.001,
            "Apply the frozen strengths once to the deoriented ensemble residual, not once per prediction or again on export")
        let destination = work.appendingPathComponent("mixed.tiff")
        _ = try FormatAwareExporter.exportImageDetailed(artifacts: resumed,
            destinationURL: destination, overrideFormat: .tiff16BitLZW)
        let reopened = try UniversalStripDecoder.decodeToMMap(imageURL: destination,
            destinationMMapURL: work.appendingPathComponent("readback.bin"))
        assertPixels(reopened.sourceBuffer, equalTo: resumed.outputBuffer, accuracy: 0.004)
        try assertHistory(reopened.metadata, artifacts: resumed, prior: [])
        try writeReport("cgnet-stop-resume-strengths", values: ["stoppedAfterPredictions": 1,
            "committedTilesAtStop": 0, "resumedMode": resumed.checkpoint.config.backbone.rawValue,
            "resumeExactlyMatchesUninterrupted": snapshot(resumed.outputBuffer) == snapshot(uninterrupted.outputBuffer),
            "maximumIndependentStrengthMixError": maximumMixError])
    }

    private func productionRegistry() -> CoreMLModelRegistry {
        CoreMLModelRegistry(initialComputeUnits: .cpuOnly,
            customSearchDirectories: [repositoryRoot.appendingPathComponent("Targets/TruongNoiserApp/Models")],
            availabilityPolicy: .requireProductionModels)
    }

    private func makeExecutor(directory: URL, registry: CoreMLModelRegistry) throws -> TiledPipelineExecutor {
        TiledPipelineExecutor(workingDirectory: directory, modelRegistry: registry,
            checkpointStore: try JobCheckpointStore(workingDirectory: directory))
    }

    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("CGNetProduction-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func officialImage() throws -> CGImage {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "noisy", withExtension: "png", subdirectory: "Fixtures"))
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
        return try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
    }

    private func photographicFixture(width: Int, height: Int, directory: URL) throws -> URL {
        let image = try officialImage()
        let space = try XCTUnwrap(CGColorSpace(name: CGColorSpace.sRGB))
        let context = try XCTUnwrap(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        context.interpolationQuality = .none
        for y in stride(from: 0, to: height, by: image.height) {
            for x in stride(from: 0, to: width, by: image.width) {
                context.draw(image, in: CGRect(x: x, y: y, width: image.width, height: image.height))
            }
        }
        let url = directory.appendingPathComponent("photo-\(width)x\(height).png")
        try write(try XCTUnwrap(context.makeImage()), type: .png, to: url)
        return url
    }

    private func hdrAlphaFixture(directory: URL) throws -> URL {
        let width = 32, height = 24
        let image = try officialImage()
        let colorSpace = try XCTUnwrap(CGColorSpace(name: CGColorSpace.sRGB))
        let context = try XCTUnwrap(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        let photo = try XCTUnwrap(context.data).assumingMemoryBound(to: UInt8.self)
        var pixels = [Float16](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                let alpha: Float = x < 4 && y < 4 ? 0.5 : [Float(0), 0.25, 0.5, 1][x / 8]
                for c in 0..<3 {
                    let linear = x < 4 && y < 4 ? [Float(3), 0.45, 0.1][c] : 0.05 + 2.8 * Float(photo[i + c]) / 255
                    pixels[i + c] = Float16(linear * alpha)
                }
                pixels[i + 3] = Float16(alpha)
            }
        }
        let provider = try XCTUnwrap(CGDataProvider(data: pixels.withUnsafeBytes { Data($0) } as CFData))
        let space = try XCTUnwrap(CGColorSpace(name: CGColorSpace.extendedLinearDisplayP3))
        let bitmap = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder16Little.rawValue | CGBitmapInfo.floatComponents.rawValue)
        let hdr = try XCTUnwrap(CGImage(width: width, height: height, bitsPerComponent: 16, bitsPerPixel: 64,
            bytesPerRow: width * 8, space: space, bitmapInfo: bitmap, provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let url = directory.appendingPathComponent("hdr-alpha.tiff")
        try write(hdr, type: .tiff, to: url)
        return url
    }

    private func write(_ image: CGImage, type: UTType, to url: URL) throws {
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, [kCGImagePropertyDepth: image.bitsPerComponent] as CFDictionary)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
    }

    @discardableResult
    private func assertFiniteNativeResult(_ artifacts: CompletedPipelineArtifacts,
        file: StaticString = #filePath, line: UInt = #line) -> Double {
        XCTAssertEqual(artifacts.sourceBuffer.width, artifacts.outputBuffer.width, file: file, line: line)
        XCTAssertEqual(artifacts.sourceBuffer.height, artifacts.outputBuffer.height, file: file, line: line)
        var change = 0.0
        for y in 0..<artifacts.outputBuffer.height {
            for x in 0..<artifacts.outputBuffer.width {
                let original = artifacts.sourceBuffer.unsafePixelPointer(x: x, y: y)
                let result = artifacts.outputBuffer.unsafePixelPointer(x: x, y: y)
                for c in 0..<4 { XCTAssertTrue(result[c].isFinite, file: file, line: line) }
                XCTAssertEqual(result[3], original[3], file: file, line: line)
                for c in 0..<3 { change += Double(abs(Float(result[c]) - Float(original[c]))) }
            }
        }
        return change / Double(artifacts.outputBuffer.width * artifacts.outputBuffer.height * 3)
    }

    private func assertHistory(_ metadata: ImageMetadataPacket, artifacts: CompletedPipelineArtifacts,
        prior: [ImageProcessingHistory.Step], file: StaticString = #filePath, line: UInt = #line) throws {
        let steps = try XCTUnwrap(metadata.processingHistory?.steps, file: file, line: line)
        XCTAssertEqual(Array(steps.dropLast()), prior, file: file, line: line)
        let step = try XCTUnwrap(steps.last, file: file, line: line)
        let config = artifacts.checkpoint.config
        XCTAssertEqual(step.id, config.id, file: file, line: line)
        XCTAssertEqual(step.app, "com.iantruong.truongnoiser", file: file, line: line)
        XCTAssertEqual(step.operation, "denoise", file: file, line: line)
        XCTAssertEqual(step.model, "\(config.backbone.rawValue) (Y \(Int(config.lumaStrength * 100))%, C \(Int(config.chromaStrength * 100))%)", file: file, line: line)
        XCTAssertNil(step.scale, file: file, line: line)
        XCTAssertEqual(step.inputWidth, artifacts.sourceBuffer.width, file: file, line: line)
        XCTAssertEqual(step.inputHeight, artifacts.sourceBuffer.height, file: file, line: line)
        XCTAssertEqual(step.outputWidth, step.inputWidth, file: file, line: line)
        XCTAssertEqual(step.outputHeight, step.inputHeight, file: file, line: line)
    }

    private func assertPixels(_ actual: MMapBuffer, equalTo expected: MMapBuffer, accuracy: Float,
        file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.width, expected.width, file: file, line: line)
        XCTAssertEqual(actual.height, expected.height, file: file, line: line)
        guard actual.width == expected.width, actual.height == expected.height else { return }
        var maximum: Float = 0
        for y in 0..<actual.height {
            for x in 0..<actual.width {
                let a = actual.unsafePixelPointer(x: x, y: y), b = expected.unsafePixelPointer(x: x, y: y)
                for c in 0..<4 {
                    XCTAssertTrue(a[c].isFinite, file: file, line: line)
                    maximum = max(maximum, abs(Float(a[c]) - Float(b[c])))
                }
            }
        }
        XCTAssertLessThanOrEqual(maximum, accuracy, "Maximum export readback error", file: file, line: line)
    }

    private func differenceMetrics(_ lhs: MMapBuffer, _ rhs: MMapBuffer, xRange: Range<Int>? = nil)
        -> (mae: Double, rmse: Double, maximum: Double) {
        var sum = 0.0, squares = 0.0, maximum = 0.0, count = 0
        for y in 0..<lhs.height {
            for x in (xRange ?? (0..<lhs.width)) {
                let a = lhs.unsafePixelPointer(x: x, y: y), b = rhs.unsafePixelPointer(x: x, y: y)
                for c in 0..<3 {
                    let difference = abs(Double(a[c]) - Double(b[c]))
                    sum += difference; squares += difference * difference; maximum = max(maximum, difference); count += 1
                }
            }
        }
        return (sum / Double(count), sqrt(squares / Double(count)), maximum)
    }

    private func residualGradient(_ lhs: MMapBuffer, _ rhs: MMapBuffer, columns: [Int]) -> Double {
        var sum = 0.0, count = 0
        for y in 0..<lhs.height {
            for x in columns {
                let a0 = lhs.unsafePixelPointer(x: x - 1, y: y), a1 = lhs.unsafePixelPointer(x: x, y: y)
                let b0 = rhs.unsafePixelPointer(x: x - 1, y: y), b1 = rhs.unsafePixelPointer(x: x, y: y)
                for c in 0..<3 {
                    sum += abs((Double(a1[c]) - Double(b1[c])) - (Double(a0[c]) - Double(b0[c])))
                    count += 1
                }
            }
        }
        return sum / Double(count)
    }

    private func componentPeak(_ buffer: MMapBuffer) -> Float {
        var peak: Float = 0
        for y in 0..<buffer.height {
            for x in 0..<buffer.width {
                let pixel = buffer.unsafePixelPointer(x: x, y: y)
                for c in 0..<3 { peak = max(peak, Float(pixel[c])) }
            }
        }
        return peak
    }

    private func snapshot(_ buffer: MMapBuffer) -> [UInt16] {
        (0..<buffer.height).flatMap { y in
            (0..<buffer.width).flatMap { x in (0..<4).map { buffer.unsafePixelPointer(x: x, y: y)[$0].bitPattern } }
        }
    }

    private func journalNames(_ directory: URL) throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: directory.path).filter { $0.hasPrefix("journal_") }
    }

    private func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    private var reportDirectory: URL {
        if let root = ProcessInfo.processInfo.environment["TRUONGNOISER_QUALITY_EVIDENCE_DIR"] {
            return URL(fileURLWithPath: root, isDirectory: true).appendingPathComponent("cgnet-pipeline-tests")
        }
        return repositoryRoot.appendingPathComponent("build/model-validation/cgnet-pipeline-tests")
    }
    private var retainImages: Bool { ProcessInfo.processInfo.environment["TRUONGNOISER_RETAIN_TEST_IMAGES"] == "1" }

    private func retainImageIfRequested(_ source: URL, name: String) throws {
        guard retainImages else { return }
        try FileManager.default.createDirectory(at: reportDirectory, withIntermediateDirectories: true)
        let destination = reportDirectory.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
        try FileManager.default.copyItem(at: source, to: destination)
    }

    private func writeReport(_ name: String, values: [String: Any]) throws {
        var report = values
        #if targetEnvironment(simulator)
        report["platform"] = "iOS Simulator"
        #else
        report["platform"] = "macOS"
        #endif
        report["scope"] = "Real required-model end-to-end Mac/simulator pipeline regressions; fixed CPU-only compute request"
        report["physicalPhoneSpeed"] = "NOT_MEASURED"
        report["physicalImageQuality"] = "NOT_MEASURED"
        report["actualANEPlacement"] = "NOT_MEASURED"
        report["commercialDenoiserParity"] = "NOT_MEASURED"
        try FileManager.default.createDirectory(at: reportDirectory, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: reportDirectory.appendingPathComponent(name + ".json"), options: .atomic)
    }
}
