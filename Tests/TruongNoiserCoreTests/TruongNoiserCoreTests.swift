import Foundation
import XCTest
@testable import TruongNoiserCore

final class TruongNoiserCoreTests: XCTestCase {
    func testNewJobKeepsNativeResolutionAndFreezesItsDenoiseRecipe() throws {
        let defaults = DenoiseJobConfig(sourceImageURL: URL(fileURLWithPath: "/tmp/input.dng"))
        XCTAssertEqual(ScaleFactor.allCases, [.x1])
        XCTAssertEqual(defaults.scaleFactor, .x1)
        XCTAssertEqual(defaults.backbone, .nafnetNormal)
        XCTAssertEqual(defaults.denoiseMode, .forceOn)
        XCTAssertEqual(defaults.lumaStrength, 0.8)
        XCTAssertEqual(defaults.chromaStrength, 1)
        XCTAssertEqual(defaults.naturalGrainAmount, 0)
        var recipe = defaults
        recipe.lumaStrength = 0.35
        recipe.chromaStrength = 0.7
        let decoded = try JSONDecoder().decode(DenoiseJobConfig.self, from: JSONEncoder().encode(recipe))
        XCTAssertEqual(decoded, recipe)
        XCTAssertNotEqual(decoded.lumaStrength, defaults.lumaStrength)
        XCTAssertNotEqual(decoded.chromaStrength, defaults.chromaStrength)
    }

    func testOpaqueUnknownModelIdentitiesSurviveWithoutBecomingSelectable() throws {
        XCTAssertEqual(BackboneModelKind.allCases, [.nafnetNormal, .cgnet, .cgnetEnsemble])
        XCTAssertFalse(BackboneModelKind.restormerAstro.isSupported)
        for name in ["future-denoiser-v9", "retired-NAFNet-custom-512", "EDSR_x2"] {
            let data = try JSONEncoder().encode(name)
            let decoded = try JSONDecoder().decode(BackboneModelKind.self, from: data)
            XCTAssertEqual(decoded.rawValue, name)
            XCTAssertFalse(decoded.isSupported)
            XCTAssertEqual(try JSONEncoder().encode(decoded), data)
        }
    }

    func testSelectableDenoiseModesKeepDistinctRawRecipesAndSharedCGNetFamily() throws {
        let modes = BackboneModelKind.allCases
        XCTAssertEqual(modes.map(\.rawValue), ["NAFNet_w64", "CGNet_SIDD", "CGNet_SIDD_x8"])
        XCTAssertEqual(modes.map(\.ensemblePassCount), [1, 1, 8])
        XCTAssertEqual(modes.map(\.baseModel), [.nafnetNormal, .cgnet, .cgnet])
        XCTAssertEqual(modes.map(\.displayName), ["NAFNet (Fast)", "CGNet (Better)", "CGNet (Best, 8-pass)"])
        XCTAssertTrue(BackboneModelKind.cgnetEnsemble.processingHint.contains("8 predictions per tile"))
        XCTAssertTrue(BackboneModelKind.cgnetEnsemble.processingHint.contains("Stop"))
        for mode in modes {
            let config = DenoiseJobConfig(sourceImageURL: URL(fileURLWithPath: "/tmp/input.dng"),
                backbone: mode, lumaStrength: 0.45, chromaStrength: 0.85)
            let decoded = try JSONDecoder().decode(DenoiseJobConfig.self, from: JSONEncoder().encode(config))
            XCTAssertEqual(decoded, config)
            XCTAssertEqual(decoded.backbone, mode)
            XCTAssertEqual(decoded.backbone.ensemblePassCount, mode.ensemblePassCount)
            XCTAssertEqual(decoded.scaleFactor, .x1)
        }
        XCTAssertNotEqual(BackboneModelKind.cgnet, .cgnetEnsemble,
            "Sharing trained weights does not make single-pass and ensemble canvases interchangeable")
    }

    func testCheckpointRoundTripKeepsNativeCanvasModelAndStrengthsForResume() throws {
        let id = UUID()
        let configuration = DenoiseJobConfig(id: id, sourceImageURL: URL(fileURLWithPath: "/tmp/original.dng"),
                                            lumaStrength: 0.4, chromaStrength: 0.75)
        let checkpoint = JobCheckpointState(jobID: id, config: configuration,
            tileConfig: TileConfiguration(tier: .tierB_ConstrainedMemory),
            sourceWidth: 4032, sourceHeight: 3024, paddedSourceWidth: 4032, paddedSourceHeight: 3024,
            outputWidth: 4032, outputHeight: 3024, globalMaxLuminance: 2, globalMaxLinearComponent: 4,
            didRunDenoise: true, sourceMMapFileName: "source.bin", outputMMapFileName: "result.bin",
            metadataPacketFileName: "metadata.json", lastCompletedTileIndex: 12, totalTiles: 234,
            updatedAt: Date(timeIntervalSince1970: 1_700_000_000),
            normalizationPeakVersion: JobCheckpointState.currentNormalizationPeakVersion)
        let decoded = try JSONDecoder().decode(JobCheckpointState.self, from: JSONEncoder().encode(checkpoint))
        XCTAssertEqual(decoded, checkpoint)
        XCTAssertEqual(decoded.config.backbone.rawValue, "NAFNet_w64")
        XCTAssertEqual(decoded.config.lumaStrength, 0.4)
        XCTAssertEqual(decoded.config.chromaStrength, 0.75)
        XCTAssertEqual(decoded.outputWidth, decoded.sourceWidth)
        XCTAssertEqual(decoded.outputHeight, decoded.sourceHeight)
        XCTAssertEqual(decoded.normalizationPeakVersion, 3)
        XCTAssertEqual(decoded.lastCompletedTileIndex, 12)
    }

    func testMemoryTiersUseMatchingNativeDenoiseShapesAndOverlap() {
        let large = TileConfiguration(tier: HardwareTelemetry.resolveDeviceTier(overrideAvailableBytes: 3_000_000_000))
        XCTAssertEqual(large.inputTileSize, 512)
        XCTAssertEqual(large.inputOverlap, 32)
        XCTAssertEqual(large.effectiveInputStride, 480)
        XCTAssertEqual(large.outputTileSize(for: .x1), 512)
        XCTAssertEqual(large.outputOverlap(for: .x1), 32)
        let small = TileConfiguration(tier: HardwareTelemetry.resolveDeviceTier(overrideAvailableBytes: 2_000_000_000))
        XCTAssertEqual(small.inputTileSize, 256)
        XCTAssertEqual(small.inputOverlap, 24)
        XCTAssertEqual(small.effectiveInputStride, 232)
        XCTAssertEqual(small.outputTileSize(for: .x1), 256)
        XCTAssertEqual(small.outputOverlap(for: .x1), 24)
    }

    func testBothCGNetModesUseRecorded32PixelOverlapAtEitherMemoryTier() {
        for mode: BackboneModelKind in [.cgnet, .cgnetEnsemble] {
            let large = TileConfiguration(tier: .tierA_HighMemory, backbone: mode)
            let small = TileConfiguration(tier: .tierB_ConstrainedMemory, backbone: mode)
            XCTAssertEqual(large.inputTileSize, 512)
            XCTAssertEqual(large.inputOverlap, 32)
            XCTAssertEqual(large.effectiveInputStride, 480)
            XCTAssertEqual(small.inputTileSize, 256)
            XCTAssertEqual(small.inputOverlap, 32)
            XCTAssertEqual(small.effectiveInputStride, 224)
            XCTAssertEqual(small.outputTileSize(for: .x1), 256)
            XCTAssertEqual(small.outputOverlap(for: .x1), 32)
        }
        XCTAssertNotEqual(TileConfiguration(tier: .tierB_ConstrainedMemory, backbone: .cgnet),
            TileConfiguration(tier: .tierB_ConstrainedMemory, backbone: .nafnetNormal))
    }

    func testDenoiseMetadataAddsItsHistoryWithoutRelabelingTheEarlierUpscale() throws {
        let scalerStep = ImageProcessingHistory.Step(id: UUID(), app: "com.iantruong.truongscaler",
            operation: "upscale", model: "EDSR_x2", scale: 2,
            inputWidth: 3024, inputHeight: 4032, outputWidth: 6048, outputHeight: 8064)
        let denoiseStep = ImageProcessingHistory.Step(id: UUID(), app: "com.iantruong.truongnoiser",
            operation: "denoise", model: "NAFNet_w64 (Y 80%, C 100%)",
            inputWidth: 6048, inputHeight: 8064, outputWidth: 6048, outputHeight: 8064)
        let before = ImageMetadataPacket(sourceURL: URL(fileURLWithPath: "/tmp/scaler.dng"),
            utTypeIdentifier: "com.adobe.raw-image", category: .cameraRAW,
            originalWidth: 6048, originalHeight: 8064,
            isAlreadyUpscaledBySuite: true, upscaledByAppFactor: 2,
            processingHistory: ImageProcessingHistory(steps: [scalerStep]))
        let history = try XCTUnwrap(before.processingHistory).appending(denoiseStep)
        let after = before.withPipelineState(isDenoised: true, isAlreadyDenoisedByNoiser: true,
                                             processingHistory: history)
        XCTAssertTrue(after.isAlreadyDenoisedByNoiser)
        XCTAssertTrue(after.isAlreadyDenoisedBySuite)
        XCTAssertTrue(after.isAlreadyUpscaledBySuite)
        XCTAssertEqual(after.upscaledByAppFactor, 2)
        XCTAssertEqual(after.originalWidth, before.originalWidth)
        XCTAssertEqual(after.originalHeight, before.originalHeight)
        let roundTrip = try XCTUnwrap(ImageProcessingHistory.parse(XCTUnwrap(after.processingHistory?.marker)))
        XCTAssertEqual(roundTrip.steps, [scalerStep, denoiseStep])
        XCTAssertEqual(roundTrip.appending(denoiseStep).steps.count, 2, "Re-export must not duplicate this job's evidence")
        XCTAssertFalse(before.isAlreadyDenoisedByNoiser, "Inspecting or forwarding an input must not mark it denoised")
    }

    func testIncomingScalerHandoffTargetsNoiserAndRetainsActualUpscaleFactor() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("NoiserHandoff-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let original = directory.appendingPathComponent("source.dng")
        let bytes = Data([0x49, 0x49, 0x2a, 0x00, 0x01, 0x02, 0x03])
        try bytes.write(to: original)
        let (payload, link) = try AppGroupHandoffManager.createHandoffPackage(
            exportedImageURL: original, originalFileName: "source.dng", sourceApp: .truongScaler,
            targetApp: .truongNoiser, isDenoised: false, isUpscaled: true, globalMaxLuminance: 2,
            upscaleFactor: 2, globalMaxLinearComponent: 4, overrideBaseDirectory: directory)
        XCTAssertEqual(link.scheme, "truongnoiser")
        let consumed = try AppGroupHandoffManager.consumeIncomingHandoff(from: link, overrideBaseDirectory: directory)
        XCTAssertEqual(consumed, payload)
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: consumed.imageFileURLPath)), bytes)
        XCTAssertEqual(consumed.pipelineSessionState.upscaleFactor, 2)
        XCTAssertEqual(consumed.pipelineSessionState.isUpscaled, true)
        XCTAssertEqual(consumed.pipelineSessionState.isDenoised, false)
        XCTAssertNil(consumed.pipelineSessionState.isDenoisedByNoiser)
        XCTAssertEqual(try Data(contentsOf: original), bytes, "A handoff must not move or mutate its source")
    }
}
