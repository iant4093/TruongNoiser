import Foundation
import XCTest
import CoreIngestion
import MMapCanvas
import TiledInferenceEngine
import TruongNoiserCore
@testable import CoreExport

final class NativeDenoiseExportTests: XCTestCase {
    func testDNGReadbackKeepsNativeDimensionsAndRecordsOnlyActualDenoising() throws {
        let fixture = try makeFixture()
        defer { fixture.remove() }
        let destination = fixture.directory.appendingPathComponent("denoised.dng")
        let result = try FormatAwareExporter.exportImageDetailed(artifacts: fixture.artifacts,
            destinationURL: destination, overrideFormat: .linearDNG16Bit)
        XCTAssertEqual(result.actualScaleFactor, 1)
        XCTAssertEqual(result.finalLinearPeak, 0.65, accuracy: 0.001)
        let info = try XCTUnwrap(LinearDNGTIFFReader.probe(imageURL: destination))
        XCTAssertEqual(info.width, 16)
        XCTAssertEqual(info.height, 12)
        XCTAssertEqual(info.bitsPerSample, [16, 16, 16])
        let decoded = try UniversalStripDecoder.decodeToMMap(imageURL: destination,
            destinationMMapURL: fixture.directory.appendingPathComponent("readback-dng.bin"))
        XCTAssertEqual(decoded.decodedWidth, 16)
        XCTAssertEqual(decoded.decodedHeight, 12)
        XCTAssertTrue(decoded.metadata.isAlreadyDenoisedBySuite)
        XCTAssertTrue(decoded.metadata.isAlreadyDenoisedByNoiser)
        XCTAssertFalse(decoded.metadata.isAlreadyUpscaledBySuite)
        XCTAssertTrue(decoded.metadata.hasSuiteLinearDNGMarker)
        try assertDenoiseStep(decoded.metadata, job: fixture.artifacts.checkpoint.jobID)
        assertPixel(decoded.sourceBuffer, expected: [0.65, 0.25, 0.1, 1], accuracy: 0.002)
        XCTAssertEqual(try Data(contentsOf: fixture.artifacts.metadata.sourceURL), fixture.originalFileBytes)
    }

    func testTIFFReadbackDoesNotApplyStrengthsAgainToTheBakedOutput() throws {
        let fixture = try makeFixture()
        defer { fixture.remove() }
        let destination = fixture.directory.appendingPathComponent("denoised.tiff")
        _ = try FormatAwareExporter.exportImage(artifacts: fixture.artifacts, destinationURL: destination,
                                           overrideFormat: .tiff16BitLZW)
        let decoded = try UniversalStripDecoder.decodeToMMap(imageURL: destination,
            destinationMMapURL: fixture.directory.appendingPathComponent("readback-tiff.bin"))
        XCTAssertEqual(decoded.decodedWidth, fixture.artifacts.sourceBuffer.width)
        XCTAssertEqual(decoded.decodedHeight, fixture.artifacts.sourceBuffer.height)
        XCTAssertTrue(decoded.metadata.isAlreadyDenoisedByNoiser)
        XCTAssertFalse(decoded.metadata.isAlreadyUpscaledBySuite)
        // Source is deliberately different. The denoise strengths were already applied before
        // committing the output tile; exporting must not blend toward source a second time.
        assertPixel(decoded.sourceBuffer, expected: [0.65, 0.25, 0.1, 1], accuracy: 0.002)
        try assertDenoiseStep(decoded.metadata, job: fixture.artifacts.checkpoint.jobID)
        let again = fixture.directory.appendingPathComponent("denoised-again.tiff")
        _ = try FormatAwareExporter.exportImage(artifacts: fixture.artifacts, destinationURL: again,
                                           overrideFormat: .tiff16BitLZW)
        let metadata = try MetadataExtractor.extractMetadata(from: again)
        XCTAssertEqual(metadata.processingHistory?.steps.count, 1, "A re-export is not another denoise pass")
    }

    func testBothContainersPreserveEarlierScalerUpscaleAndOrderedHistory() throws {
        let upstream = ImageProcessingHistory.Step(id: UUID(), app: "com.iantruong.truongscaler",
            operation: "upscale", model: "EDSR_x2", scale: 2,
            inputWidth: 8, inputHeight: 6, outputWidth: 16, outputHeight: 12)
        let fixture = try makeFixture(upstream: upstream)
        defer { fixture.remove() }
        for (format, name) in [(ExportFormat.linearDNG16Bit, "chain.dng"), (.tiff16BitLZW, "chain.tiff")] {
            let destination = fixture.directory.appendingPathComponent(name)
            let result = try FormatAwareExporter.exportImageDetailed(artifacts: fixture.artifacts,
                destinationURL: destination, overrideFormat: format)
            XCTAssertEqual(result.actualScaleFactor, 1, "Noiser did not upscale this already enlarged image")
            let decoded = try UniversalStripDecoder.decodeToMMap(imageURL: destination,
                destinationMMapURL: fixture.directory.appendingPathComponent(name + ".bin"))
            XCTAssertTrue(decoded.metadata.isAlreadyUpscaledBySuite)
            XCTAssertEqual(decoded.metadata.upscaledByAppFactor, 2, "Keep Scaler's historical factor, not this job's1×")
            XCTAssertTrue(decoded.metadata.isAlreadyDenoisedByNoiser)
            let history = try XCTUnwrap(decoded.metadata.processingHistory)
            XCTAssertEqual(history.steps.count, 2)
            XCTAssertEqual(history.steps.first, upstream)
            try assertDenoiseStep(decoded.metadata, job: fixture.artifacts.checkpoint.jobID)
        }
    }

    func testHDRAlphaTIFFRetainsActualLinearPixelsAndOriginalCanvas() throws {
        let fixture = try makeFixture(outputPixel: [2, 0.5, 0.125, 0.5], hasAlpha: true)
        defer { fixture.remove() }
        let before = snapshot(fixture.artifacts.sourceBuffer)
        let destination = fixture.directory.appendingPathComponent("hdr-alpha.tiff")
        let result = try FormatAwareExporter.exportImageDetailed(artifacts: fixture.artifacts,
            destinationURL: destination, overrideFormat: .tiff16BitLZW)
        XCTAssertTrue(result.isHDR)
        XCTAssertEqual(result.finalLinearPeak, 2, accuracy: 0.001)
        XCTAssertEqual(result.actualScaleFactor, 1)
        let decoded = try UniversalStripDecoder.decodeToMMap(imageURL: destination,
            destinationMMapURL: fixture.directory.appendingPathComponent("readback-hdr-alpha.bin"))
        XCTAssertEqual(decoded.decodedWidth, 16)
        XCTAssertEqual(decoded.decodedHeight, 12)
        XCTAssertTrue(decoded.metadata.hasAlpha)
        assertPixel(decoded.sourceBuffer, expected: [2, 0.5, 0.125, 0.5], accuracy: 0.005)
        XCTAssertEqual(snapshot(fixture.artifacts.sourceBuffer), before, "Export must not mutate the retained comparison source")
    }

    func testHDRDNGKeepsAbsoluteHighlightsAtNativeResolution() throws {
        let fixture = try makeFixture(outputPixel: [4, 2, 0.5, 1])
        defer { fixture.remove() }
        let destination = fixture.directory.appendingPathComponent("hdr.dng")
        let result = try FormatAwareExporter.exportImageDetailed(artifacts: fixture.artifacts,
            destinationURL: destination, overrideFormat: .linearDNG16Bit)
        XCTAssertTrue(result.isHDR)
        XCTAssertEqual(result.dngHeadroomScale, 4, accuracy: 0.001)
        let decoded = try UniversalStripDecoder.decodeToMMap(imageURL: destination,
            destinationMMapURL: fixture.directory.appendingPathComponent("readback-hdr.bin"))
        XCTAssertEqual(decoded.decodedWidth, 16)
        XCTAssertEqual(decoded.decodedHeight, 12)
        assertPixel(decoded.sourceBuffer, expected: [4, 2, 0.5, 1], accuracy: 0.005)
        XCTAssertTrue(decoded.metadata.isAlreadyDenoisedByNoiser)
        XCTAssertFalse(decoded.metadata.isAlreadyUpscaledBySuite)
    }

    func testUnprocessedInspectionExportDoesNotInventDenoiseHistory() throws {
        let fixture = try makeFixture(didRunDenoise: false)
        defer { fixture.remove() }
        let destination = fixture.directory.appendingPathComponent("inspection.tiff")
        _ = try FormatAwareExporter.exportImage(artifacts: fixture.artifacts, destinationURL: destination,
                                           overrideFormat: .tiff16BitLZW)
        let metadata = try MetadataExtractor.extractMetadata(from: destination)
        XCTAssertFalse(metadata.isAlreadyDenoisedByNoiser)
        XCTAssertFalse(metadata.isAlreadyDenoisedBySuite)
        XCTAssertFalse(metadata.isAlreadyUpscaledBySuite)
        XCTAssertTrue(metadata.processingHistory?.steps.isEmpty ?? true)
    }

    func testExportCannotReplaceTheOriginalSourceWithItsDenoisedResult() throws {
        let fixture = try makeFixture()
        defer { fixture.remove() }
        XCTAssertThrowsError(try FormatAwareExporter.exportImage(artifacts: fixture.artifacts,
            destinationURL: fixture.artifacts.metadata.sourceURL, overrideFormat: .linearDNG16Bit))
        XCTAssertEqual(try Data(contentsOf: fixture.artifacts.metadata.sourceURL), fixture.originalFileBytes)
    }

    func testInvalidSavedStrengthsRemainInspectableAndCannotReplaceAnExport() throws {
        for strength in [Float.nan, .infinity, -.infinity, -0.1, 1.1] {
            let fixture = try makeFixture(lumaStrength: strength)
            defer { fixture.remove() }
            XCTAssertNil(FormatAwareExporter.processedMetadata(artifacts: fixture.artifacts).processingHistory)
            let destination = fixture.directory.appendingPathComponent("existing.tiff")
            let retained = Data("retained output".utf8)
            try retained.write(to: destination)
            XCTAssertThrowsError(try FormatAwareExporter.exportImage(artifacts: fixture.artifacts,
                destinationURL: destination, overrideFormat: .tiff16BitLZW))
            XCTAssertEqual(try Data(contentsOf: destination), retained)
            XCTAssertEqual(try Data(contentsOf: fixture.artifacts.metadata.sourceURL), fixture.originalFileBytes)
        }
    }

    private func assertDenoiseStep(_ metadata: ImageMetadataPacket, job: UUID,
                                   file: StaticString = #filePath, line: UInt = #line) throws {
        let steps = try XCTUnwrap(metadata.processingHistory?.steps, file: file, line: line)
        let step = try XCTUnwrap(steps.last, file: file, line: line)
        XCTAssertEqual(step.id, job, file: file, line: line)
        XCTAssertEqual(step.app, "com.iantruong.truongnoiser", file: file, line: line)
        XCTAssertEqual(step.operation, "denoise", file: file, line: line)
        XCTAssertEqual(step.model, "NAFNet_w64 (Y 35%, C 60%)", file: file, line: line)
        XCTAssertNil(step.scale, file: file, line: line)
        XCTAssertEqual(step.inputWidth, 16, file: file, line: line)
        XCTAssertEqual(step.inputHeight, 12, file: file, line: line)
        XCTAssertEqual(step.outputWidth, 16, file: file, line: line)
        XCTAssertEqual(step.outputHeight, 12, file: file, line: line)
    }

    private func assertPixel(_ buffer: MMapBuffer, expected: [Float], accuracy: Float,
                             file: StaticString = #filePath, line: UInt = #line) {
        for (x, y) in [(0, 0), (7, 5), (15, 11)] {
            let pixel = buffer.unsafePixelPointer(x: x, y: y)
            for channel in 0..<4 {
                XCTAssertEqual(Float(pixel[channel]), expected[channel], accuracy: accuracy,
                               "pixel(\(x),\(y)) channel\(channel)", file: file, line: line)
            }
        }
    }

    private func snapshot(_ buffer: MMapBuffer) -> [UInt16] {
        (0..<buffer.height).flatMap { y in
            (0..<buffer.width).flatMap { x in
                (0..<4).map { buffer.unsafePixelPointer(x: x, y: y)[$0].bitPattern }
            }
        }
    }

    private struct Fixture {
        let directory: URL
        let artifacts: CompletedPipelineArtifacts
        let originalFileBytes: Data
        func remove() { try? FileManager.default.removeItem(at: directory) }
    }

    private func makeFixture(outputPixel: [Float] = [0.65, 0.25, 0.1, 1], hasAlpha: Bool = false,
                             upstream: ImageProcessingHistory.Step? = nil,
                             didRunDenoise: Bool = true, lumaStrength: Float = 0.35) throws -> Fixture {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("NoiserExport-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let original = directory.appendingPathComponent("original.dng")
        let originalBytes = Data("retained original fixture bytes".utf8)
        try originalBytes.write(to: original)
        let metadata = ImageMetadataPacket(sourceURL: original, utTypeIdentifier: "com.adobe.raw-image",
            category: .cameraRAW, originalWidth: 16, originalHeight: 12, hasAlpha: hasAlpha,
            isAlreadyUpscaledBySuite: upstream != nil, upscaledByAppFactor: upstream == nil ? nil : 2,
            processingHistory: upstream.map { ImageProcessingHistory(steps: [$0]) })
        let source = try MMapBuffer(fileURL: directory.appendingPathComponent("source.bin"), width: 16, height: 12)
        let output = try MMapBuffer(fileURL: directory.appendingPathComponent("output.bin"), width: 16, height: 12)
        for y in 0..<12 {
            for x in 0..<16 {
                let input = source.unsafePixelPointer(x: x, y: y)
                input[0] = 0.95; input[1] = 0.8; input[2] = 0.7; input[3] = 1
                let result = output.unsafePixelPointer(x: x, y: y)
                for channel in 0..<4 { result[channel] = Float16(outputPixel[channel]) }
            }
        }
        let id = UUID()
        var config = DenoiseJobConfig(id: id, sourceImageURL: original, lumaStrength: 0.35, chromaStrength: 0.6)
        config.lumaStrength = lumaStrength // simulate an untrusted saved artifact, bypassing new-job clamping
        let checkpoint = JobCheckpointState(jobID: id, config: config,
            tileConfig: TileConfiguration(tier: .tierB_ConstrainedMemory), sourceWidth: 16, sourceHeight: 12,
            paddedSourceWidth: 256, paddedSourceHeight: 256, outputWidth: 16, outputHeight: 12,
            globalMaxLuminance: 1, globalMaxLinearComponent: 1, didRunDenoise: didRunDenoise,
            sourceMMapFileName: "source.bin", outputMMapFileName: "output.bin", metadataPacketFileName: "metadata.json",
            lastCompletedTileIndex: 0, totalTiles: 1, updatedAt: Date(timeIntervalSince1970: 1_700_000_000),
            normalizationPeakVersion: JobCheckpointState.currentNormalizationPeakVersion)
        return Fixture(directory: directory, artifacts: CompletedPipelineArtifacts(checkpoint: checkpoint,
            metadata: metadata, sourceBuffer: source, outputBuffer: output), originalFileBytes: originalBytes)
    }
}
