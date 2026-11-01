import Foundation
import XCTest
@testable import MetalColorEngine
import MMapCanvas
import TruongNoiserCore

final class DenoiseColorAdapterTests: XCTestCase {
    func testTrainingInputUsesDevelopedSRGBRatherThanLinearOrLogLight() throws {
        let fixture = try makeFixture(pixels: [[0.18, 0.18, 0.18, 1]])
        defer { fixture.remove() }
        let prepared = try DenoiseColorAdapter.prepare(source: fixture.buffer, tile: tile(), componentPeak: 1)
        // IEC sRGB transfer of linear neutral 18% gray is about 0.461, not 0.18 or log-P3.
        for channel in 0..<3 {
            XCTAssertEqual(Float(prepared.input[channel * 16]), 0.461356, accuracy: 0.0005)
        }
        XCTAssertEqual(prepared.exposure, 1)
        XCTAssertEqual(prepared.input.count, 3 * 4 * 4)
    }

    func testIdentityResidualPreservesHDRWideGamutAndPremultipliedAlphaExactly() throws {
        let fixture = try makeFixture(pixels: [
            [8, 2, 0.5, 1], [0, 0.75, 0, 1], [0.4, 0.1, 0.2, 0.5],
            [0.01, 0.02, 0.03, 0.125], [0, 0, 0, 0], [0.18, 0.18, 0.18, 1]
        ])
        defer { fixture.remove() }
        let prepared = try DenoiseColorAdapter.prepare(source: fixture.buffer, tile: tile(), componentPeak: 8)
        XCTAssertTrue(prepared.input.allSatisfy { $0.isFinite && (0...1).contains($0) })
        let strengthsToCheck: [(Float, Float)] = [(0, 0), (1, 1), (0.8, 1), (1, 0), (0, 1)]
        for strengths in strengthsToCheck {
            let restored = try restore(prepared.input, prepared, luma: strengths.0, chroma: strengths.1)
            XCTAssertEqual(restored.map(\.bitPattern), prepared.originalRGBA.map(\.bitPattern),
                           "A zero model residual must preserve the original gamut/HDR/alpha, at every strength")
        }
    }

    func testHDRResidualRestoresExposureAndAlphaInLinearLight() throws {
        let fixture = try makeFixture(pixels: [[1, 1, 1, 0.5]])
        defer { fixture.remove() }
        let prepared = try DenoiseColorAdapter.prepare(source: fixture.buffer, tile: tile(), componentPeak: 4)
        // Straight RGB is 2; model input is sRGB(2/4), independent of premultiplication.
        XCTAssertEqual(Float(prepared.input[0]), 0.735357, accuracy: 0.0006)
        let prediction = [Float16](repeating: 0.5, count: prepared.input.count)
        let output = try restore(prediction, prepared, luma: 1, chroma: 1)
        // sRGB^-1(0.5) = 0.214041; restore exposure4 and alpha0.5 => 0.428082.
        for channel in 0..<3 { XCTAssertEqual(Float(output[channel]), 0.428082, accuracy: 0.002) }
        XCTAssertEqual(output[3], 0.5)
    }

    func testIndependentLuminanceAndChromaStrengthsPreserveTheirComplement() throws {
        let fixture = try makeFixture(pixels: [[0.2, 0.2, 0.2, 1]])
        defer { fixture.remove() }
        let prepared = try DenoiseColorAdapter.prepare(source: fixture.buffer, tile: tile(), componentPeak: 1)
        var prediction = prepared.input
        for index in 0..<16 {
            prediction[index] = 0.65
            prediction[16 + index] = 0.45
            prediction[32 + index] = 0.55
        }
        let full = try restore(prediction, prepared, luma: 1, chroma: 1)
        let neither = try restore(prediction, prepared, luma: 0, chroma: 0)
        XCTAssertEqual(neither.map(\.bitPattern), prepared.originalRGBA.map(\.bitPattern))
        XCTAssertNotEqual(full[0], neither[0], "The fixture must contain a real residual")

        let chromaOnly = try restore(prediction, prepared, luma: 0, chroma: 1)
        let lumaOnly = try restore(prediction, prepared, luma: 1, chroma: 0)
        XCTAssertEqual(luminance(chromaOnly), luminance(prepared.originalRGBA), accuracy: 0.0005)
        let deltaR = Float(lumaOnly[0]) - Float(prepared.originalRGBA[0])
        let deltaG = Float(lumaOnly[1]) - Float(prepared.originalRGBA[1])
        let deltaB = Float(lumaOnly[2]) - Float(prepared.originalRGBA[2])
        XCTAssertEqual(deltaR, deltaG, accuracy: 0.0005)
        XCTAssertEqual(deltaG, deltaB, accuracy: 0.0005)
        XCTAssertEqual(luminance(lumaOnly), luminance(full), accuracy: 0.0005)
        for result in [full, neither, chromaOnly, lumaOnly] { XCTAssertEqual(result[3], 1) }
    }

    func testInvalidPredictionIsRefusedEvenWhenBothStrengthsAreZero() throws {
        let fixture = try makeFixture(pixels: [[0.18, 0.18, 0.18, 1]])
        defer { fixture.remove() }
        let prepared = try DenoiseColorAdapter.prepare(source: fixture.buffer, tile: tile(), componentPeak: 1)
        for invalid in [Float16.nan, .infinity, -.infinity, -0.5, 2] {
            var prediction = prepared.input
            prediction[17] = invalid
            XCTAssertThrowsError(try restore(prediction, prepared, luma: 0, chroma: 0))
        }
        XCTAssertThrowsError(try restore(Array(prepared.input.dropLast()), prepared))
        XCTAssertThrowsError(try restore(prepared.input + [0], prepared))
    }

    func testNonFiniteSourceAndInvalidExposureAreRefused() throws {
        let fixture = try makeFixture(pixels: [[0.18, 0.18, 0.18, 1]])
        defer { fixture.remove() }
        for invalid in [Float.nan, .infinity, 0, -1] {
            XCTAssertThrowsError(try DenoiseColorAdapter.prepare(source: fixture.buffer, tile: tile(), componentPeak: invalid))
        }
        fixture.buffer.unsafePixelPointer(x: 0, y: 0)[0] = .nan
        XCTAssertThrowsError(try DenoiseColorAdapter.prepare(source: fixture.buffer, tile: tile(), componentPeak: 1))
        fixture.buffer.unsafePixelPointer(x: 0, y: 0)[0] = 0.18
        fixture.buffer.unsafePixelPointer(x: 0, y: 0)[3] = .nan
        XCTAssertThrowsError(try DenoiseColorAdapter.prepare(source: fixture.buffer, tile: tile(), componentPeak: 1))
    }

    func testInvalidStrengthsAreRefusedRatherThanSilentlyChangingTheRecipe() throws {
        let fixture = try makeFixture(pixels: [[0.18, 0.18, 0.18, 1]])
        defer { fixture.remove() }
        let prepared = try DenoiseColorAdapter.prepare(source: fixture.buffer, tile: tile(), componentPeak: 1)
        for invalid in [Float.nan, .infinity, -.infinity] {
            XCTAssertThrowsError(try restore(prepared.input, prepared, luma: invalid, chroma: 1))
            XCTAssertThrowsError(try restore(prepared.input, prepared, luma: 1, chroma: invalid))
        }
    }

    private func restore(_ prediction: [Float16], _ prepared: PreparedDenoiseTile,
                         luma: Float = 0.8, chroma: Float = 1) throws -> [Float16] {
        try DenoiseColorAdapter.restore(prediction, prepared: prepared,
            modelName: "NAFNet_w64_256", tileIndex: 0, allowed: -0.25...1.25,
            lumaStrength: luma, chromaStrength: chroma)
    }

    private func luminance(_ rgba: [Float16]) -> Float {
        0.22897456 * Float(rgba[0]) + 0.69173852 * Float(rgba[1]) + 0.07928691 * Float(rgba[2])
    }

    private func tile() -> TileDescriptor {
        TileDescriptor(tileIndex: 0, row: 0, col: 0, inputOriginX: 0, inputOriginY: 0,
                       inputSize: 4, outputOriginX: 0, outputOriginY: 0, outputSize: 4,
                       actualLeftOverlapOut: 0, actualTopOverlapOut: 0)
    }

    private struct Fixture {
        let directory: URL
        let buffer: MMapBuffer
        func remove() { try? FileManager.default.removeItem(at: directory) }
    }

    private func makeFixture(pixels: [[Float]]) throws -> Fixture {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("NoiserColor-\(UUID())")
        let buffer = try MMapBuffer(fileURL: directory.appendingPathComponent("source.bin"), width: 4, height: 4)
        for index in 0..<16 {
            let pixel = pixels[index % pixels.count]
            let destination = buffer.unsafePixelPointer(x: index % 4, y: index / 4)
            for channel in 0..<4 { destination[channel] = Float16(pixel[channel]) }
        }
        return Fixture(directory: directory, buffer: buffer)
    }
}
