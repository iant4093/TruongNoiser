import XCTest
@testable import TiledInferenceEngine

final class TileOrientationEnsembleTests: XCTestCase {
    func testAllEightOrientationsAreDistinctAndInvertExactlyWithoutMixingChannels() throws {
        let source = (1...27).map { Float16($0) }
        let expectedFirstPlanes: [[Float16]] = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [7, 4, 1, 8, 5, 2, 9, 6, 3],
            [9, 8, 7, 6, 5, 4, 3, 2, 1],
            [3, 6, 9, 2, 5, 8, 1, 4, 7],
            [3, 2, 1, 6, 5, 4, 9, 8, 7],
            [9, 6, 3, 8, 5, 2, 7, 4, 1],
            [7, 8, 9, 4, 5, 6, 1, 2, 3],
            [1, 4, 7, 2, 5, 8, 3, 6, 9],
        ]
        var unique = Set<[UInt16]>()
        for orientation in TileOrientation.allCases {
            let transformed = try TileOrientationEnsemble.transformed(source, size: 3, orientation: orientation)
            unique.insert(transformed.map(\.bitPattern))
            for channel in 0..<3 {
                XCTAssertEqual(Array(transformed[(channel * 9)..<((channel + 1) * 9)]),
                               expectedFirstPlanes[orientation.rawValue].map { $0 + Float16(channel * 9) })
            }
            XCTAssertEqual(try TileOrientationEnsemble.restored(transformed, size: 3, orientation: orientation), source)
        }
        XCTAssertEqual(unique.count, 8)
    }

    func testStreamingAverageDeorientsNonEquivariantPredictions() throws {
        let size = 3
        let source = (0..<27).map { Float16($0) / 64 }
        var accumulator = try TileOrientationEnsemble.Accumulator(size: size, allowed: -0.25...1.25)
        for orientation in TileOrientation.allCases {
            let transformed = try TileOrientationEnsemble.transformed(source, size: size, orientation: orientation)
            // A coordinate-dependent predictor is intentionally not rotation-equivariant.
            // With all eight symmetries, every original pixel visits each x-coordinate evenly.
            let prediction = transformed.enumerated().map { index, sample in
                Float16(Float(sample) / 2 + Float(index % size) / 16 + Float(orientation.rawValue) / 128)
            }
            try accumulator.add(prediction, orientation: orientation)
        }
        XCTAssertEqual(accumulator.completedPassCount, 8)
        // mean(x)=1 and mean(orientationID)=3.5; all operands are exactly binary-representable.
        XCTAssertEqual(try accumulator.average(), source.map { $0 / 2 + 23 / 256 })
    }

    func testInvalidRawPassCannotBeHiddenByAnOtherwiseValidAverageOrPartiallyAdded() throws {
        let good = [Float16](repeating: 0.25, count: 27)
        for invalid in [Float16.nan, .infinity, -.infinity, -0.5, 1.5] {
            var accumulator = try TileOrientationEnsemble.Accumulator(size: 3, allowed: -0.25...1.25)
            try accumulator.add(good, orientation: .identity)
            var bad = [Float16](repeating: 0.75, count: 27)
            bad[26] = invalid // Reject before the preceding 26 samples can affect the sums.
            XCTAssertThrowsError(try accumulator.add(bad, orientation: .rotate90)) { error in
                guard case TileOrientationEnsemble.Error.invalidPrediction(.rotate90, 26) = error else {
                    return XCTFail("Unexpected error: \(error)")
                }
            }
            XCTAssertEqual(accumulator.completedPassCount, 1)
            // Replacing the failed pass must succeed and cannot retain any of its earlier values.
            for orientation in TileOrientation.allCases where orientation != .identity {
                try accumulator.add(good, orientation: orientation)
            }
            XCTAssertEqual(try accumulator.average(), good)
        }
    }

    func testIncompleteDuplicateAndMalformedPassesAreRefused() throws {
        var accumulator = try TileOrientationEnsemble.Accumulator(size: 3, allowed: -0.25...1.25)
        let good = [Float16](repeating: 0.25, count: 27)
        XCTAssertThrowsError(try accumulator.average())
        XCTAssertThrowsError(try accumulator.add([0.25], orientation: .identity))
        XCTAssertEqual(accumulator.completedPassCount, 0)
        try accumulator.add(good, orientation: .identity)
        XCTAssertThrowsError(try accumulator.add(good, orientation: .identity))
        XCTAssertEqual(accumulator.completedPassCount, 1)
        XCTAssertThrowsError(try accumulator.average())
        for size in [0, -1, 513, Int.max] {
            XCTAssertThrowsError(try TileOrientationEnsemble.Accumulator(size: size, allowed: -0.25...1.25))
            XCTAssertThrowsError(try TileOrientationEnsemble.transformed(good, size: size, orientation: .rotate90))
        }
        XCTAssertThrowsError(try TileOrientationEnsemble.restored([0.25], size: 3, orientation: .rotate90))
        XCTAssertThrowsError(try TileOrientationEnsemble.Accumulator(size: 3, allowed: -.infinity...1.25))
    }

    func testFiniteHalfExtremesAccumulateWithoutHalfPrecisionOverflow() throws {
        let max = Float16.greatestFiniteMagnitude
        let source: [Float16] = [max, -max, max]
        var accumulator = try TileOrientationEnsemble.Accumulator(size: 1, allowed: -Float(max)...Float(max))
        for orientation in TileOrientation.allCases { try accumulator.add(source, orientation: orientation) }
        XCTAssertEqual(try accumulator.average(), source)
    }
}
