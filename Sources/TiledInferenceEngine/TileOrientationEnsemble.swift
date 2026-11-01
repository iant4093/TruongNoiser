/// The eight symmetries of a square, in a stable order used by the Best denoise mode.
/// A mirrored orientation first reflects the source left-to-right, then rotates clockwise.
/// Each prediction sees a differently oriented copy of the same noisy input, never another
/// prediction. Applying strengths belongs after deorientation and averaging.
enum TileOrientation: Int, CaseIterable, Sendable {
    case identity
    case rotate90
    case rotate180
    case rotate270
    case mirrorHorizontal
    case mirrorHorizontalRotate90
    case mirrorHorizontalRotate180
    case mirrorHorizontalRotate270

    /// Returns the destination offset for an original `(x,y)` coordinate.
    fileprivate func destinationIndex(x: Int, y: Int, size: Int) -> Int {
        let reflectedX = rawValue >= 4 ? size - 1 - x : x
        switch rawValue % 4 {
        case 1: return reflectedX * size + size - 1 - y
        case 2: return (size - 1 - y) * size + size - 1 - reflectedX
        case 3: return (size - 1 - reflectedX) * size + y
        default: return y * size + reflectedX
        }
    }
}

/// Bounded planar `[1,3,H,W]` orientation transforms and a streaming Float32 accumulator.
/// Only one oriented input/prediction is needed at a time; predictions are never retained as a batch.
enum TileOrientationEnsemble {
    enum Error: Swift.Error, Sendable, CustomStringConvertible {
        case unsupportedTileSize(Int)
        case invalidTensorShape(expected: Int, actual: Int)
        case invalidAllowedRange
        case invalidPrediction(orientation: TileOrientation, index: Int)
        case duplicateOrientation(TileOrientation)
        case incompletePasses(completed: Int)

        var description: String {
            switch self {
            case .unsupportedTileSize(let size):
                return "Orientation ensemble requires a square tile of 1...512 pixels, got \(size)."
            case .invalidTensorShape(let expected, let actual):
                return "Orientation ensemble expected \(expected) planar samples, got \(actual)."
            case .invalidAllowedRange:
                return "Orientation ensemble prediction bounds must be finite."
            case .invalidPrediction(let orientation, let index):
                return "Orientation prediction \(orientation.rawValue + 1)/8 has an invalid sample at \(index)."
            case .duplicateOrientation(let orientation):
                return "Orientation prediction \(orientation.rawValue + 1)/8 was added more than once."
            case .incompletePasses(let completed):
                return "Orientation ensemble requires all eight predictions; \(completed) are complete."
            }
        }
    }

    /// Sizes are bounded by the currently shipped native denoiser contract. Checking before
    /// multiplication/allocation also refuses corrupt or overflowing dimensions without allocating.
    private static func elementCount(size: Int) throws -> Int {
        guard (1...512).contains(size) else { throw Error.unsupportedTileSize(size) }
        return 3 * size * size
    }

    private static func validateShape(_ values: [Float16], size: Int) throws {
        let expected = try elementCount(size: size)
        guard values.count == expected else {
            throw Error.invalidTensorShape(expected: expected, actual: values.count)
        }
    }

    static func transformed(_ values: [Float16], size: Int, orientation: TileOrientation) throws -> [Float16] {
        try validateShape(values, size: size)
        if orientation == .identity { return values }
        let planeSize = size * size
        var output = [Float16](repeating: 0, count: values.count)
        for channel in 0..<3 {
            let base = channel * planeSize
            for y in 0..<size {
                for x in 0..<size {
                    output[base + orientation.destinationIndex(x: x, y: y, size: size)] = values[base + y * size + x]
                }
            }
        }
        return output
    }

    /// Maps an oriented prediction back to the original pixel coordinates without interpolation.
    static func restored(_ values: [Float16], size: Int, orientation: TileOrientation) throws -> [Float16] {
        try validateShape(values, size: size)
        if orientation == .identity { return values }
        let planeSize = size * size
        var output = [Float16](repeating: 0, count: values.count)
        for channel in 0..<3 {
            let base = channel * planeSize
            for y in 0..<size {
                for x in 0..<size {
                    output[base + y * size + x] = values[base + orientation.destinationIndex(x: x, y: y, size: size)]
                }
            }
        }
        return output
    }

    struct Accumulator {
        private let size: Int
        private let allowed: ClosedRange<Float>
        private var sums: [Float]
        private var completedMask: UInt8 = 0

        var completedPassCount: Int { completedMask.nonzeroBitCount }

        init(size: Int, allowed: ClosedRange<Float>) throws {
            guard allowed.lowerBound.isFinite, allowed.upperBound.isFinite else {
                throw Error.invalidAllowedRange
            }
            let count = try TileOrientationEnsemble.elementCount(size: size)
            self.size = size
            self.allowed = allowed
            self.sums = [Float](repeating: 0, count: count)
        }

        /// Validates the entire raw pass before mutating the accumulator. A bad pass cannot be
        /// hidden by averaging with good passes or partially contribute before an error is raised.
        mutating func add(_ prediction: [Float16], orientation: TileOrientation) throws {
            try TileOrientationEnsemble.validateShape(prediction, size: size)
            let bit = UInt8(1) << orientation.rawValue
            guard completedMask & bit == 0 else { throw Error.duplicateOrientation(orientation) }
            for (index, sample) in prediction.enumerated() {
                guard sample.isFinite, allowed.contains(Float(sample)) else {
                    throw Error.invalidPrediction(orientation: orientation, index: index)
                }
            }
            let planeSize = size * size
            for channel in 0..<3 {
                let base = channel * planeSize
                for y in 0..<size {
                    for x in 0..<size {
                        sums[base + y * size + x] += Float(prediction[base + orientation.destinationIndex(x: x, y: y, size: size)])
                    }
                }
            }
            completedMask |= bit
        }

        func average() throws -> [Float16] {
            guard completedMask == .max else { throw Error.incompletePasses(completed: completedPassCount) }
            // Float32 accumulation prevents eight Float16 rounding steps; the mean is quantized
            // only once. Each finite Float16 input has already passed its raw prediction bounds.
            return sums.map { Float16($0 / 8) }
        }
    }
}
