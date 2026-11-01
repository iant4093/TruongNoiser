import XCTest
import TruongNoiserCore
@testable import TiledInferenceEngine

final class CGNetBackendSafetyTests: XCTestCase {
    func testObservedSilentZeroBackendResultIsRefusedForBothProductionShapes() {
        for model: CoreMLModelIdentifier in [.cgnet256, .cgnet512] {
            let zero = [Float16](repeating: 0, count: 3 * model.inputTileSize * model.inputTileSize)
            XCTAssertThrowsError(try CoreMLModelRegistry.validateCGNetBackendOutput(zero,
                modelID: model, policy: .requireProductionModels, tileIndex: 7)) { error in
                guard case ScalerError.modelInferenceFailed(let name, let index, let reason) = error else {
                    return XCTFail("Unexpected backend error: \(error)")
                }
                XCTAssertEqual(name, model.rawValue); XCTAssertEqual(index, 7)
                XCTAssertTrue(reason.contains("all-zero"))
            }
        }
    }

    func testExplicitTestFallbackAndNonzeroCGNetOutputsKeepTheirExistingMeaning() throws {
        try CoreMLModelRegistry.validateCGNetBackendOutput([0, 0, 0], modelID: .cgnet256,
            policy: .allowDeterministicFallback, tileIndex: 0)
        try CoreMLModelRegistry.validateCGNetBackendOutput([0, 0, 0], modelID: .nafnet256,
            policy: .requireProductionModels, tileIndex: 0)
        try CoreMLModelRegistry.validateCGNetBackendOutput([0, 0.001, 0], modelID: .cgnet512,
            policy: .requireProductionModels, tileIndex: 0)
    }
}
