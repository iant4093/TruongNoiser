import XCTest
import Foundation
import TruongNoiserCore
@testable import DeepZoomGalleryUI

/// Portable before/after records own the applied recipe, including zero-strength channel choices.
/// Byte fixtures exercise storage/identity only and are never presented as trained model output.
final class ProcessedLibraryStrengthTests: XCTestCase {
    private var directory: URL!
    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("NoiserRecipeRecord-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws {
        if let directory { try? FileManager.default.removeItem(at: directory) }
        directory = nil
    }
    private func folder(_ name: String) throws -> URL {
        let url = directory.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
    private func file(_ name: String, bytes: [UInt8]) throws -> URL {
        let url = directory.appendingPathComponent(name); try Data(bytes).write(to: url); return url
    }
    private func draft(id: UUID = UUID(), luma: Float? = 0.80, chroma: Float? = 1,
        model: String = BackboneModelKind.nafnetNormal.rawValue) -> ProcessedPhotoDraft {
        ProcessedPhotoDraft(id: id, modelRawValue: model, sourceWidth: 16, sourceHeight: 12,
            outputWidth: 16, outputHeight: 12, scaleFactor: 1, grainAmount: 0,
            lumaStrength: luma, chromaStrength: chroma,
            processedAt: Date(timeIntervalSince1970: 1_792_962_000), formatRawValue: ExportFormat.png16Bit.rawValue)
    }
    func testAppliedStrengthsAndNativeDimensionsSurviveFolderMoveRestartAndTrashRestore() async throws {
        let store = ProcessedLibraryStore(), root = try folder("before-move")
        let input = try file("source.png", bytes: [1, 2, 3]), output = try file("result.png", bytes: [4, 5, 6])
        let original = try await store.archiveOriginal(at: input, named: input.lastPathComponent, under: root)
        let record = try await store.publish(exportedURL: output, original: original,
            draft: draft(luma: 0, chroma: 0.65), under: root)
        XCTAssertEqual(record.authoredBy, "com.iantruong.truongnoiser")
        XCTAssertEqual(record.scaleFactor, 1)
        XCTAssertEqual(record.sourceWidth, record.outputWidth)
        XCTAssertEqual(record.sourceHeight, record.outputHeight)
        XCTAssertEqual(record.lumaStrength, 0)
        XCTAssertEqual(record.chromaStrength, 0.65)
        XCTAssertTrue(record.outputRelativePath.contains("TN_"))
        let moved = directory.appendingPathComponent("after-move")
        try FileManager.default.moveItem(at: root, to: moved)
        let restarted = ProcessedLibraryStore()
        let listed = try await restarted.records(under: moved)
        XCTAssertEqual(listed, [record])
        let opened = try await restarted.open(record, under: moved)
        XCTAssertNotNil(opened.originalURL)
        XCTAssertEqual(try Data(contentsOf: opened.outputURL), Data([4, 5, 6]))
        try await restarted.trash(record: record, under: moved)
        let entries = try await restarted.trashed(under: moved)
        let entry = try XCTUnwrap(entries.first)
        XCTAssertEqual(entry.record?.lumaStrength, 0)
        XCTAssertEqual(entry.record?.chromaStrength, 0.65)
        try await ProcessedLibraryStore().restore(entry, under: moved)
        let restored = try await ProcessedLibraryStore().records(under: moved)
        XCTAssertEqual(restored, [record])
        XCTAssertEqual(try Data(contentsOf: input), Data([1, 2, 3]))
    }
    func testOlderRecordsWithoutStrengthsDecodeUnknownRatherThanInventingDefaults() async throws {
        let store = ProcessedLibraryStore(), root = try folder("old-record")
        let original = try await store.archiveOriginal(at: file("input.png", bytes: [1]), named: "input.png", under: root)
        let record = try await store.publish(exportedURL: file("output.png", bytes: [2]), original: original,
            draft: draft(luma: nil, chroma: nil, model: "opaque-previous-denoiser"), under: root)
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(record)) as? [String: Any])
        json.removeValue(forKey: "lumaStrength"); json.removeValue(forKey: "chromaStrength")
        let oldBytes = try JSONSerialization.data(withJSONObject: json, options: .sortedKeys)
        let decoded = try JSONDecoder().decode(ProcessedPhotoRecord.self, from: oldBytes)
        XCTAssertNil(decoded.lumaStrength)
        XCTAssertNil(decoded.chromaStrength)
        XCTAssertEqual(decoded.modelRawValue, "opaque-previous-denoiser")
        XCTAssertEqual(decoded.outputWidth, 16)
        XCTAssertEqual(decoded.outputHeight, 12)
        try ProcessedLibraryStore.validate(decoded)
        let metadata = root.appendingPathComponent(decoded.metadataRelativePath)
        try oldBytes.write(to: metadata)
        let listed = try await store.records(under: root)
        XCTAssertEqual(listed, [decoded])
        XCTAssertEqual(try Data(contentsOf: metadata), oldBytes, "Reopen may not relabel or migrate a historical recipe")
    }
    func testSavedJobCannotBeRelabelledWithDifferentChannelStrengths() async throws {
        let store = ProcessedLibraryStore(), root = try folder("retry")
        let original = try await store.archiveOriginal(at: file("input.png", bytes: [1]), named: "input.png", under: root)
        let output = try file("output.png", bytes: [2, 3]), recipe = draft(luma: 0.35, chroma: 0.75)
        let record = try await store.publish(exportedURL: output, original: original, draft: recipe, under: root)
        let metadataURL = root.appendingPathComponent(record.metadataRelativePath)
        let before = try Data(contentsOf: metadataURL)
        for changed in [draft(id: recipe.id, luma: 0.36, chroma: 0.75), draft(id: recipe.id, luma: 0.35, chroma: 0.76)] {
            do {
                _ = try await store.publish(exportedURL: output, original: original, draft: changed, under: root)
                XCTFail("A saved canvas may not adopt the next job's channel recipe")
            } catch {}
            XCTAssertEqual(try Data(contentsOf: metadataURL), before)
            XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent(record.outputRelativePath)), Data([2, 3]))
        }
    }

    func testSinglePassAndEnsembleRecordsKeepDistinctModeLabelsAfterReopenAndRestore() async throws {
        let store = ProcessedLibraryStore(), root = try folder("model-modes")
        let input = try file("original.png", bytes: [1, 2, 3])
        let original = try await store.archiveOriginal(at: input, named: "original.png", under: root)
        let single = try await store.publish(exportedURL: file("single.png", bytes: [4, 5]),
            original: original, draft: draft(model: BackboneModelKind.cgnet.rawValue), under: root)
        let ensemble = try await store.publish(exportedURL: file("ensemble.png", bytes: [6, 7]),
            original: original, draft: draft(model: BackboneModelKind.cgnetEnsemble.rawValue), under: root)
        XCTAssertEqual(single.modelLabel, "CGNet (Better)")
        XCTAssertEqual(ensemble.modelLabel, "CGNet (Best, 8-pass)")
        XCTAssertNotEqual(single.modelRawValue, ensemble.modelRawValue)
        XCTAssertEqual(single.original, ensemble.original, "Two model modes may share the same archived input")
        let reopened = try await ProcessedLibraryStore().records(under: root)
        XCTAssertEqual(Set(reopened.map(\.modelRawValue)), ["CGNet_SIDD", "CGNet_SIDD_x8"])
        try await store.trash(record: ensemble, under: root)
        let entries = try await store.trashed(under: root)
        let entry = try XCTUnwrap(entries.first)
        XCTAssertEqual(entry.record?.modelRawValue, "CGNet_SIDD_x8")
        try await ProcessedLibraryStore().restore(entry, under: root)
        let restored = try await ProcessedLibraryStore().records(under: root)
        XCTAssertEqual(restored.first(where: { $0.id == ensemble.id }), ensemble)
        let opened = try await ProcessedLibraryStore().open(ensemble, under: root)
        XCTAssertEqual(try Data(contentsOf: opened.outputURL), Data([6, 7]))
        XCTAssertEqual(try Data(contentsOf: XCTUnwrap(opened.originalURL)), Data([1, 2, 3]))
    }

    func testSavedSinglePassCanvasCannotBeRepublishedAsEnsembleOrAnotherModel() async throws {
        let store = ProcessedLibraryStore(), root = try folder("mode-conflict")
        let original = try await store.archiveOriginal(at: file("original.png", bytes: [1, 2]),
            named: "original.png", under: root)
        let output = try file("single.png", bytes: [3, 4]), recipe = draft(model: BackboneModelKind.cgnet.rawValue)
        let record = try await store.publish(exportedURL: output, original: original, draft: recipe, under: root)
        let metadataURL = root.appendingPathComponent(record.metadataRelativePath)
        let metadataBefore = try Data(contentsOf: metadataURL)
        for model: BackboneModelKind in [.cgnetEnsemble, .nafnetNormal] {
            do {
                _ = try await store.publish(exportedURL: output, original: original,
                    draft: draft(id: recipe.id, model: model.rawValue), under: root)
                XCTFail("A saved result must keep its actual inference mode")
            } catch {}
            XCTAssertEqual(try Data(contentsOf: metadataURL), metadataBefore)
            XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent(record.outputRelativePath)), Data([3, 4]))
            XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent(original.relativePath)), Data([1, 2]))
        }
    }
}
