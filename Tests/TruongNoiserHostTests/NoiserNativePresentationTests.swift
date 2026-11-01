import XCTest
import UIKit
import SwiftUI
import CoreGraphics
import UniformTypeIdentifiers
import Observation
import TruongNoiserCore
import CoreIngestion
import BackgroundJobManager
@testable import TiledInferenceEngine
@testable import DeepZoomGalleryUI

/// Actual app-scene native presentation regressions. All files/defaults are private test fixtures;
/// The integration case requires actual built-app weights; other cases exercise presentation only.
/// No case manipulates the App Group or selects a physical phone.
@MainActor
final class NoiserNativePresentationTests: XCTestCase {
    func testBuiltAppBestCGNetSavesAndReopensNativeResolutionBeforeAfter() async throws {
        XCTAssertEqual(Bundle.main.bundleIdentifier, "com.iantruong.truongnoiser")
        let bundleModels = try XCTUnwrap(Bundle.main.resourceURL).appendingPathComponent("Models", isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath()
        // Replace the entire discovery search path. A missing app resource must fail this test;
        // the developer's source checkout, other test bundles and caches cannot mask it.
        let registry = CoreMLModelRegistry(initialComputeUnits: .cpuOnly,
            availabilityPolicy: .requireProductionModels, searchDirectoriesOverride: [bundleModels])
        let discovered = await registry.discoveredProductionModelURLs()
        XCTAssertEqual(Set(discovered.keys), Set(CoreMLModelIdentifier.allCases))
        XCTAssertEqual(discovered.count, 4)
        for identifier in CoreMLModelIdentifier.allCases {
            let url = try XCTUnwrap(discovered[identifier]).standardizedFileURL.resolvingSymlinksInPath()
            XCTAssertEqual(url.deletingLastPathComponent(), bundleModels)
            XCTAssertEqual(url.lastPathComponent, identifier.rawValue + ".mlmodelc")
            XCTAssertTrue(FileManager.default.fileExists(atPath: url.appendingPathComponent("weights/weight.bin").path))
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("NoiserBuiltAppIntegration-\(UUID())")
        let input = directory.appendingPathComponent("input"), output = directory.appendingPathComponent("output")
        for folder in [input, output] { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true) }
        let suite = "NoiserBuiltAppIntegration-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer {
            defaults.removePersistentDomain(forName: suite)
            try? FileManager.default.removeItem(at: directory)
        }
        // Official NAFNet repository's noisy256 photograph, shared with the independently validated
        // trained-model suite. This is a real noisy photo, not a Scaler result or an identity fixture.
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("TiledInferenceEngineTests/Fixtures/noisy.png")
        let photo = input.appendingPathComponent("official-noisy.png")
        let originalBytes = try Data(contentsOf: fixture)
        try originalBytes.write(to: photo)
        let resultStore = ProcessedLibraryStore()
        func makeViewModel(_ workName: String) throws -> StudioViewModel {
            let work = directory.appendingPathComponent(workName)
            let executor = TiledPipelineExecutor(workingDirectory: work, modelRegistry: registry,
                checkpointStore: try JobCheckpointStore(workingDirectory: work))
            return StudioViewModel(workingDirectory: work, pipelineExecutor: executor,
                pipeline: TiledExecutorPipelineAdapter(executor: executor), decoder: CachingStudioImageDecoder(),
                exporter: BlockingStudioImageExporter(), scanner: BlockingLibraryScanner(),
                handoffManager: AppGroupHandoffManagerSeam(overrideBaseDirectory: directory.appendingPathComponent("unused-handoff")),
                externalOpener: SystemExternalApplicationOpener(), scopeProvider: SystemSecurityScopeGrantProvider(),
                backgroundTaskRouter: BackgroundTaskDeliveryRouter(), defaults: defaults,
                requiresOutputFolder: true, resultStore: resultStore)
        }
        let vm = try makeViewModel("processing-work")
        await vm.setOutputFolder(url: output)
        await vm.setLibraryFolder(url: input)
        await vm.ingestFile(at: photo)
        vm.exportFormat = .tiff16BitLZW
        XCTAssertEqual(vm.lumaStrength, 0.80, accuracy: 0.00001)
        XCTAssertEqual(vm.chromaStrength, 1, accuracy: 0.00001)
        vm.selectedModel = .cgnetEnsemble
        vm.startProcessing()
        let deadline = ContinuousClock.now + .seconds(180)
        while vm.isProcessing, .now < deadline { try await Task.sleep(for: .milliseconds(20)) }
        if vm.isProcessing {
            vm.cancelProcessing()
            await vm.waitForStopToFinish()
            XCTFail("The actual built-app required-model job did not finish within its deadline")
            throw CocoaError(.featureUnsupported)
        }
        XCTAssertEqual(vm.currentStage, .readyForInspection, vm.errorMessage ?? "")
        XCTAssertNil(vm.errorMessage)
        let artifacts = try XCTUnwrap(vm.completedArtifacts)
        XCTAssertEqual(artifacts.checkpoint.config.backbone, .cgnetEnsemble)
        XCTAssertEqual(artifacts.checkpoint.config.scaleFactor, .x1)
        XCTAssertTrue(artifacts.checkpoint.didRunDenoise)
        XCTAssertEqual(artifacts.sourceBuffer.width, 256)
        XCTAssertEqual(artifacts.sourceBuffer.height, 256)
        XCTAssertEqual(artifacts.outputBuffer.width, 256)
        XCTAssertEqual(artifacts.outputBuffer.height, 256)
        var absoluteChange: Float = 0
        for y in 0..<256 {
            let before = artifacts.sourceBuffer.unsafeRowPointer(y: y).assumingMemoryBound(to: Float16.self)
            let after = artifacts.outputBuffer.unsafeRowPointer(y: y).assumingMemoryBound(to: Float16.self)
            for x in 0..<256 { for channel in 0..<3 { absoluteChange += abs(Float(after[x * 4 + channel]) - Float(before[x * 4 + channel])) } }
        }
        XCTAssertGreaterThan(absoluteChange / Float(256 * 256 * 3), 0.0001, "Required trained denoising of this noisy photograph must not be identity")
        await vm.reloadOutputFolder()
        XCTAssertEqual(vm.outputDaySections.flatMap(\.items).count, 1)
        let records = try await resultStore.records(under: output)
        let record = try XCTUnwrap(records.first)
        XCTAssertEqual(records.count, 1)
        XCTAssertEqual(record.modelRawValue, BackboneModelKind.cgnetEnsemble.rawValue)
        XCTAssertEqual(record.scaleFactor, 1)
        XCTAssertEqual(record.sourceWidth, record.outputWidth)
        XCTAssertEqual(record.sourceHeight, record.outputHeight)
        XCTAssertEqual(try XCTUnwrap(record.lumaStrength), 0.80, accuracy: 0.00001)
        XCTAssertEqual(try XCTUnwrap(record.chromaStrength), 1, accuracy: 0.00001)
        let savedURL = output.appendingPathComponent(record.outputRelativePath)
        let savedBytes = try Data(contentsOf: savedURL)
        let recordedBytes = try Data(contentsOf: output.appendingPathComponent(record.metadataRelativePath))
        let savedMetadata = try MetadataExtractor.extractMetadata(from: savedURL)
        XCTAssertTrue(savedMetadata.isAlreadyDenoisedByNoiser)
        XCTAssertTrue(savedMetadata.isAlreadyDenoisedBySuite)
        XCTAssertFalse(savedMetadata.isAlreadyUpscaledBySuite)
        let history = try XCTUnwrap(savedMetadata.processingHistory?.steps.last)
        XCTAssertEqual(history.app, "com.iantruong.truongnoiser")
        XCTAssertEqual(history.operation, "denoise")
        XCTAssertEqual(history.inputWidth, 256)
        XCTAssertEqual(history.outputWidth, 256)
        XCTAssertTrue(history.model?.contains("CGNet_SIDD_x8") == true)
        XCTAssertTrue(history.model?.contains("Y 80%, C 100%") == true)
        XCTAssertEqual(try Data(contentsOf: photo), originalBytes)
        XCTAssertEqual(try Data(contentsOf: output.appendingPathComponent(record.original.relativePath)), originalBytes)

        // A new view model must recover the association from the portable folder alone.
        let reopened = try makeViewModel("reopening-work")
        reopened.lumaStrength = 0.15; reopened.chromaStrength = 0.25
        await reopened.setOutputFolder(url: output)
        reopened.selectedCollection = .output
        await reopened.reloadOutputFolder()
        let item = try XCTUnwrap(reopened.outputDaySections.flatMap(\.items).first)
        await reopened.openLibraryPhoto(item)
        XCTAssertEqual(reopened.currentStage, .readyForInspection, reopened.errorMessage ?? "")
        XCTAssertNil(reopened.comparisonUnavailableReason)
        XCTAssertEqual(reopened.displayedOutputRecord, record)
        XCTAssertEqual(reopened.sourceBuffer?.width, 256)
        XCTAssertEqual(reopened.outputBuffer?.width, 256)
        XCTAssertEqual(try XCTUnwrap(reopened.appliedLumaStrength), 0.80, accuracy: 0.00001)
        XCTAssertEqual(try XCTUnwrap(reopened.appliedChromaStrength), 1, accuracy: 0.00001)
        XCTAssertEqual(try Data(contentsOf: savedURL), savedBytes)
        XCTAssertEqual(try Data(contentsOf: output.appendingPathComponent(record.metadataRelativePath)), recordedBytes)

        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first)
        let previousKeyWindow = scene.windows.first(where: \.isKeyWindow)
        let window = UIWindow(windowScene: scene)
        window.frame = CGRect(x: 0, y: 0, width: 402, height: 874)
        window.overrideUserInterfaceStyle = .dark
        window.rootViewController = UIHostingController(rootView: StudioWorkspaceView(viewModel: reopened).preferredColorScheme(.dark))
        window.makeKeyAndVisible()
        defer {
            reopened.viewerDidDismiss()
            window.isHidden = true; window.rootViewController = nil
            previousKeyWindow?.makeKey()
        }
        window.layoutIfNeeded()
        try await Task.sleep(for: .milliseconds(900))
        func tiledViews(in view: UIView) -> [MMapTiledImageView] {
            (view as? MMapTiledImageView).map { [$0] } ?? view.subviews.flatMap { tiledViews(in: $0) }
        }
        XCTAssertEqual(tiledViews(in: window).count, 2, "The actual native viewer must mount both original and saved result")
        let capture = UIGraphicsImageRenderer(bounds: window.bounds).image { _ in
            XCTAssertTrue(window.drawHierarchy(in: window.bounds, afterScreenUpdates: true))
        }
        let attachment = XCTAttachment(image: capture)
        attachment.name = "Noiser-Best-CGNet-x8-built-model-saved-before-after"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    func testUnpresentedShareRetiresOnTeardownExactlyOnce() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("UnpresentedShare-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let session = PhotoShareSession(id: UUID(), surface: .viewer, fileURLs: [], directory: directory)
        let anchor = PhotoShareAnchor()
        anchor.loadViewIfNeeded()
        var retired: [UUID] = []
        anchor.update(session: session, onDismiss: { retired.append($0) })
        await Task.yield()
        anchor.tearDown(); anchor.tearDown()
        for _ in 0..<100 where retired.isEmpty { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertEqual(retired, [session.id])
    }

    func testObsoleteNativeShareSignalsCannotDismissTheNextSession() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("ShareIdentity-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("fixture.png")
        try XCTUnwrap(UIGraphicsImageRenderer(size: CGSize(width: 4, height: 4)).image { context in
            UIColor.systemGreen.setFill(); context.fill(CGRect(x: 0, y: 0, width: 4, height: 4))
        }.pngData()).write(to: url)
        let first = PhotoShareSession(id: UUID(), surface: .viewer, fileURLs: [url], directory: directory)
        let second = PhotoShareSession(id: UUID(), surface: .viewer, fileURLs: [url], directory: directory)
        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first)
        let previousKeyWindow = scene.windows.first(where: \.isKeyWindow)
        let window = UIWindow(windowScene: scene)
        window.frame = CGRect(x: 0, y: 0, width: 402, height: 874)
        let parent = UIViewController(), anchor = PhotoShareAnchor()
        window.rootViewController = parent; window.makeKeyAndVisible()
        parent.addChild(anchor); parent.view.addSubview(anchor.view)
        anchor.view.frame = parent.view.bounds; anchor.didMove(toParent: parent)
        addTeardownBlock { @MainActor in
            // Keep the presenting window alive until UIKit finishes dismissing its share service,
            // including a failed assertion's cleanup. Detaching it mid-transition can crash the host.
            if parent.presentedViewController != nil {
                await withCheckedContinuation { continuation in
                    parent.dismiss(animated: false) { continuation.resume() }
                }
            }
            anchor.tearDown()
            window.isHidden = true
            window.rootViewController = nil
            previousKeyWindow?.makeKey()
        }
        func waitFor(_ phase: String, _ predicate: () -> Bool) async throws {
            // A fresh hosted simulator starts the system share service lazily. Its first native
            // presentation can exceed 1.5 seconds even while the main actor remains responsive.
            let deadline = ContinuousClock.now + .seconds(15)
            while .now < deadline {
                if predicate() { return }
                try await Task.sleep(for: .milliseconds(20))
            }
            XCTFail("The native share transition did not complete: \(phase); presented=\(String(describing: parent.presentedViewController)); presenting=\(parent.presentedViewController?.isBeingPresented == true)")
            throw CocoaError(.featureUnsupported)
        }
        var retired: [UUID] = []
        anchor.update(session: first, onDismiss: { retired.append($0) })
        try await waitFor("first presentation") {
            parent.presentedViewController is UIActivityViewController &&
                parent.presentedViewController?.isBeingPresented == false &&
                parent.presentedViewController?.transitionCoordinator == nil
        }
        let oldController = try XCTUnwrap(parent.presentedViewController as? UIActivityViewController)
        let oldPresentation = try XCTUnwrap(oldController.presentationController)
        let oldCompletion = oldController.completionWithItemsHandler
        await withCheckedContinuation { continuation in parent.dismiss(animated: false) { continuation.resume() } }
        anchor.presentationControllerDidDismiss(oldPresentation)
        try await waitFor("first retirement") { retired == [first.id] }
        anchor.update(session: second, onDismiss: { retired.append($0) })
        try await waitFor("second presentation") {
            parent.presentedViewController is UIActivityViewController &&
                parent.presentedViewController?.isBeingPresented == false &&
                parent.presentedViewController?.transitionCoordinator == nil
        }
        let current = try XCTUnwrap(parent.presentedViewController as? UIActivityViewController)
        oldCompletion?(nil, false, nil, nil)
        anchor.presentationControllerDidDismiss(oldPresentation)
        try await Task.sleep(for: .milliseconds(200))
        XCTAssertTrue(parent.presentedViewController === current)
        XCTAssertFalse(current.isBeingDismissed)
        XCTAssertEqual(retired, [first.id])
        let currentPresentation = try XCTUnwrap(current.presentationController)
        await withCheckedContinuation { continuation in parent.dismiss(animated: false) { continuation.resume() } }
        anchor.presentationControllerDidDismiss(currentPresentation)
        try await waitFor("second retirement") { retired == [first.id, second.id] }
    }

    func testBlockedShareRequestRetiresInsteadOfWedging() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("BlockedShare-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("fixture.png")
        try Data([0]).write(to: url)
        let session = PhotoShareSession(id: UUID(), surface: .viewer, fileURLs: [url], directory: directory)
        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first)
        let window = UIWindow(windowScene: scene)
        window.frame = CGRect(x: 0, y: 0, width: 402, height: 874)
        let parent = UIViewController(), anchor = PhotoShareAnchor()
        window.rootViewController = parent; window.makeKeyAndVisible()
        parent.addChild(anchor); parent.view.addSubview(anchor.view)
        anchor.view.frame = parent.view.bounds; anchor.didMove(toParent: parent)
        defer { anchor.tearDown(); window.isHidden = true; window.rootViewController = nil }
        let inspector = UIViewController(); inspector.modalPresentationStyle = .pageSheet
        await withCheckedContinuation { continuation in parent.present(inspector, animated: false) { continuation.resume() } }
        var retired = false
        anchor.update(session: session, onDismiss: { _ in retired = true })
        try await Task.sleep(for: .milliseconds(150))
        XCTAssertTrue(parent.presentedViewController === inspector, "Sharing must not replace the user's newer native sheet")
        XCTAssertTrue(retired, "An unpresentable share must release its request so the next Export tap is usable")
        await withCheckedContinuation { continuation in parent.dismiss(animated: false) { continuation.resume() } }
    }

    func testCancelledDeletedPreviewCannotClearTheNextRenderedPhoto() async throws {
        let fixture = DeletedPreviewFixture()
        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first)
        let window = UIWindow(windowScene: scene)
        window.frame = CGRect(x: 0, y: 0, width: 402, height: 874)
        window.rootViewController = UIHostingController(rootView: DeletedPreviewFixtureView(fixture: fixture))
        window.makeKeyAndVisible()
        defer { window.isHidden = true; window.rootViewController = nil; fixture.releasePendingReads() }
        func waitFor(_ condition: () -> Bool) async throws {
            for _ in 0..<300 {
                if condition() { return }
                try await Task.sleep(for: .milliseconds(5))
            }
            XCTFail("The actual deleted-photo preview did not finish the requested transition")
            throw CocoaError(.featureUnsupported)
        }
        func renderedPhoto(in view: UIView) -> CGImage? {
            if let scroll = view as? OriginalScrollView { return scroll.imageView.image?.cgImage }
            return view.subviews.compactMap { renderedPhoto(in: $0) }.first
        }
        let first = fixture.entry.id
        try await waitFor { fixture.pending[first] != nil }
        fixture.entry = DeletedPreviewFixture.makeEntry()
        let second = fixture.entry.id
        try await waitFor { fixture.pending[second] != nil }
        let image = try XCTUnwrap(UIGraphicsImageRenderer(size: CGSize(width: 30, height: 40)).image { context in
            UIColor.systemGreen.setFill(); context.fill(CGRect(x: 0, y: 0, width: 30, height: 40))
        }.cgImage)
        fixture.complete(second, with: image)
        try await waitFor { renderedPhoto(in: window) === image }
        // The production view model returns nil when an obsolete disk read drains after task
        // cancellation. That old completion used to erase the already rendered second photo.
        fixture.complete(first, with: nil)
        try await waitFor { fixture.completed.contains(first) }
        XCTAssertTrue(fixture.cancelledCompletions.contains(first), "SwiftUI must actually cancel the previous entry's task")
        try await Task.sleep(for: .milliseconds(200))
        XCTAssertTrue(renderedPhoto(in: window) === image)
    }

    func testEmptyRecentlyDeletedAndRoomierControlsInNativeSheet() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("GalleryPolish-\(UUID())")
        let suite = "GalleryPolish-\(UUID())", defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { try? FileManager.default.removeItem(at: directory); defaults.removePersistentDomain(forName: suite) }
        let vm = StudioViewModel(workingDirectory: directory, defaults: defaults)
        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first)
        let window = UIWindow(windowScene: scene)
        window.frame = CGRect(x: 0, y: 0, width: 402, height: 874)
        let parent = UIViewController(); window.rootViewController = parent; window.makeKeyAndVisible()
        defer { parent.dismiss(animated: false); window.isHidden = true }
        func attach(_ name: String) {
            let capture = UIGraphicsImageRenderer(bounds: window.bounds).image { _ in
                window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
            }
            let attachment = XCTAttachment(image: capture); attachment.name = name; attachment.lifetime = .keepAlways; add(attachment)
        }
        let controls = UIHostingController(rootView: VStack {
            Spacer()
            ProcessingProgressBarView(progress: JobProgressPresentation(phase: .tileCommitted, completedTiles: 117, totalTiles: 234), onStop: {}, onShowDetails: {})
            LibraryCollectionBar(selection: .constant(.input)).padding(.bottom, 16)
        }.background(.black).preferredColorScheme(.dark))
        parent.addChild(controls); parent.view.addSubview(controls.view); controls.view.frame = parent.view.bounds
        controls.didMove(toParent: parent)
        try await Task.sleep(for: .milliseconds(200)); attach("roomier-gallery-controls")
        let trash = UIHostingController(rootView: RecentlyDeletedView(viewModel: vm))
        await withCheckedContinuation { continuation in parent.present(trash, animated: false) { continuation.resume() } }
        try await Task.sleep(for: .milliseconds(250))
        XCTAssertTrue(vm.recentlyDeleted.isEmpty)
        XCTAssertNotNil(trash.view.window)
        attach("recently-deleted-empty-cam-style")
    }
}

@MainActor @Observable private final class DeletedPreviewFixture {
    var entry = DeletedPreviewFixture.makeEntry()
    var pending: [UUID: CheckedContinuation<CGImage?, Never>] = [:]
    var completed: Set<UUID> = []
    var cancelledCompletions: Set<UUID> = []
    static func makeEntry() -> OutputTrashEntry {
        OutputTrashEntry(id: UUID(), schemaVersion: 1, authoredBy: "com.iantruong.truongnoiser", deletedAt: Date(),
            filename: "fixture.dng", record: nil, state: .trashed,
            files: [.init(originalPath: "output/fixture.dng", storedName: "0.dng", sha256: String(repeating: "0", count: 64), bytes: 4)])
    }
    func read(_ entry: OutputTrashEntry) async -> CGImage? {
        let image = await withCheckedContinuation { pending[entry.id] = $0 }
        if Task.isCancelled { cancelledCompletions.insert(entry.id) }
        completed.insert(entry.id)
        return image
    }
    func complete(_ id: UUID, with image: CGImage?) { pending.removeValue(forKey: id)?.resume(returning: image) }
    func releasePendingReads() {
        let reads = pending.values
        pending.removeAll()
        for read in reads { read.resume(returning: nil) }
    }
}

private struct DeletedPreviewFixtureView: View {
    let fixture: DeletedPreviewFixture
    var body: some View {
        DeletedPhotoPreview(entry: fixture.entry, loadThumbnail: { await fixture.read($0) },
            previous: {}, next: {}, close: {})
    }
}


extension NoiserNativePresentationTests {
    func testBuiltAppDeclaresOnlyItsOwnCompanionDocumentHandler() throws {
        let own = SuiteCompanionApp.truongNoiser
        let other = SuiteCompanionApp.truongScaler
        let declarations = try XCTUnwrap(Bundle.main.object(forInfoDictionaryKey: "CFBundleDocumentTypes") as? [[String: Any]])
        XCTAssertEqual(declarations.compactMap { $0["LSItemContentTypes"] as? [String] }.flatMap { $0 }, [own.handoffDocumentTypeIdentifier])
        let exported = try XCTUnwrap(Bundle.main.object(forInfoDictionaryKey: "UTExportedTypeDeclarations") as? [[String: Any]])
        let imported = try XCTUnwrap(Bundle.main.object(forInfoDictionaryKey: "UTImportedTypeDeclarations") as? [[String: Any]])
        XCTAssertEqual(exported.compactMap { $0["UTTypeIdentifier"] as? String }, [own.handoffDocumentTypeIdentifier])
        XCTAssertEqual(imported.compactMap { $0["UTTypeIdentifier"] as? String }, [other.handoffDocumentTypeIdentifier])
        XCTAssertEqual(UTType(filenameExtension: own.handoffDocumentExtension)?.identifier, own.handoffDocumentTypeIdentifier)
        XCTAssertEqual(Bundle.main.object(forInfoDictionaryKey: "LSSupportsOpeningDocumentsInPlace") as? Bool, false)
    }

    func testNativeCompanionOpenInMenuRetainsCopyUntilDismissalWithUpdatedCompanionInstalled() async throws {
        guard ProcessInfo.processInfo.environment["TRUONGSUITE_COMPANION_AVAILABLE"] == "1" else {
            throw XCTSkip("Install updated TruongScaler on the task-owned Simulator before this two-app integration case")
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("TruongNoiser-NativeCompanion-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let input = directory.appendingPathComponent("native-fixture.png")
        let image = UIGraphicsImageRenderer(size: CGSize(width: 8, height: 6)).image { context in
            UIColor.systemGreen.setFill(); context.fill(CGRect(x: 0, y: 0, width: 8, height: 6))
        }
        try XCTUnwrap(image.pngData()).write(to: input)
        var document: PortableSuiteHandoff.Document? = try PortableSuiteHandoff.createDocument(exportedImageURL: input,
            originalFileName: input.lastPathComponent, sourceApp: .truongNoiser, targetApp: .truongScaler,
            isDenoised: false, isUpscaled: false, globalMaxLuminance: 1, upscaleFactor: nil,
            globalMaxLinearComponent: 1, directory: directory.appendingPathComponent("documents"))
        weak let weakDocument = document
        var session: CompanionHandoffSession? = CompanionHandoffSession(document: try XCTUnwrap(document))
        let file = try XCTUnwrap(session?.fileURL), id = try XCTUnwrap(session?.id)
        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first)
        let previous = scene.windows.first(where: \.isKeyWindow)
        let owner = UIViewController(), window = UIWindow(windowScene: scene)
        window.frame = CGRect(x: 0, y: 0, width: 402, height: 874)
        window.rootViewController = owner
        let button = UIButton(type: .system)
        button.frame = CGRect(x: 275, y: 780, width: 48, height: 48)
        button.setTitle("Open", for: .normal)
        owner.view.addSubview(button)
        let anchor = CompanionHandoffAnchor()
        owner.addChild(anchor); button.addSubview(anchor.view); anchor.view.frame = button.bounds; anchor.didMove(toParent: owner)
        window.makeKeyAndVisible(); window.layoutIfNeeded()
        defer {
            anchor.tearDown()
            window.isHidden = true; window.rootViewController = nil; previous?.makeKey()
        }
        try await Task.sleep(for: .milliseconds(250))
        let initialTransfers = CompanionDocumentTransfer.activeTransferCount
        var finishes: [(UUID, Bool)] = []
        anchor.update(session: session) { finishes.append(($0, $1)) }
        document = nil; session = nil
        try await Task.sleep(for: .milliseconds(800))
        XCTAssertEqual(CompanionDocumentTransfer.activeTransferCount, initialTransfers + 1,
            "Updated TruongScaler must accept its registered document in the actual OS Open In menu")
        XCTAssertTrue(finishes.isEmpty, "The document must remain owned while the native menu is visible")
        XCTAssertNotNil(weakDocument)
        XCTAssertTrue(FileManager.default.fileExists(atPath: file.path))
        XCTAssertEqual(anchor.view.superview, button)
        let capture = UIGraphicsImageRenderer(bounds: window.bounds).image { _ in
            _ = window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }
        let attachment = XCTAttachment(image: capture)
        attachment.name = "TruongNoiser-native-companion-open-in-menu"; attachment.lifetime = .keepAlways; add(attachment)
        anchor.tearDown()
        try await Task.sleep(for: .milliseconds(400))
        XCTAssertEqual(finishes.count, 1)
        XCTAssertEqual(finishes.first?.0, id)
        XCTAssertEqual(finishes.first?.1, false)
        XCTAssertEqual(CompanionDocumentTransfer.activeTransferCount, initialTransfers)
        XCTAssertNil(weakDocument)
        XCTAssertFalse(FileManager.default.fileExists(atPath: file.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: input.path))
    }
}
