import XCTest
import UIKit
import SwiftUI
import CoreGraphics
import Observation
import CryptoKit
import TruongNoiserCore
@testable import DeepZoomGalleryUI

/// Opt-in, repeatable README captures using only private libraries and shipped trained models.
/// Normal CI skips capture; model/native regression suites remain separate required checks.
@MainActor
final class ReadmeScreenshotCaptureTests: XCTestCase {
    /// Opt-in documentation capture of the production screens, with real required-model output.
    /// The pinned upstream noisy demo and its attribution live in the maintained README fixture set.
    /// Normal test runs never write documentation or populate the person's photo libraries.
    func testCaptureREADMEProductionScreenshots() async throws {
        guard let capturePath = ProcessInfo.processInfo.environment["TRUONGNOISER_README_SCREENSHOT_DIR"],
              !capturePath.isEmpty else {
            throw XCTSkip("Set TRUONGNOISER_README_SCREENSHOT_DIR to capture the README screens")
        }
        let captures = URL(fileURLWithPath: capturePath, isDirectory: true)
        try FileManager.default.createDirectory(at: captures, withIntermediateDirectories: true)
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("README-Capture-\(UUID())", isDirectory: true)
        let directory = scratch.appendingPathComponent("TruongNoiser Demo", isDirectory: true)
        let input = directory.appendingPathComponent("Sample photos", isDirectory: true)
        let output = directory.appendingPathComponent("Denoised photos", isDirectory: true)
        for folder in [input, output] { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true) }
        let suite = "README-Capture-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer {
            defaults.removePersistentDomain(forName: suite)
            try? FileManager.default.removeItem(at: scratch)
        }
        let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("TruongNoiserHostTests/Fixtures/README", isDirectory: true)
        let samples = [
            (fixture: "official-noisy-demo.png", name: "Noisy text detail.png"),
        ]
        var photos: [URL] = []
        for sample in samples {
            let destination = input.appendingPathComponent(sample.name)
            try FileManager.default.copyItem(at: fixtures.appendingPathComponent(sample.fixture), to: destination)
            photos.append(destination)
        }
        // The public production initializer requires the shipped trained-model bundle.
        // Do not replace this with the deterministic test fallback: these are README photographs.
        let vm = StudioViewModel(workingDirectory: directory.appendingPathComponent("work"), defaults: defaults)
        await vm.setOutputFolder(url: output)
        await vm.setLibraryFolder(url: input)
        XCTAssertEqual(vm.daySections.flatMap(\.items).count, samples.count)
        XCTAssertNil(vm.errorMessage)
        let scene = try XCTUnwrap(UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first)
        let previousKeyWindow = scene.windows.first(where: \.isKeyWindow)
        let window = UIWindow(windowScene: scene)
        window.frame = CGRect(x: 0, y: 0, width: 402, height: 874)
        window.overrideUserInterfaceStyle = .dark
        let library = UIHostingController(rootView: LibraryScreen(viewModel: vm).preferredColorScheme(.dark))
        window.rootViewController = library
        window.makeKeyAndVisible()
        defer {
            library.dismiss(animated: false)
            window.isHidden = true
            window.rootViewController = nil
            previousKeyWindow?.makeKey()
            vm.viewerDidDismiss()
        }
        func settle() async throws {
            window.layoutIfNeeded()
            try await Task.sleep(for: .milliseconds(800))
            window.layoutIfNeeded()
        }
        var capturedNames: [String] = []
        func capture(_ name: String) throws {
            XCTAssertEqual(window.bounds.size, CGSize(width: 402, height: 874))
            let format = UIGraphicsImageRendererFormat()
            format.scale = 3
            format.opaque = true
            let image = UIGraphicsImageRenderer(bounds: window.bounds, format: format).image { _ in
                XCTAssertTrue(window.drawHierarchy(in: window.bounds, afterScreenUpdates: true))
            }
            let pixels = try XCTUnwrap(image.cgImage)
            XCTAssertEqual(pixels.width, 1206)
            XCTAssertEqual(pixels.height, 2622)
            try XCTUnwrap(image.pngData()).write(to: captures.appendingPathComponent(name + ".png"), options: .atomic)
            let attachment = XCTAttachment(image: image)
            attachment.name = "README-" + name
            attachment.lifetime = .keepAlways
            add(attachment)
            capturedNames.append(name + ".png")
        }
        try await settle()
        try capture("input-library")

        vm.exportFormat = .tiff16BitLZW
        for photo in photos {
            let model: BackboneModelKind = .cgnet
            vm.selectedModel = model
            await vm.ingestFile(at: photo)
            vm.startProcessing()
            let deadline = ContinuousClock.now + .seconds(180)
            while vm.isProcessing, .now < deadline { try await Task.sleep(for: .milliseconds(30)) }
            XCTAssertFalse(vm.isProcessing, "The required-model screenshot job exceeded its deadline")
            if vm.isProcessing {
                vm.cancelProcessing()
                await vm.waitForStopToFinish()
                throw CocoaError(.featureUnsupported)
            }
            XCTAssertEqual(vm.currentStage, .readyForInspection, vm.errorMessage ?? "")
            XCTAssertTrue(vm.isProcessedPhoto)
            XCTAssertEqual(vm.completedArtifacts?.checkpoint.config.backbone, model)
            XCTAssertEqual(vm.completedArtifacts?.checkpoint.outputWidth, 256)
            XCTAssertEqual(vm.completedArtifacts?.checkpoint.outputHeight, 256)
            guard vm.currentStage == .readyForInspection else { throw CocoaError(.featureUnsupported) }
        }
        await vm.reloadOutputFolder()
        XCTAssertEqual(vm.outputDaySections.flatMap(\.items).count, samples.count)
        // Reopen the saved denoised result to exercise the portable before/after record, rather than
        // holding an unpersisted in-memory canvas only for a screenshot.
        let records = try await ProcessedLibraryStore().records(under: output)
        let denoisedRecord = try XCTUnwrap(records.first { $0.original.filename == samples[0].name })
        XCTAssertEqual(denoisedRecord.modelRawValue, BackboneModelKind.cgnet.rawValue)
        let denoisedURL = output.appendingPathComponent(denoisedRecord.outputRelativePath)
        let denoisedResult = try XCTUnwrap(vm.outputDaySections.flatMap(\.items).first { $0.fileURL == denoisedURL })
        vm.selectedCollection = .output
        await vm.openLibraryPhoto(denoisedResult)
        XCTAssertEqual(vm.currentStage, .readyForInspection, vm.errorMessage ?? "")
        XCTAssertNil(vm.comparisonUnavailableReason)
        let viewer = UIHostingController(rootView: StudioWorkspaceView(viewModel: vm).preferredColorScheme(.dark))
        viewer.modalPresentationStyle = .fullScreen
        await withCheckedContinuation { continuation in library.present(viewer, animated: false) { continuation.resume() } }
        try await settle()
        try capture("processed-viewer")

        // Exercise the production comparison viewer's shared zoom transform. This is a native
        // viewport capture of the same saved original/result pair, not a resized composite.
        func comparisonScrollView(_ view: UIView) -> SplitZoomScrollView? {
            if let scroll = view as? SplitZoomScrollView { return scroll }
            return view.subviews.compactMap { comparisonScrollView($0) }.first
        }
        let comparison = try XCTUnwrap(comparisonScrollView(viewer.view))
        let detailRect = CGRect(x: 64, y: 80, width: 128, height: 128)
        comparison.zoom(to: detailRect, animated: false)
        try await settle()
        XCTAssertGreaterThan(comparison.zoomScale, comparison.minimumZoomScale * 1.9)
        XCTAssertGreaterThan(comparison.contentSize.width, comparison.bounds.width * 1.9)
        let detailZoom: [String: Any] = [
            "requestedCanvasRect": ["x": Double(detailRect.minX), "y": Double(detailRect.minY),
                                    "width": Double(detailRect.width), "height": Double(detailRect.height)],
            "canvasWidth": 256, "canvasHeight": 256,
            "zoomScale": Double(comparison.zoomScale),
            "relativeToFit": Double(comparison.zoomScale / comparison.minimumZoomScale),
            "comparison": "Left: unchanged noisy input; right: saved CGNet (Better) output; shared native viewer transform",
        ]
        try capture("detail-comparison")
        await withCheckedContinuation { continuation in library.dismiss(animated: false) { continuation.resume() } }
        vm.viewerDidDismiss()
        try await settle()
        try capture("output-library")

        vm.selectedCollection = .input
        await vm.ingestFile(at: photos[0])
        XCTAssertEqual(vm.metadata?.originalWidth, 256)
        vm.selectedModel = .cgnet
        let options = UIHostingController(rootView: DenoiseOptionsFormSheet(viewModel: vm, onPickFileOrFolder: {})
            .preferredColorScheme(.dark))
        options.modalPresentationStyle = .pageSheet
        if let sheet = options.sheetPresentationController { sheet.detents = [.large()] }
        await withCheckedContinuation { continuation in library.present(options, animated: false) { continuation.resume() } }
        try await settle()
        // Use the real form's native scroll view to expose the existing model and export controls.
        func formScrollView(_ view: UIView) -> UIScrollView? {
            if let scroll = view as? UIScrollView, scroll.contentSize.height > scroll.bounds.height + 100 { return scroll }
            return view.subviews.compactMap { formScrollView($0) }.first
        }
        let form = try XCTUnwrap(formScrollView(options.view))
        let bottom = max(-form.adjustedContentInset.top, form.contentSize.height - form.bounds.height + form.adjustedContentInset.bottom)
        form.setContentOffset(CGPoint(x: 0, y: min(460, bottom)), animated: false)
        try await settle()
        try capture("model-settings")
        await withCheckedContinuation { continuation in library.dismiss(animated: false) { continuation.resume() } }

        XCTAssertEqual(capturedNames.count, 5)
        let attribution = try JSONSerialization.jsonObject(with: Data(contentsOf: fixtures.appendingPathComponent("fixture-provenance.json"))) as? [String: Any]
        let sourceHashes = try Dictionary(uniqueKeysWithValues: samples.map { sample in
            let data = try Data(contentsOf: fixtures.appendingPathComponent(sample.fixture))
            return (sample.fixture, SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
        })
        let manifest: [String: Any] = [
            "capture": "Actual production SwiftUI/UIKit screens in an iOS simulator, 402×874 points at 3×; private temporary libraries",
            "processing": "Real StudioViewModel production initializer (.requireProductionModels), CGNet (Better), native 1× with luminance/color strengths 80%/100%; the exact upstream 256×256 noisy demo is saved as a 256×256 TIFF result and reopened through its portable before/after record",
            "settings": "Current three-choice production form with CGNet (Better) selected for the next job and luminance/color controls",
            "qualityScope": "Screenshots illustrate the interface; they do not establish physical-phone performance or image-quality superiority",
            "screenshots": capturedNames,
            "detailComparison": detailZoom,
            "fixtureSHA256": sourceHashes,
            "photographSources": attribution?["sources"] ?? [],
            "fixtureProvenance": "Tests/TruongNoiserHostTests/Fixtures/README/fixture-provenance.json",
        ]
        try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
            .write(to: captures.appendingPathComponent("provenance.json"), options: .atomic)
    }

}
