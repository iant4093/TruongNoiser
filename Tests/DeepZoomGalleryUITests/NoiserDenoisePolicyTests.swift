import XCTest
import Foundation
import TruongNoiserCore
import MMapCanvas
import TiledInferenceEngine
import BackgroundJobManager
@testable import DeepZoomGalleryUI

/// Noiser app policy, independent control ownership and cross-app direction. Fake inference is
/// explicit test scaffolding; trained model/reference fidelity is exercised by the engine suite.
@MainActor
final class NoiserDenoisePolicyTests: XCTestCase {
    private var directory: URL!
    private var defaults: UserDefaults!
    private var suite: String!

    override func setUp() async throws {
        try await super.setUp()
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("NoiserPolicy-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        suite = "NoiserPolicy-\(UUID())"
        defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    }

    override func tearDown() async throws {
        if let suite { defaults?.removePersistentDomain(forName: suite) }
        defaults = nil; suite = nil
        if let directory { try? FileManager.default.removeItem(at: directory) }
        directory = nil
        try await super.tearDown()
    }

    private func source(_ name: String = "input.png") throws -> URL {
        let url = directory.appendingPathComponent(name)
        try Data([1, 2, 3]).write(to: url)
        return url
    }

    private func makeViewModel(pipeline: any StudioPipelineExecuting,
        decoder: any StudioImageDecoding = FakeDecoder(),
        handoff: FakeHandoffManager = FakeHandoffManager(),
        opener: RecordingOpener = RecordingOpener()) -> StudioViewModel {
        let executor = TiledPipelineExecutor(workingDirectory: directory,
            modelRegistry: CoreMLModelRegistry(initialComputeUnits: .cpuOnly, availabilityPolicy: .allowDeterministicFallback),
            checkpointStore: JobCheckpointStore(directory: directory))
        return StudioViewModel(workingDirectory: directory, pipelineExecutor: executor,
            pipeline: pipeline, decoder: decoder, exporter: FakeExporter(), scanner: FakeScanner(),
            handoffManager: handoff, externalOpener: opener, scopeProvider: RecordingScopeProvider(),
            backgroundTaskRouter: BackgroundTaskDeliveryRouter(), defaults: defaults)
    }

    private func waitUntil(_ condition: () -> Bool) async -> Bool {
        let deadline = ContinuousClock.now + .seconds(5)
        while .now < deadline {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return condition()
    }

    func testNewJobDefaultsToNativeResolutionFastNAFNetAndIndependentStrengths() throws {
        let url = try source(), id = UUID()
        let config = NoiserDenoisePolicy.makeJobConfig(id: id, sourceImageURL: url,
            exportFormat: .tiff16BitLZW, sessionState: nil)
        XCTAssertEqual(config.id, id)
        XCTAssertEqual(config.sourceImageURL, url)
        XCTAssertEqual(config.scaleFactor, .x1)
        XCTAssertEqual(config.backbone, .nafnetNormal)
        XCTAssertEqual(config.denoiseMode, .forceOn)
        XCTAssertEqual(config.naturalGrainAmount, 0)
        XCTAssertEqual(config.lumaStrength, 0.80, accuracy: 0.00001)
        XCTAssertEqual(config.chromaStrength, 1.00, accuracy: 0.00001)
        XCTAssertEqual(BackboneModelKind.allCases, [.nafnetNormal, .cgnet, .cgnetEnsemble])
        XCTAssertNoThrow(try NoiserDenoisePolicy.validateJobConfig(config))
    }

    func testEachSelectableModeCreatesItsOwnValidNativeResolutionRecipe() throws {
        let url = try source()
        for model in BackboneModelKind.allCases {
            let config = NoiserDenoisePolicy.makeJobConfig(sourceImageURL: url, model: model,
                lumaStrength: 0.35, chromaStrength: 0.65, exportFormat: .png16Bit, sessionState: nil)
            XCTAssertEqual(config.backbone, model)
            XCTAssertEqual(config.scaleFactor, .x1)
            XCTAssertEqual(config.denoiseMode, .forceOn)
            XCTAssertEqual(config.lumaStrength, 0.35)
            XCTAssertEqual(config.chromaStrength, 0.65)
            XCTAssertNoThrow(try NoiserDenoisePolicy.validateJobConfig(config))
        }
    }

    private func checkpoint(_ original: JobCheckpointState,
        replacingTiles tiles: TileConfiguration) throws -> JobCheckpointState {
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(original)) as? [String: Any])
        json["tileConfig"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(tiles))
        return try JSONDecoder().decode(JobCheckpointState.self,
            from: JSONSerialization.data(withJSONObject: json, options: .sortedKeys))
    }

    func testResumeRequiresTheRecordedModelsGeometryAndPreservesEnsembleIdentity() throws {
        let url = try source()
        for mode in BackboneModelKind.allCases {
            let artifacts = try StudioTestArtifacts.make(directory: directory, sourceURL: url, backbone: mode)
            for tier: DeviceTier in [.tierA_HighMemory, .tierB_ConstrainedMemory] {
                let compatible = try checkpoint(artifacts.checkpoint,
                    replacingTiles: TileConfiguration(tier: tier, backbone: mode))
                XCTAssertNoThrow(try NoiserDenoisePolicy.validateResume(compatible))
                XCTAssertEqual(compatible.config.backbone, mode)
                XCTAssertEqual(compatible.config.backbone.ensemblePassCount, mode.ensemblePassCount)
            }
            let otherFamily: BackboneModelKind = mode == .nafnetNormal ? .cgnet : .nafnetNormal
            let incompatible = try checkpoint(artifacts.checkpoint,
                replacingTiles: TileConfiguration(tier: .tierB_ConstrainedMemory, backbone: otherFamily))
            XCTAssertTrue(NoiserDenoisePolicy.resumeDisposition(for: incompatible).isRefused)
            XCTAssertThrowsError(try NoiserDenoisePolicy.validateResume(incompatible))
            XCTAssertEqual(incompatible.config.backbone, mode, "Refusal must not migrate the recorded model")
        }
    }

    func testPolicyRejectsUnsupportedScaleModelModeAndMutatedStrengths() throws {
        let baseline = NoiserDenoisePolicy.makeJobConfig(sourceImageURL: try source(),
            exportFormat: .linearDNG16Bit, sessionState: nil)
        var invalid: [DenoiseJobConfig] = []
        var scale = baseline; scale.scaleFactor = .x2; invalid.append(scale)
        var previous = baseline; previous.backbone = BackboneModelKind(rawValue: "opaque-previous-model"); invalid.append(previous)
        var mode = baseline; mode.denoiseMode = .off; invalid.append(mode)
        var grain = baseline; grain.naturalGrainAmount = 0.10; invalid.append(grain)
        for value: Float in [-0.01, 1.01, .nan, .infinity] {
            var luma = baseline; luma.lumaStrength = value; invalid.append(luma)
            var chroma = baseline; chroma.chromaStrength = value; invalid.append(chroma)
        }
        for config in invalid {
            XCTAssertThrowsError(try NoiserDenoisePolicy.validateJobConfig(config))
        }
        for strength: Float in [0, 1] {
            var boundary = baseline; boundary.lumaStrength = strength; boundary.chromaStrength = strength
            XCTAssertNoThrow(try NoiserDenoisePolicy.validateJobConfig(boundary))
        }
    }

    func testCompatibleResumeKeepsRecordedRecipeAndUnknownModelRemainsOpaque() throws {
        let artifacts = try StudioTestArtifacts.make(directory: directory, sourceURL: source(),
            lumaStrength: 0.35, chromaStrength: 0.65)
        let encoded = try JSONEncoder().encode(artifacts.checkpoint)
        let checkpoint = try JSONDecoder().decode(JobCheckpointState.self, from: encoded)
        XCTAssertNoThrow(try NoiserDenoisePolicy.validateResume(checkpoint))
        XCTAssertEqual(checkpoint.config.lumaStrength, 0.35, accuracy: 0.00001)
        XCTAssertEqual(checkpoint.config.chromaStrength, 0.65, accuracy: 0.00001)
        let previous = try StudioTestArtifacts.make(directory: directory, sourceURL: source("previous.png"),
            backbone: BackboneModelKind(rawValue: "previous-denoiser-v1"))
        XCTAssertTrue(NoiserDenoisePolicy.resumeDisposition(for: previous.checkpoint).isRefused)
        let previousRoundTrip = try JSONDecoder().decode(JobCheckpointState.self,
            from: JSONEncoder().encode(previous.checkpoint))
        XCTAssertEqual(previousRoundTrip.config.backbone.rawValue, "previous-denoiser-v1")
        XCTAssertEqual(previousRoundTrip.outputWidth, previous.checkpoint.outputWidth)
        XCTAssertEqual(previousRoundTrip.outputHeight, previous.checkpoint.outputHeight)
    }

    func testViewModelFreezesAppliedStrengthsWhileNextJobControlsChange() async throws {
        let pipeline = FakePipelineExecutor(directory: directory); pipeline.gateExecution = false
        let vm = makeViewModel(pipeline: pipeline)
        XCTAssertEqual(vm.lumaStrength, 0.80, accuracy: 0.00001)
        XCTAssertEqual(vm.chromaStrength, 1, accuracy: 0.00001)
        await vm.ingestFile(at: try source())
        vm.lumaStrength = 0.25; vm.chromaStrength = 0.70
        vm.startProcessing()
        let completed = await waitUntil { !vm.isProcessing && vm.currentStage == .readyForInspection }
        XCTAssertTrue(completed, vm.errorMessage ?? "")
        let execution = try XCTUnwrap(pipeline.recordedExecutions.last)
        XCTAssertEqual(execution.config.scaleFactor, .x1)
        XCTAssertEqual(execution.config.denoiseMode, .forceOn)
        XCTAssertEqual(execution.config.lumaStrength, 0.25, accuracy: 0.00001)
        XCTAssertEqual(execution.config.chromaStrength, 0.70, accuracy: 0.00001)
        XCTAssertEqual(vm.completedArtifacts?.checkpoint.sourceWidth, vm.completedArtifacts?.checkpoint.outputWidth)
        XCTAssertEqual(vm.completedArtifacts?.checkpoint.sourceHeight, vm.completedArtifacts?.checkpoint.outputHeight)
        vm.lumaStrength = 0.90; vm.chromaStrength = 0.10
        XCTAssertEqual(try XCTUnwrap(vm.appliedLumaStrength), 0.25, accuracy: 0.00001)
        XCTAssertEqual(try XCTUnwrap(vm.appliedChromaStrength), 0.70, accuracy: 0.00001)
        XCTAssertEqual(vm.completedArtifacts?.checkpoint.config, execution.config)
    }

    func testViewModelFreezesEnsembleRecipeBeforeAwaitAndKeepsArtifactLabel() async throws {
        let pipeline = FakePipelineExecutor(directory: directory)
        defer { pipeline.openExecutionGate() }
        let vm = makeViewModel(pipeline: pipeline)
        await vm.ingestFile(at: try source())
        vm.selectedModel = .cgnetEnsemble
        vm.lumaStrength = 0.35; vm.chromaStrength = 0.65; vm.exportFormat = .png16Bit
        vm.startProcessing()
        let started = await waitUntil { pipeline.executionCount == 1 }
        XCTAssertTrue(started, vm.errorMessage ?? "The job did not reach the test inference gate")
        // Programmatic changes simulate a queued UI update arriving after the run began.
        vm.selectedModel = .nafnetNormal
        vm.lumaStrength = 0.95; vm.chromaStrength = 0.10; vm.exportFormat = .jpeg95
        pipeline.openExecutionGate()
        let completed = await waitUntil { !vm.isProcessing && vm.currentStage == .readyForInspection }
        XCTAssertTrue(completed, vm.errorMessage ?? "")
        let config = try XCTUnwrap(pipeline.recordedExecutions.last?.config)
        XCTAssertEqual(config.backbone, .cgnetEnsemble)
        XCTAssertEqual(config.lumaStrength, 0.35)
        XCTAssertEqual(config.chromaStrength, 0.65)
        XCTAssertEqual(config.exportFormat, .png16Bit)
        XCTAssertEqual(vm.backbone, .cgnetEnsemble)
        XCTAssertEqual(vm.inspectionModelLabel, BackboneModelKind.cgnetEnsemble.displayName)
        XCTAssertEqual(vm.completedArtifacts?.checkpoint.config, config)
        XCTAssertEqual(vm.selectedModel, .nafnetNormal, "The next-job choice remains independent")
    }

    func testBatchFreezesItsModeAndStrengthsBeforeOpeningTheFirstQueuedPhoto() async throws {
        let pipeline = FakePipelineExecutor(directory: directory)
        defer { pipeline.openExecutionGate() }
        let vm = makeViewModel(pipeline: pipeline)
        let urls = try [source("first.png"), source("second.png")]
        let items = urls.map { LibraryPhotoItem(id: $0.path, fileURL: $0, filename: $0.lastPathComponent,
            capturedAt: Date(timeIntervalSince1970: 1_700_000_000), byteCount: 3) }
        vm.daySections = [LibraryDaySection(id: "test-day", displayTitle: "Test", items: items)]
        vm.selectedModel = .cgnetEnsemble
        vm.lumaStrength = 0.25; vm.chromaStrength = 0.75; vm.exportFormat = .png16Bit
        let batch = Task { await vm.batchProcessSelected(ids: Set(items.map(\.id))) }
        let started = await waitUntil { pipeline.executionCount == 1 }
        guard started else {
            batch.cancel(); pipeline.openExecutionGate(); await batch.value
            XCTFail(vm.errorMessage ?? "The batch did not reach the test inference gate")
            return
        }
        vm.selectedModel = .cgnet
        vm.lumaStrength = 0.90; vm.chromaStrength = 0.10; vm.exportFormat = .jpeg95
        pipeline.openExecutionGate()
        await batch.value
        XCTAssertFalse(vm.isBatchProcessing)
        XCTAssertEqual(pipeline.recordedExecutions.count, 2, vm.errorMessage ?? "")
        for execution in pipeline.recordedExecutions {
            XCTAssertEqual(execution.config.backbone, .cgnetEnsemble)
            XCTAssertEqual(execution.config.lumaStrength, 0.25)
            XCTAssertEqual(execution.config.chromaStrength, 0.75)
            XCTAssertEqual(execution.config.exportFormat, .png16Bit)
        }
        XCTAssertEqual(vm.selectedModel, .cgnet)
        XCTAssertEqual(vm.inspectionModelLabel, BackboneModelKind.cgnetEnsemble.displayName)
        XCTAssertEqual(pipeline.maxConcurrentExecutions, 1)
    }

    func testForwardingUnprocessedInputToScalerNeverMarksItDenoised() async throws {
        let pipeline = FakePipelineExecutor(directory: directory), handoff = FakeHandoffManager(), opener = RecordingOpener()
        let vm = makeViewModel(pipeline: pipeline, handoff: handoff, opener: opener)
        await vm.ingestFile(at: try source())
        await vm.sendToTruongScaler()
        let call = try XCTUnwrap(handoff.packageCalls.last)
        XCTAssertFalse(call.isDenoised)
        XCTAssertFalse(call.isUpscaled)
        XCTAssertNil(call.upscaleFactor)
        XCTAssertEqual(opener.openedURLs.last?.scheme, "truongscaler")
        XCTAssertTrue(pipeline.recordedExecutions.isEmpty)
    }

    func testCompletedNoiserHandoffPreservesPriorScalerDimensionsAndHistory() async throws {
        let pipeline = FakePipelineExecutor(directory: directory); pipeline.gateExecution = false
        let decoder = FakeDecoder()
        var incoming = FakeDecodeConfiguration()
        incoming.isAlreadyUpscaledBySuite = true; incoming.upscaledByAppFactor = 2
        decoder.setDefaultConfiguration(incoming)
        let handoff = FakeHandoffManager(), opener = RecordingOpener()
        let vm = makeViewModel(pipeline: pipeline, decoder: decoder, handoff: handoff, opener: opener)
        await vm.ingestFile(at: try source())
        vm.startProcessing()
        let completed = await waitUntil { !vm.isProcessing && vm.currentStage == .readyForInspection }
        XCTAssertTrue(completed, vm.errorMessage ?? "")
        await vm.sendToTruongScaler()
        let call = try XCTUnwrap(handoff.packageCalls.last)
        XCTAssertTrue(call.isDenoised)
        XCTAssertTrue(call.isUpscaled)
        XCTAssertEqual(call.upscaleFactor, 2)
        XCTAssertEqual(opener.openedURLs.last?.scheme, "truongscaler")
        XCTAssertEqual(vm.completedArtifacts?.checkpoint.config.scaleFactor, .x1)
    }
}
