import Foundation
import TruongNoiserCore

/// New denoise jobs retain the source's dimensions and freeze the selected trained model/mode.
public enum NoiserDenoisePolicy {
    public static let denoiseMode: DenoiseMode = .forceOn
    public static let scaleFactor: ScaleFactor = .x1
    public static let backbone: BackboneModelKind = .nafnetNormal
    public static let summary = "Noise reduction · original resolution"
    public static func makeJobConfig(id: UUID = UUID(), sourceImageURL: URL,
        model: BackboneModelKind = backbone, naturalGrainAmount: Float = 0,
        lumaStrength: Float = 0.80, chromaStrength: Float = 1,
        exportFormat: ExportFormat, sessionState: PipelineSessionState?) -> DenoiseJobConfig {
        DenoiseJobConfig(id: id, sourceImageURL: sourceImageURL, scaleFactor: .x1,
            backbone: model, denoiseMode: .forceOn, naturalGrainAmount: 0,
            lumaStrength: lumaStrength, chromaStrength: chromaStrength,
            exportFormat: exportFormat, sessionState: sessionState)
    }
    public enum ConfigurationError: Error, LocalizedError, Sendable {
        case incompatibleSavedJob
        public var errorDescription: String? {
            "This job does not match Noiser's validated native-resolution model contract. Its files were left unchanged."
        }
    }
    public static func validateJobConfig(_ config: DenoiseJobConfig) throws {
        guard config.scaleFactor == .x1, config.backbone.isSupported,
              config.denoiseMode == .forceOn, config.lumaStrength.isFinite,
              config.chromaStrength.isFinite, (0...1).contains(config.lumaStrength),
              (0...1).contains(config.chromaStrength), config.naturalGrainAmount == 0 else {
            throw ConfigurationError.incompatibleSavedJob
        }
    }
    public enum ResumeDisposition: Sendable, Equatable {
        case compatible
        case refused(reason: String)
        public var isRefused: Bool { if case .refused = self { return true }; return false }
    }
    public static func resumeDisposition(for checkpoint: JobCheckpointState) -> ResumeDisposition {
        guard (try? validateJobConfig(checkpoint.config)) != nil,
              checkpoint.normalizationPeakVersion == JobCheckpointState.currentNormalizationPeakVersion,
              checkpoint.tileConfig == TileConfiguration(tier: checkpoint.tileConfig.tier,
                  backbone: checkpoint.config.backbone),
              checkpoint.sourceWidth > 0, checkpoint.sourceHeight > 0,
              checkpoint.outputWidth == checkpoint.sourceWidth,
              checkpoint.outputHeight == checkpoint.sourceHeight else {
            return .refused(reason: ConfigurationError.incompatibleSavedJob.localizedDescription)
        }
        return .compatible
    }
    public static func validateResume(_ checkpoint: JobCheckpointState) throws {
        if resumeDisposition(for: checkpoint).isRefused { throw ConfigurationError.incompatibleSavedJob }
    }
}
