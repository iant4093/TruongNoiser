import Foundation

/// Noiser processes at native resolution; legacy dimensions remain decodable for inspection.
public enum ScaleFactor: Int, Codable, CaseIterable, Sendable, Identifiable {
    case x1 = 1
    case x2 = 2
    case x4 = 4

    public static var allCases: [Self] { [.x1] }
    public var id: Int { rawValue }

    public var displayName: String {
        switch self {
        case .x1: return "Original resolution"
        case .x2: return "2× Super-Resolution"
        case .x4: return "4× Super-Resolution"
        }
    }
}

/// Denoiser identity, preserved verbatim on checkpoints and portable results.
public struct BackboneModelKind: RawRepresentable, Codable, CaseIterable, Sendable, Identifiable, Hashable {
    public let rawValue: String
    public init(rawValue: String) { self.rawValue = rawValue }
    public static let nafnetNormal = Self(rawValue: "NAFNet_w64")
    public static let cgnet = Self(rawValue: "CGNet_SIDD")
    public static let cgnetEnsemble = Self(rawValue: "CGNet_SIDD_x8")
    public static let restormerAstro = Self(rawValue: "Restormer_Real")
    public static let unavailable = Self(rawValue: "unavailable-model")
    public static var allCases: [Self] { [.nafnetNormal, .cgnet, .cgnetEnsemble] }
    public var id: String { rawValue }
    public var isSupported: Bool { Self.allCases.contains(self) }
    /// The saved mode remains distinct even when two modes use the same trained weights.
    public var ensemblePassCount: Int { self == .cgnetEnsemble ? 8 : 1 }
    public var baseModel: Self { self == .cgnetEnsemble ? .cgnet : self }
    public var displayName: String {
        switch self {
        case .nafnetNormal: return "NAFNet (Fast)"
        case .cgnet: return "CGNet (Better)"
        case .cgnetEnsemble: return "CGNet (Best, 8-pass)"
        case .restormerAstro: return "Restormer (not installed)"
        default: return "Previous model"
        }
    }
    public var processingHint: String {
        switch self {
        case .nafnetNormal: return "One NAFNet prediction per tile. The default for quick processing."
        case .cgnet: return "One CGNet prediction per tile."
        case .cgnetEnsemble: return "Uses 8 predictions per tile. Expect slower processing; Stop is checked between predictions."
        default: return "This saved photo keeps its original model identity."
        }
    }

    /// The official SIDD regressor returns raw overshoot. Allow
    /// bounded edge ringing before display-domain restoration, but refuse large excursions and
    /// non-finite samples. Adversarial checker/noise cases remain errors, never silent fallbacks.
    public var predictionSampleRange: ClosedRange<Float> {
        -0.25...1.25
    }
    public init(from decoder: Decoder) throws { rawValue = try decoder.singleValueContainer().decode(String.self) }
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer(); try container.encode(rawValue)
    }
}

/// Legacy stage flags remain decodable; new Noiser jobs require forceOn.
public enum DenoiseMode: String, Codable, CaseIterable, Sendable, Identifiable {
    case auto
    case forceOn
    case off

    public var id: String { rawValue }
}

/// Target export container format (`README.md`).
public enum ExportFormat: String, Codable, CaseIterable, Sendable, Identifiable {
    case autoMatchSource
    case linearDNG16Bit
    case tiff16BitLZW
    case heic10BitHDR
    case png16Bit
    case jpeg95

    public var id: String { rawValue }
}

/// Session-level pipeline state carried from ingestion/handoff into execution, resume, export and
/// outgoing handoff. `nil` means "preserve the value already on the metadata packet"; an explicit
/// `false` clears an inherited true marker.
public struct PipelineSessionState: Codable, Sendable, Equatable {
    public var isDenoised: Bool?
    public var isUpscaled: Bool?
    public var upscaleFactor: Int?
    public var linearHeadroomScale: Float?
    public var isDenoisedByNoiser: Bool?
    public var processingHistory: ImageProcessingHistory?

    public init(
        isDenoised: Bool? = nil,
        isUpscaled: Bool? = nil,
        upscaleFactor: Int? = nil,
        linearHeadroomScale: Float? = nil,
        isDenoisedByNoiser: Bool? = nil,
        processingHistory: ImageProcessingHistory? = nil
    ) {
        self.isDenoised = isDenoised
        self.isUpscaled = isUpscaled
        self.upscaleFactor = upscaleFactor
        self.linearHeadroomScale = linearHeadroomScale
        self.isDenoisedByNoiser = isDenoisedByNoiser
        self.processingHistory = processingHistory
    }

    public var isEmpty: Bool {
        isDenoised == nil && isUpscaled == nil && upscaleFactor == nil && linearHeadroomScale == nil && isDenoisedByNoiser == nil && processingHistory == nil
    }
}

/// User-configured native-resolution denoise job parameters.
public struct DenoiseJobConfig: Codable, Sendable, Equatable, Identifiable {
    public let id: UUID
    public let sourceImageURL: URL
    public var scaleFactor: ScaleFactor
    public var backbone: BackboneModelKind
    public var denoiseMode: DenoiseMode
    public var naturalGrainAmount: Float
    public var lumaStrength: Float
    public var chromaStrength: Float
    public var exportFormat: ExportFormat
    /// Validated session state from a handoff or a forwarding session. Merged into the decoded
    /// metadata before the denoise decision and persisted in the checkpoint.
    public var sessionState: PipelineSessionState?

    public init(
        id: UUID = UUID(),
        sourceImageURL: URL,
        scaleFactor: ScaleFactor = .x1,
        backbone: BackboneModelKind = .nafnetNormal,
        denoiseMode: DenoiseMode = .forceOn,
        naturalGrainAmount: Float = 0,
        lumaStrength: Float = 0.80,
        chromaStrength: Float = 1.00,
        exportFormat: ExportFormat = .autoMatchSource,
        sessionState: PipelineSessionState? = nil
    ) {
        self.id = id
        self.sourceImageURL = sourceImageURL
        self.scaleFactor = scaleFactor
        self.backbone = backbone
        self.denoiseMode = denoiseMode
        self.naturalGrainAmount = 0
        self.lumaStrength = lumaStrength.isFinite ? min(max(lumaStrength,0),1) : 0.80
        self.chromaStrength = chromaStrength.isFinite ? min(max(chromaStrength,0),1) : 1.00
        self.exportFormat = exportFormat
        self.sessionState = sessionState
    }
}

/// Pipeline execution lifecycle stage (`README.md`).
public enum JobStage: String, Codable, Sendable, Equatable {
    case idle
    case ingestingAndDecodingStrips
    case denoisingAndUpscalingTiles
    case pausedOnDisk
    case readyForInspection
    case exportingContainer
    case completed
    case failed
}

/// Atomic tile-level checkpoint state persisted to disk (`README.md`).
public struct JobCheckpointState: Codable, Sendable, Equatable {
    /// Version of the durable per-tile undo-journal transaction protocol (`TileUndoJournal`).
    public static let currentTransactionJournalVersion = 1
    /// Version 2 checkpoints guarantee `globalMaxLinearComponent` was derived from the decoded RGB
    /// component peak (not luminance). Older checkpoints are rescanned before resume.
    public static let currentNormalizationPeakVersion = 3

    public let jobID: UUID
    public let config: DenoiseJobConfig
    public let tileConfig: TileConfiguration
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let paddedSourceWidth: Int
    public let paddedSourceHeight: Int
    public let outputWidth: Int
    public let outputHeight: Int
    public let globalMaxLuminance: Float
    /// Per-channel linear component peak used for DNG/export headroom. Older checkpoints decode with
    /// this defaulted to `globalMaxLuminance`.
    public let globalMaxLinearComponent: Float
    public let alphaCurveConstant: Float
    public let didRunDenoise: Bool
    public let sourceMMapFileName: String
    public let outputMMapFileName: String
    public let metadataPacketFileName: String
    public var lastCompletedTileIndex: Int
    public let totalTiles: Int
    public var updatedAt: Date
    /// Additive recovery-protocol version. Older JSON without it decodes as version 1.
    public let transactionJournalVersion: Int
    /// Additive normalization-protocol version. Older JSON without it decodes as version 1 and is
    /// rescanned on resume because its component peak may only be luminance.
    public let normalizationPeakVersion: Int

    private enum CodingKeys: String, CodingKey {
        case jobID
        case config
        case tileConfig
        case sourceWidth
        case sourceHeight
        case paddedSourceWidth
        case paddedSourceHeight
        case outputWidth
        case outputHeight
        case globalMaxLuminance
        case globalMaxLinearComponent
        case alphaCurveConstant
        case didRunDenoise
        case sourceMMapFileName
        case outputMMapFileName
        case metadataPacketFileName
        case lastCompletedTileIndex
        case totalTiles
        case updatedAt
        case transactionJournalVersion
        case normalizationPeakVersion
    }

    public init(
        jobID: UUID,
        config: DenoiseJobConfig,
        tileConfig: TileConfiguration,
        sourceWidth: Int,
        sourceHeight: Int,
        paddedSourceWidth: Int,
        paddedSourceHeight: Int,
        outputWidth: Int,
        outputHeight: Int,
        globalMaxLuminance: Float,
        globalMaxLinearComponent: Float? = nil,
        alphaCurveConstant: Float = 16.0,
        didRunDenoise: Bool,
        sourceMMapFileName: String,
        outputMMapFileName: String,
        metadataPacketFileName: String,
        lastCompletedTileIndex: Int = -1,
        totalTiles: Int,
        updatedAt: Date = Date(),
        transactionJournalVersion: Int = JobCheckpointState.currentTransactionJournalVersion,
        normalizationPeakVersion: Int = 1
    ) {
        self.jobID = jobID
        self.config = config
        self.tileConfig = tileConfig
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.paddedSourceWidth = paddedSourceWidth
        self.paddedSourceHeight = paddedSourceHeight
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight
        self.globalMaxLuminance = globalMaxLuminance
        self.globalMaxLinearComponent = globalMaxLinearComponent ?? globalMaxLuminance
        self.alphaCurveConstant = alphaCurveConstant
        self.didRunDenoise = didRunDenoise
        self.sourceMMapFileName = sourceMMapFileName
        self.outputMMapFileName = outputMMapFileName
        self.metadataPacketFileName = metadataPacketFileName
        self.lastCompletedTileIndex = lastCompletedTileIndex
        self.totalTiles = totalTiles
        self.updatedAt = updatedAt
        self.transactionJournalVersion = transactionJournalVersion
        self.normalizationPeakVersion = normalizationPeakVersion
    }

    /// Custom decode so checkpoints written before the durable-journal fields existed still load:
    /// `globalMaxLinearComponent` falls back to `globalMaxLuminance`, and the journal version to 1.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        jobID = try container.decode(UUID.self, forKey: .jobID)
        config = try container.decode(DenoiseJobConfig.self, forKey: .config)
        tileConfig = try container.decode(TileConfiguration.self, forKey: .tileConfig)
        sourceWidth = try container.decode(Int.self, forKey: .sourceWidth)
        sourceHeight = try container.decode(Int.self, forKey: .sourceHeight)
        paddedSourceWidth = try container.decode(Int.self, forKey: .paddedSourceWidth)
        paddedSourceHeight = try container.decode(Int.self, forKey: .paddedSourceHeight)
        outputWidth = try container.decode(Int.self, forKey: .outputWidth)
        outputHeight = try container.decode(Int.self, forKey: .outputHeight)
        globalMaxLuminance = try container.decode(Float.self, forKey: .globalMaxLuminance)
        globalMaxLinearComponent = try container.decodeIfPresent(Float.self, forKey: .globalMaxLinearComponent)
            ?? globalMaxLuminance
        alphaCurveConstant = try container.decode(Float.self, forKey: .alphaCurveConstant)
        didRunDenoise = try container.decode(Bool.self, forKey: .didRunDenoise)
        sourceMMapFileName = try container.decode(String.self, forKey: .sourceMMapFileName)
        outputMMapFileName = try container.decode(String.self, forKey: .outputMMapFileName)
        metadataPacketFileName = try container.decode(String.self, forKey: .metadataPacketFileName)
        lastCompletedTileIndex = try container.decode(Int.self, forKey: .lastCompletedTileIndex)
        totalTiles = try container.decode(Int.self, forKey: .totalTiles)
        updatedAt = try container.decode(Date.self, forKey: .updatedAt)
        transactionJournalVersion = try container.decodeIfPresent(Int.self, forKey: .transactionJournalVersion)
            ?? JobCheckpointState.currentTransactionJournalVersion
        normalizationPeakVersion = try container.decodeIfPresent(Int.self, forKey: .normalizationPeakVersion)
            ?? 1
    }
}
