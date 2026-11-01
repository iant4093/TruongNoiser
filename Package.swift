// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "TruongNoiserPackage",
    platforms: [
        .iOS("26.0"),
        .macOS("26.0")
    ],
    products: [
        .library(name: "TruongNoiserCore", targets: ["TruongNoiserCore"]),
        .library(name: "DeepZoomGalleryUI", targets: ["DeepZoomGalleryUI"]),
        .library(name: "NoiserActivityUI", targets: ["NoiserActivityUI"]),
        // Exposed explicitly so the app target can register/route background tasks without
        // relying on the transitive DeepZoomGalleryUI dependency.
        .library(name: "BackgroundJobManager", targets: ["BackgroundJobManager"])
    ],
    targets: [
        .target(
            name: "TruongNoiserCore",
            dependencies: []
        ),
        .target(name: "NoiserActivityUI", dependencies: ["TruongNoiserCore"]),
        .target(
            name: "MMapCanvas",
            dependencies: ["TruongNoiserCore"]
        ),
        .target(
            name: "CoreIngestion",
            dependencies: ["TruongNoiserCore", "MMapCanvas", "DNGWriterBridge"],
            swiftSettings: [.unsafeFlags(["-O"], .when(configuration: .debug))]
        ),
        .target(
            name: "MetalColorEngine",
            dependencies: ["TruongNoiserCore", "MMapCanvas"],
            // Image arithmetic must remain representative on a phone launched with Xcode Run.
            // Keep normal -O checks; never use -Ounchecked or fast-math. UI stays unoptimized.
            swiftSettings: [.unsafeFlags(["-O"], .when(configuration: .debug))]
        ),
        .target(
            name: "TiledInferenceEngine",
            dependencies: ["TruongNoiserCore", "MMapCanvas", "CoreIngestion", "MetalColorEngine"],
            swiftSettings: [.unsafeFlags(["-O"], .when(configuration: .debug))]
        ),
        .target(
            name: "DNGWriterBridge",
            dependencies: [],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("include"),
                .unsafeFlags(["-std=c++17"])
            ]
        ),
        .target(
            name: "CoreExport",
            dependencies: ["TruongNoiserCore", "MMapCanvas", "MetalColorEngine", "TiledInferenceEngine", "CoreIngestion", "DNGWriterBridge"],
            swiftSettings: [.unsafeFlags(["-O"], .when(configuration: .debug))]
        ),
        .target(
            name: "BackgroundJobManager",
            dependencies: ["TruongNoiserCore", "TiledInferenceEngine"]
        ),
        .target(
            name: "DeepZoomGalleryUI",
            dependencies: [
                "TruongNoiserCore",
                "MMapCanvas",
                "CoreIngestion",
                "MetalColorEngine",
                "TiledInferenceEngine",
                "CoreExport",
                "BackgroundJobManager"
            ]
        ),
        .testTarget(name: "TruongNoiserCoreTests", dependencies: ["TruongNoiserCore"]),
        .testTarget(name: "MMapCanvasTests", dependencies: ["MMapCanvas", "TruongNoiserCore"]),
        .testTarget(name: "CoreIngestionTests", dependencies: ["CoreIngestion", "MMapCanvas", "TruongNoiserCore"], resources: [.copy("Fixtures")]),
        .testTarget(name: "MetalColorEngineTests", dependencies: ["MetalColorEngine", "MMapCanvas", "TruongNoiserCore"]),
        .testTarget(name: "TiledInferenceEngineTests", dependencies: ["TiledInferenceEngine", "MetalColorEngine", "CoreIngestion", "CoreExport", "MMapCanvas", "TruongNoiserCore"], resources: [.copy("Fixtures")]),
        .testTarget(
            name: "DNGWriterBridgeTests",
            dependencies: ["DNGWriterBridge", "CoreIngestion", "TruongNoiserCore", "MMapCanvas"]
        ),
        .testTarget(name: "CoreExportTests", dependencies: ["CoreExport", "TiledInferenceEngine", "CoreIngestion", "MMapCanvas", "TruongNoiserCore"]),
        .testTarget(
            name: "BackgroundJobManagerTests",
            dependencies: ["BackgroundJobManager", "TiledInferenceEngine", "TruongNoiserCore"]
        ),
        .testTarget(
            name: "DeepZoomGalleryUITests",
            dependencies: [
                "DeepZoomGalleryUI",
                "TruongNoiserCore",
                "MMapCanvas",
                "CoreIngestion",
                "CoreExport",
                "TiledInferenceEngine",
                "BackgroundJobManager",
                "NoiserActivityUI",
            ]
        )
    ]
)
