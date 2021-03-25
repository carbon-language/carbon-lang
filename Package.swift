// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "barcon",
    products: [
        // Products define the executables and libraries a package produces, and
        // make them visible to other packages.
        .executable(
            name: "barcon",
            targets: ["barcon"]),
        .library(
            name: "Syntax",
            targets: ["Syntax"]),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can
        // define a module or a test suite.  Targets can depend on other targets
        // in this package, and on products in packages this package depends on.
        .target(
            name: "barcon",
            dependencies: ["Syntax"]),
        .target(
            name: "Syntax",
            dependencies: [],
            exclude: ["Parser.citron"]),
        .testTarget(
            name: "barconTests",
            dependencies: ["barcon"]),
    ]
)
