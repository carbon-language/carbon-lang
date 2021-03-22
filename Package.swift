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
            name: "AST",
            targets: ["AST"]),
        .library(
            name: "Syntax",
            targets: ["Syntax"]),
    ],
    dependencies: [
      .package(
        url: "https://github.com/yassram/SwiParse.git",
        from: "1.0.0"),
      .package(
        url: "https://github.com/yassram/SwiLex.git",
        from: "1.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can
        // define a module or a test suite.  Targets can depend on other targets
        // in this package, and on products in packages this package depends on.
        .target(
            name: "barcon",
            dependencies: ["AST"]),
        .target(
            name: "AST",
            dependencies: []),
        .target(
            name: "Syntax",
            dependencies: ["SwiLex", "SwiParse"]),
        .testTarget(
            name: "barconTests",
            dependencies: ["barcon"]),
    ]
)
