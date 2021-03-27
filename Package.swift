// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "CarbonInterpreter",
    products: [
        .library(
            name: "CarbonInterpreter",
            targets: ["CarbonInterpreter"]),
    ],
    targets: [
        .target(
            name: "CarbonInterpreter",
            dependencies: [],
            path: "Sources",
            // Swift Package manager doesn't yet know how to process this file
            // so we handle it with Make.  See Makefile for details.
            exclude: ["Parser.citron"]
        ), 
        .testTarget(
            name: "Tests",
            dependencies: ["CarbonInterpreter"],
            path: "Tests"
        ),
    ]
)
