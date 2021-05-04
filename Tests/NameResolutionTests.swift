// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter
import Foundation

final class NameResolutionTests: XCTestCase {
  func testInit() {
    // Make sure we can even create one.
    _ = CarbonParser()
  }

  func testExamples() throws {
    let testdata = 
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("testdata")

    for f in try! FileManager().contentsOfDirectory(atPath: testdata.path) {
      let p = testdata.appendingPathComponent(f).path

      // Skip experimental syntax for now.
      if f.hasPrefix("experimental_") { continue }

      if !f.hasSuffix("_fail.6c") {
        let ast = try checkNoThrow(
          try String(contentsOfFile: p).parsedAsCarbon(fromFile: p))

        let executable = try checkNoThrow(try ExecutableProgram(ast))
        _ = executable
      }
    }
  }
}
