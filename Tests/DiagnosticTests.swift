// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

/// Tests that the format of error messages match the Gnu standard, which can be
/// interpreted by many tools.
final class GnuFormattingTests: XCTestCase {
  func testSourcePosition() {
    XCTAssertEqual("\(SourcePosition(line: 1, column: 5))", "1.5")
  }

  func testSourceRegion() {
    let sample = SourceRegion(
      fileName: "someFile",
      .init(line: 2, column: 7) ..< .init(line: 9, column: 2))
    XCTAssertEqual("\(sample)", "someFile:2.7-9.1")
  }

  func testCompileError() {
    let r0 = SourceRegion(
      fileName: "someFile",
      .init(line: 3, column: 2) ..< .init(line: 4, column: 9))

    let r1 = SourceRegion(
      fileName: "otherFile",
      .init(line: 5, column: 1) ..< .init(line: 9, column: 2))

    let sample = CompileError(
      "heck problem", at: ASTSite(devaluing: r0),
      notes: [(message: "this also", site: ASTSite(devaluing: r1))])

    XCTAssertEqual(
      "\(sample)",
      """
      someFile:3.2-4.8: error: heck problem
      otherFile:5.1-9.1: note(0): this also
      """)
  }
}
