// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

extension String {
  /// Returns the results of parsing, name lookup, and typechecking `self`.
  func typeChecked() throws
    -> (ExecutableProgram, typeChecker: TypeChecker, errors: ErrorLog)
  {
    let executable = try self.checkExecutable()
    let typeChecker = TypeChecker(executable)
    return (executable, typeChecker, typeChecker.errors)
  }
}

final class TypeCheckNominalTypeDeclaration: XCTestCase {

  func testStruct() throws {
    try XCTAssertEqual(
      "struct X { var Int: y; }"
        .typeChecked().errors, [])
  }

  func testStructStructMember() throws {
    try XCTAssertEqual(
      """
      struct X { var Int: y; }
      struct Z { var X: a; }
      """
        .typeChecked().errors, [])
  }

  func testStructNonTypeExpression0() throws {
    try "struct X { var 42: y; }"
      .typeChecked().errors.checkForMessageExcerpt("Not a type expression")
  }

  func testChoice() throws {
    try XCTAssertEqual(
      """
      choice X {
        Box,
        Car(Int),
        Children(Int, Bool)
      }
      """.typeChecked().errors, [])
  }

  func testChoiceChoiceMember() throws {
    try XCTAssertEqual(
      """
      choice Y {
        Fork, Knife(X), Spoon(X, X)
      }
      choice X {
        Box,
        Car(Int),
        Children(Int, Bool)
      }
      """.typeChecked().errors, [])
  }

  func testChoiceNonTypeExpression() throws {
    try "choice X { Bog(42) }"
      .typeChecked().errors.checkForMessageExcerpt("Not a type expression")
  }
}

final class TypeCheckExamples: XCTestCase {
  /*
  func testExamples() {
    let testdata =
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("testdata")

    for f in try! FileManager().contentsOfDirectory(atPath: testdata.path) {
      let p = testdata.appendingPathComponent(f).path

      // Skip experimental syntax for now.
      if f.hasPrefix("experimental_") { continue }

      if !f.hasSuffix("_fail.6c") {
        if let ast = checkNoThrow(
             try String(contentsOfFile: p).parsedAsCarbon(fromFile: p)) {

          let executable = checkNoThrow(try ExecutableProgram(ast))
          _ = executable
        }
      }
    }
  }
  */
}
