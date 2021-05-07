// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

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

final class TypeCheckFunctionSignatures: XCTestCase {
  //
  // Simplest test cases.
  //

  func testTrivial() throws {
    try XCTAssertEqual("fn f() {}".typeChecked().errors, [])
  }

  func testOneParameter() throws {
    try XCTAssertEqual("fn f(Int: x) {}".typeChecked().errors, [])
  }

  func testOneResult() throws {
    try XCTAssertEqual("fn f() -> Int { return 3; }".typeChecked().errors, [])
  }

  func testDoubleArrow() throws {
    try XCTAssertEqual("fn f() => 3;".typeChecked().errors, [])
  }

  func testDoubleArrowIdentity() throws {
    try XCTAssertEqual("fn f(Int: x) => x;".typeChecked().errors, [])
  }

  func testEvaluateTupleLiteral() throws {
    try XCTAssertEqual("fn f((Int, Int): x) => x;".typeChecked().errors, [])
  }

  //
  // Exercising code paths that return the type of a declared entity.
  //

  func testDeclaredTypeStruct() throws {
    try XCTAssertEqual(
      """
      struct X {}
      fn f() -> X { return X(); }
      fn g(X) {}
      """.typeChecked().errors, [])
  }

  func testDeclaredTypeChoice() throws {
    try XCTAssertEqual(
      """
      choice X { Bonk }
      fn f() -> X { return X.Bonk; }
      fn g(X) {}
      """.typeChecked().errors, [])
  }

  func testDeclaredTypeAlternative() throws {
    try XCTAssertEqual(
      """
      choice X { Bonk(Int) }
      fn f() => X.Bonk(3);
      """.typeChecked().errors, [])
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
