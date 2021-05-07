// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

final class TypeCheckNominalTypeDeclaration: XCTestCase {

  func testStruct() throws {
    "struct X { var Int: y; }".checkTypeChecks()
  }

  func testStructStructMember() throws {
    """
    struct X { var Int: y; }
    struct Z { var X: a; }
    """.checkTypeChecks()
  }

  func testStructNonTypeExpression0() throws {
    try "struct X { var 42: y; }"
      .typeChecked().errors.checkForMessageExcerpt("Not a type expression")
  }

  func testChoice() throws {
    """
    choice X {
      Box,
      Car(Int),
      Children(Int, Bool)
    }
    """.checkTypeChecks()
  }

  func testChoiceChoiceMember() throws {
    """
    choice Y {
      Fork, Knife(X), Spoon(X, X)
    }
    choice X {
      Box,
      Car(Int),
      Children(Int, Bool)
    }
    """.checkTypeChecks()
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
    "fn f() {}".checkTypeChecks()
  }

  func testOneParameter() throws {
    "fn f(Int: x) {}".checkTypeChecks()
  }

  func testOneResult() throws {
    "fn f() -> Int { return 3; }".checkTypeChecks()
  }

  func testDoubleArrow() throws {
    "fn f() => 3;".checkTypeChecks()
  }

  func testDoubleArrowIdentity() throws {
    "fn f(Int: x) => x;".checkTypeChecks()
  }

  func testEvaluateTupleLiteral() throws {
    "fn f((Int, Int): x) => (x, x);".checkTypeChecks()
  }

  func testEvaluateFunctionType() throws {
    """
    fn g(Int: a, Int: b)->Int { return a; }
    fn f(fnty (Int, Int)->Int: x) => x;
    fn h() => f(g)(3, 4);
    """.checkTypeChecks()
  }

  func testFunctionCallArityMismatch() throws {
    try """
      fn g(Int: a, Int: b) => a;
      fn f(Bool: x) => g(x);
      """.typeChecked().errors.checkForMessageExcerpt(
      "do not match parameter types")
  }

  func testFunctionCallParameterTypeMismatch() throws {
    try """
      fn g(Int: a, Int: b) => a;
      fn f(Bool: x) => g(1, x);
      """.typeChecked().errors.checkForMessageExcerpt(
      "do not match parameter types")
  }

  func testFunctionCallLabelMismatch() throws {
    try """
      fn g(.first = Int: a, Int: b) => a;
      fn f(Bool: x) => g(.last = 1, 2);
      """.typeChecked().errors.checkForMessageExcerpt(
      "do not match parameter types")
  }

  func testFunctionCallLabel() throws {
    """
  fn g(.first = Int: a, .second = Int: b) => a;
  fn f(Bool: x) => g(.first = 1, .second = 2);
  """.checkTypeChecks()
}


  //
  // Exercising code paths that return the type of a declared entity.
  //

  func testDeclaredTypeStruct() throws {
    """
    struct X {}
    fn f() -> X { return X(); }
    fn g(X) {}
    """.checkTypeChecks()
  }

  func testDeclaredTypeChoice() throws {
    """
    choice X { Bonk }
    fn f() -> X { return X.Bonk; }
    fn g(X) {}
    """.checkTypeChecks()
  }

  func testDeclaredTypeAlternative() throws {
    """
    choice X { Bonk(Int) }
    fn f() => X.Bonk(3);
    """.checkTypeChecks()
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
