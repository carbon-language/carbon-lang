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

/// Tests that go along with having implemented checking of nominal type bodies
/// and function signatures.
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

  func testDuplicateLabel() throws {
    try "fn f(.x = Int: x, .x = Int: y) => x;".typeChecked().errors
      .checkForMessageExcerpt("Duplicate label")
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
      """.typeChecked().errors
      .checkForMessageExcerpt("do not match parameter types")
  }

  func testFunctionCallParameterTypeMismatch() throws {
    try """
      fn g(Int: a, Int: b) => a;
      fn f(Bool: x) => g(1, x);
      """.typeChecked().errors
      .checkForMessageExcerpt("do not match parameter types")
  }

  func testFunctionCallLabelMismatch() throws {
    try """
      fn g(.first = Int: a, Int: b) => a;
      fn f(Bool: x) => g(.last = 1, 2);
      """.typeChecked().errors
      .checkForMessageExcerpt("do not match parameter types")
  }

  func testFunctionCallLabel() throws {
    """
  fn g(.first = Int: a, .second = Int: b) => a;
  fn f(Bool: x) => g(.first = 1, .second = 2);
  """.checkTypeChecks()
  }

  func testAlternativePayloadMismatches() throws {
    try """
      choice X { One }
      fn f() => X.One(1);
      """.typeChecked().errors
      .checkForMessageExcerpt("do not match payload type")

    try """
      choice X { One(Int) }
      fn f() => X.One();
      """.typeChecked().errors
      .checkForMessageExcerpt("do not match payload type")

    try """
      choice X { One(.x = Int) }
      fn f() => X.One(1);
      """.typeChecked().errors
      .checkForMessageExcerpt("do not match payload type")

    try """
      choice X { One(Int) }
      fn f() => X.One(.x = 1);
      """.typeChecked().errors
      .checkForMessageExcerpt("do not match payload type")
  }

  func testSimpleTypeTypeExpressions() throws {
    """
    fn f() => Int;
    fn g(Type: _) => 1;
    fn h() => g(f());
    """.checkTypeChecks()

    """
    fn f() => Bool;
    fn g(Type: _) => 1;
    fn h() => g(f());
    """.checkTypeChecks()

    """
    fn f() => Type;
    fn g(Type: _) => 1;
    fn h() => g(f());
    """.checkTypeChecks()

    """
    fn f() => fnty (Int)->Int;
    fn g(Type: _) => 1;
    fn h() => g(f());
    """.checkTypeChecks()

    """
    struct X {}
    fn f() => X;
    fn g(Type: _) => 1;
    fn h() => g(f());
    """.checkTypeChecks()

    """
    choice X { Bob }
    fn f() => X;
    fn g(Type: _) => 1;
    fn h() => g(f());
    """.checkTypeChecks()
  }

  func testBooleanLiteral() {
    // This test was worth one line of coverage at some point.
    """
    fn f() => false;
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

  func testNonStructTypeValueIsNotCallable() throws {
    try """
    choice X { One(Int) }
    fn f() => X();
    """.typeChecked()
    .errors.checkForMessageExcerpt("type X is not callable.")

    try """
    fn f() => Int();
    """.typeChecked()
    .errors.checkForMessageExcerpt("type Int is not callable.")

    try """
    fn f() => Bool();
    """.typeChecked()
    .errors.checkForMessageExcerpt("type Bool is not callable.")

    try """
    fn f() => Type();
    """.typeChecked()
    .errors.checkForMessageExcerpt("type Type is not callable.")

    try """
    fn f() => (fnty ()->Int)();
    """.typeChecked()
    .errors.checkForMessageExcerpt("is not callable.")
  }

  func testTypeOfStructConstruction() {
    """
    struct X {}
    fn f(X: _) => 1;
    fn g() => f(X());
    """.checkTypeChecks()
  }

  func testStructConstructionArgumentMismatch() throws {
    try """
    struct X {}
    fn f() => X(1);
    """.typeChecked().errors
    .checkForMessageExcerpt("do not match required initializer parameters")
  }

  func testNonCallableNonTypeValues() throws {
    try """
    fn f() => false();
    """.typeChecked()
    .errors.checkForMessageExcerpt("value of type Bool is not callable.")

    try """
    fn f() => 1();
    """.typeChecked()
    .errors.checkForMessageExcerpt("value of type Int is not callable.")

    try """
    struct X {}
    fn f() => X()();
    """.typeChecked()
    .errors.checkForMessageExcerpt("value of type X is not callable.")
  }

  func testStructMemberAccess() throws {
    """
    struct X { var Int: a; var Bool: b; }
    fn f(X: y) => (y.a, y.b);
    """.checkTypeChecks()
  }

  func testTupleNamedAccess() throws {
    """
    fn f() => (.x = 0, .y = false).x;
    fn g() => (.x = 0, .y = false).y;
    """.checkTypeChecks()
  }

  func testInvalidMemberAccesses() throws {
    try """
    fn f() => (.x = 0, .y = false).c;
    """.typeChecked()
    .errors.checkForMessageExcerpt("has no field 'c'")

    try """
    struct X { var Int: a; var Bool: b; }
    fn f(X: y) => (y.a, y.c);
    """.typeChecked()
    .errors.checkForMessageExcerpt("has no member 'c'")

    try """
    choice X {}
    fn f() => X.One();
    """.typeChecked()
   .errors.checkForMessageExcerpt("choice X has no alternative 'One'")

    try """
    fn f() => Int.One;
    """.typeChecked().errors.checkForMessageExcerpt(
      "expression of type Type does not have named members")

    try """
    fn f() => 1.One;
    """.typeChecked().errors.checkForMessageExcerpt(
      "expression of type Int does not have named members")
  }

  func testTuplePatternType() {
    """
    fn f((1, Int: x), Bool: y) => x;
    fn g() => f((1, 2), true);
    """.checkTypeChecks()
  }

  func testFunctionCallPatternType() throws {
    """
    choice X { One(Int, Bool), Two }
    fn f(X.One(Int: a, Bool: b), X.Two()) => b;
    fn g(Bool: _) => 1;
    fn h() => g(f(X.One(3, true), X.Two()));
    """.checkTypeChecks()

    """
    struct X { var Int: a; var Bool: b; }
    fn f(X(.a = Int: a, .b = Bool: b)) => b;
    fn g(Bool: _) => 1;
    fn h() => g(f(X(.a = 3, .b = false)));
    """.checkTypeChecks()

    try """
    fn f(Int(Bool: _));
    """.typeChecked().errors.checkForMessageExcerpt(
      "Called type must be a struct, not 'Int'")

    try """
    struct X { var Int: a; var Bool: b; }
    fn f(X(.a = Bool: a, .b = Bool: b)) => b;
    """.typeChecked().errors.checkForMessageExcerpt(
      "doesn't match struct initializer type")

    try """
    choice X { One(Int, Bool), Two }
    fn f(X.One(Bool: a, Bool: b), X.Two()) => b;
    """.typeChecked().errors.checkForMessageExcerpt(
      "doesn't match alternative payload type")

    try """
    fn f(1(Bool: _));
    """.typeChecked().errors.checkForMessageExcerpt(
      "instance of type Int is not callable")
  }

  /*
   TBD
  func testFunctionTypePatternType() {
    """
    fn f(fnty(true)->1) => 0;
    """.checkTypeChecks()
  }

   */
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
