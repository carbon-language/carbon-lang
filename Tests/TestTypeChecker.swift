// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

final class TypeCheckNominalTypeDeclaration: XCTestCase {

  func testStruct() {
    "struct X { var Int: y; }".checkTypeChecks()
  }

  func testStructStructMember() {
    """
    struct X { var Int: y; }
    struct Z { var X: a; }
    """.checkTypeChecks()
  }

  func testStructNonTypeExpression0()  {
    "struct X { var 42: y; }".checkFailsToTypeCheck(
      withMessage: "Not a type expression (value has type Int)")
  }

  func testChoice() {
    """
    choice X {
      Box,
      Car(Int),
      Children(Int, Bool)
    }
    """.checkTypeChecks()
  }

  func testChoiceChoiceMember() {
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

  func testChoiceNonTypeExpression() {
    "choice X { Bog(42) }".checkFailsToTypeCheck(
      withMessage: "Not a type expression (value has type (Int))")
  }
}

/// Tests that go along with having implemented checking of nominal type bodies
/// and function signatures.
final class TypeCheckFunctionSignatures: XCTestCase {
  //
  // Simplest test cases.
  //

  func testTrivial() {
    "fn f() {}".checkTypeChecks()
  }

  func testOneParameter() {
    "fn f(Int: x) {}".checkTypeChecks()
  }

  func testOneResult() {
    "fn f() -> Int { return 3; }".checkTypeChecks()
  }

  func testDoubleArrow() {
    "fn f() => 3;".checkTypeChecks()
  }

  func testDoubleArrowIdentity() {
    "fn f(Int: x) => x;".checkTypeChecks()
  }

  func testDuplicateLabel() {
    "fn f(.x = Int: x, .x = Int: y) => x;".checkFailsToTypeCheck(
      withMessage: "Duplicate label x")
  }

  func testEvaluateTupleLiteral() {
    "fn f((Int, Int): x) => (x, x);".checkTypeChecks()
  }

  func testEvaluateFunctionType() {
    """
    fn g(Int: a, Int: b)->Int { return a; }
    fn f(fnty (Int, Int)->Int: x) => x;
    fn h() => f(g)(3, 4);
    """.checkTypeChecks()
  }

  func testFunctionCallArityMismatch() {
    """
    fn g(Int: a, Int: b) => a;
    fn f(Bool: x) => g(x);
    """.checkFailsToTypeCheck(
      withMessage:
        "argument types (Bool) do not match parameter types (Int, Int)")
  }

  func testFunctionCallParameterTypeMismatch() {
    """
    fn g(Int: a, Int: b) => a;
    fn f(Bool: x) => g(1, x);
    """.checkFailsToTypeCheck(
      withMessage:
        "argument types (Int, Bool) do not match parameter types (Int, Int)")
  }

  func testFunctionCallLabelMismatch() {
    """
    fn g(.first = Int: a, Int: b) => a;
    fn f(Bool: x) => g(.last = 1, 2);
    """.checkFailsToTypeCheck(
      withMessage:
        "argument types (.last = Int, Int) "
        + "do not match parameter types (.first = Int, Int)")
  }

  func testFunctionCallLabel() {
    """
    fn g(.first = Int: a, .second = Int: b) => a;
    fn f(Bool: x) => g(.first = 1, .second = 2);
    """.checkTypeChecks()
  }

  func testAlternativePayloadMismatches() {
    """
    choice X { One }
    fn f() => X.One(1);
    """.checkFailsToTypeCheck(withMessage:"do not match payload type")

    """
    choice X { One(Int) }
    fn f() => X.One();
    """.checkFailsToTypeCheck(withMessage:"do not match payload type")

    """
    choice X { One(.x = Int) }
    fn f() => X.One(1);
    """.checkFailsToTypeCheck(withMessage:"do not match payload type")

    """
    choice X { One(Int) }
    fn f() => X.One(.x = 1);
    """.checkFailsToTypeCheck(withMessage:"do not match payload type")
  }

  func testSimpleTypeTypeExpressions() {
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

  func testDeclaredTypeStruct() {
    """
    struct X {}
    fn f() -> X { return X(); }
    fn g(X) {}
    """.checkTypeChecks()
  }

  func testDeclaredTypeChoice() {
    """
    choice X { Bonk }
    fn f() -> X { return X.Bonk; }
    fn g(X) {}
    """.checkTypeChecks()
  }

  func testDeclaredTypeAlternative() {
    """
    choice X { Bonk(Int) }
    fn f() => X.Bonk(3);
    """.checkTypeChecks()
  }

  func testDeclaredTypeFunctionDefinition() {
    """
    fn g() => f();
    fn f() => 1;
    """.checkTypeChecks()
  }

  func testNonStructTypeValueIsNotCallable() {
    """
    choice X { One(Int) }
    fn f() => X();
    """.checkFailsToTypeCheck(withMessage:"type X is not callable.")

    "fn f() => Int();".checkFailsToTypeCheck(
      withMessage: "type Int is not callable.")

    "fn f() => Bool();".checkFailsToTypeCheck(
      withMessage: "type Bool is not callable.")

    "fn f() => Type();".checkFailsToTypeCheck(
      withMessage: "type Type is not callable.")

    "fn f() => (fnty ()->Int)();".checkFailsToTypeCheck(
      withMessage: "type fnty () -> Int is not callable.")
  }

  func testTypeOfStructConstruction() {
    """
    struct X {}
    fn f(X: _) => 1;
    fn g() => f(X());
    """.checkTypeChecks()
  }

  func testStructConstructionArgumentMismatch() {
    """
    struct X {}
    fn f() => X(1);
    """.checkFailsToTypeCheck(withMessage:
        "argument types (Int) do not match required initializer parameters ()")
  }

  func testNonCallableNonTypeValues() {
    "fn f() => false();".checkFailsToTypeCheck(
      withMessage:"value of type Bool is not callable.")

    "fn f() => 1();".checkFailsToTypeCheck(
      withMessage:"value of type Int is not callable.")

    """
    struct X {}
    fn f() => X()();
    """.checkFailsToTypeCheck(withMessage:"value of type X is not callable.")
  }

  func testStructMemberAccess() {
    """
    struct X { var Int: a; var Bool: b; }
    fn f(X: y) => (y.a, y.b);
    """.checkTypeChecks()
  }

  func testTupleNamedAccess() {
    """
    fn f() => (.x = 0, .y = false).x;
    fn g() => (.x = 0, .y = false).y;
    """.checkTypeChecks()
  }

  func testInvalidMemberAccesses() {
    "fn f() => (.x = 0, .y = false).c;".checkFailsToTypeCheck(
      withMessage: "tuple type (.x = Int, .y = Bool) has no field 'c'")

    """
    struct X { var Int: a; var Bool: b; }
    fn f(X: y) => (y.a, y.c);
    """.checkFailsToTypeCheck(withMessage:"struct X has no member 'c'")

    """
    choice X {}
    fn f() => X.One();
    """.checkFailsToTypeCheck(withMessage:"choice X has no alternative 'One'")

    "fn f() => Int.One;".checkFailsToTypeCheck(
      withMessage: "expression of type Type does not have named members")

    "fn f() => 1.One;".checkFailsToTypeCheck(
      withMessage: "expression of type Int does not have named members")
  }

  func testTuplePatternType() {
    """
    fn f((1, Int: x), Bool: y) => x;
    fn g() => f((1, 2), true);
    """.checkTypeChecks()
  }

  func testFunctionCallPatternType() {
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

    "fn f(Int(Bool: _));".checkFailsToTypeCheck(
      withMessage: "Called type must be a struct, not 'Int'")

    """
    struct X { var Int: a; var Bool: b; }
    fn f(X(.a = Bool: a, .b = Bool: b)) => b;
    """.checkFailsToTypeCheck(withMessage:
      "Argument tuple type (.a = Bool, .b = Bool) doesn't match"
        + " struct initializer type (.a = Int, .b = Bool)")

    """
    choice X { One(Int, Bool), Two }
    fn f(X.One(Bool: a, Bool: b), X.Two()) => b;
    """.checkFailsToTypeCheck(withMessage:
      "Argument tuple type (Bool, Bool) doesn't match"
        + " alternative payload type (Int, Bool)")

    """
    fn f(1(Bool: _));
    """.checkFailsToTypeCheck(withMessage:
      "instance of type Int is not callable")
  }

  func testFunctionTypePatternType() {
    "fn f(fnty(Type: x)) => 0;".checkTypeChecks()

    "fn f(fnty(Type: x)->Bool) => 0;".checkTypeChecks()

    "fn f(fnty(Type: x)->Type: y) => 0;".checkTypeChecks()

    "fn f(fnty(Int)->Type: y) => 0;".checkTypeChecks()

    "fn f(fnty(4)->Type: y) => 0;".checkFailsToTypeCheck(
      withMessage: "Not a type expression (value has type (Int))")

    "fn f(fnty(Int: x)) => 0;".checkFailsToTypeCheck(
      withMessage:
        "Pattern in this context must match type values, not Int values")

    "fn f(fnty(auto: x)) => 0;".checkFailsToTypeCheck(
      withMessage: "No initializer available to deduce type for auto")

    // A tuple of types is a valid type.
    "fn f(fnty((Type, Type): x)->Type: y) => 0;".checkTypeChecks()

    "fn f(fnty((Int, Int): x)->Type: y) => 0;".checkFailsToTypeCheck(
      withMessage:
        "Pattern in this context must match type values, not (Int, Int) values")

    """
    fn g(Int: x) => Int;
    fn f(fnty((Int, Int): x)->g(3)) => 0;
    """.checkFailsToTypeCheck(
      withMessage:
        "Pattern in this context must match type values, not (Int, Int) values")
  }

  func testIndexExpression() {
    "fn f((Int,): r) => r[0];".checkTypeChecks()

    """
    fn f((Int, Bool): r) => r[0];
    fn g(Int: _) => 1;
    fn h() => g(f((1, false)));
    """.checkTypeChecks()

    """
    fn f((Int, Bool): r) => r[1];
    fn g(Bool: _) => 1;
    fn h() => g(f((1, false)));
    """.checkTypeChecks()

    "fn f(Int: x) => x[0];".checkFailsToTypeCheck(
      withMessage:"Can't index non-tuple type Int")

    "fn f((Int,): x) => x[Int];".checkFailsToTypeCheck(
      withMessage: "Index type must be Int, not Type")

    "fn f((.x = Int, Int, Bool): r) => r[3];".checkFailsToTypeCheck(
      withMessage:
        "Tuple type (.x = Int, Int, Bool) has no value at position 3")
  }

  func testTypeOfUnaryOperator() {
    """
    fn f() => -3;
    fn g(Int: _) => 0;
    fn h() => g(f());
    """.checkTypeChecks("unary minus")

    """
    fn f() => not false;
    fn g(Bool: _) => 0;
    fn h() => g(f());
    """.checkTypeChecks("logical not")

    "fn f() => -false;".checkFailsToTypeCheck(withMessage:
      "Expected expression of type Int, not Bool")

    "fn f() => not 3;".checkFailsToTypeCheck(withMessage:
        "Expected expression of type Bool, not Int")
  }

  func testTypeOfBinaryOperator() {
    """
    fn f(Int: a, Int: b) => a == b;
    fn g(Bool: _) => 0;
    fn h() => g(f(1, 2));
    """.checkTypeChecks()

    """
    fn f(Int: a, Int: b) => a + b;
    fn g(Int: _) => 0;
    fn h() => g(f(1, 2));
    """.checkTypeChecks()

    """
    fn f(Int: a, Int: b) => a - b;
    fn g(Int: _) => 0;
    fn h() => g(f(1, 2));
    """.checkTypeChecks()

    """
    fn f(Bool: a, Bool: b) => a and b;
    fn g(Bool: _) => 0;
    fn h() => g(f(true, false));
    """.checkTypeChecks()

    """
    fn f(Bool: a, Bool: b) => a or b;
    fn g(Bool: _) => 0;
    fn h() => g(f(true, false));
    """.checkTypeChecks()
  }

  func testTypeDependencyLoop() {
    """
    fn f() => g();
    fn g() => f();
    """.checkFailsToTypeCheck(withMessage: "type dependency loop")
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
             if let errors = String(contentsOfFile: p).parsedAsCarbon(fromFile: p)) {

          let executable = checkNoThrow(try ExecutableProgram(ast))
          _ = executable
        }
      }
    }
  }
  */
}
