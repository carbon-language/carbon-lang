// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

final class InterpreterTests: XCTestCase {
  func testMinimal0() {
    guard let exe = "fn main() -> Int { return 0; }".checkExecutable() else {
      return
    }

    var engine = Interpreter(exe)
    XCTAssertEqual(0, engine.run())
  }

  func testMinimal1() {
    guard let exe = "fn main() -> Int { return 42; }".checkExecutable() else {
      return
    }

    var engine = Interpreter(exe)
    XCTAssertEqual(42, engine.run())
  }

  func testExpressionStatement1() {
    guard let exe = "fn main() -> Int { 777; return 42; }".checkExecutable()
    else { return }
    var engine = Interpreter(exe)
    XCTAssertEqual(42, engine.run())
  }

  func testExpressionStatement2() {
    guard let exe = "fn main() -> Int { var Int x = 777; x + 1; return 42; }".checkExecutable()
    else { return }
    var engine = Interpreter(exe)
    XCTAssertEqual(42, engine.run())
  }

  func run(_ testFile: String, tracing: Bool = false) -> Int? {
    let testdata =
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("testdata")

    let sourcePath = testdata.appendingPathComponent(testFile).path
    let source = try! String(contentsOfFile: sourcePath)

    guard let program = source.checkExecutable(fromFile: sourcePath)
    else { return nil }
    var engine = Interpreter(program)
    engine.tracing = tracing
    return engine.run()
  }

  func test1() {
    // XCTAssertEqual(run("fun1.6c", tracing: true), 0)
  }

  func testExamples() {
    XCTAssertEqual(run("assignment_copy1.6c"), 0)
    XCTAssertEqual(run("assignment_copy2.6c"), 0)
    XCTAssertEqual(run("block1.6c"), 0)
    XCTAssertEqual(run("block2.6c"), 0)
    XCTAssertEqual(run("break1.6c"), 0)
    XCTAssertEqual(run("choice1.6c"), 0)
    XCTAssertEqual(run("continue1.6c"), 0)
    // XCTAssertEqual(run("experimental_continuation1.6c"), 0)
    // XCTAssertEqual(run("experimental_continuation2.6c"), 0)
    // XCTAssertEqual(run("experimental_continuation3.6c"), 0)
    // XCTAssertEqual(run("experimental_continuation4.6c"), 0)
    // XCTAssertEqual(run("experimental_continuation5.6c"), 0)
    // XCTAssertEqual(run("experimental_continuation6.6c"), 0)
    // XCTAssertEqual(run("experimental_continuation7.6c"), 0)
    // XCTAssertEqual(run("experimental_continuation8.6c"), 0)
    // XCTAssertEqual(run("experimental_continuation9.6c"), 0)
    XCTAssertEqual(run("fun1.6c"), 0)
    XCTAssertEqual(run("fun2.6c"), 0)
    XCTAssertEqual(run("fun3.6c"), 0)
    XCTAssertEqual(run("fun4.6c"), 0)
    XCTAssertEqual(run("fun5.6c"), 0)
    // XCTAssertEqual(run("fun6_fail_type.6c"), 0)
    XCTAssertEqual(run("fun_named_params.6c"), 0)
    XCTAssertEqual(run("fun_named_params2.6c"), 0)
    XCTAssertEqual(run("fun_recur.6c"), 0)
    XCTAssertEqual(run("funptr1.6c"), 0)
    // XCTAssertEqual(run("global_variable1.6c"), 0)
    // XCTAssertEqual(run("global_variable2.6c"), 0)
    // XCTAssertEqual(run("global_variable3.6c"), 0)
    // XCTAssertEqual(run("global_variable4.6c"), 0)
    // XCTAssertEqual(run("global_variable5.6c"), 0)
    // XCTAssertEqual(run("global_variable6.6c"), 0)
    // XCTAssertEqual(run("global_variable7.6c"), 0)
    // XCTAssertEqual(run("global_variable8.6c"), 0)
    XCTAssertEqual(run("if1.6c"), 0)
    XCTAssertEqual(run("if2.6c"), 0)
    XCTAssertEqual(run("if3.6c"), 0)
    XCTAssertEqual(run("match_any_int.6c"), 0)
    XCTAssertEqual(run("match_int.6c"), 0)
    XCTAssertEqual(run("match_int_default.6c"), 0)
    // XCTAssertEqual(run("match_type.6c"), 0)
    // XCTAssertEqual(run("next.6c"), 0)
    XCTAssertEqual(run("pattern_init.6c"), 0)
    // XCTAssertEqual(run("pattern_variable_fail.6c"), 0)
    XCTAssertEqual(run("record1.6c"), 0)
    XCTAssertEqual(run("struct1.6c"), 0)
    XCTAssertEqual(run("struct2.6c"), 0)
    XCTAssertEqual(run("struct3.6c"), 0)
    XCTAssertEqual(run("tuple1.6c"), 0)
    XCTAssertEqual(run("tuple2.6c"), 0)
    // XCTAssertEqual(run("tuple2_fail.6c"), 0)
    XCTAssertEqual(run("tuple3.6c"), 0)
    // TODO: This is now supposed to cause a type checking error?
    XCTAssertEqual(run("tuple4.6c"), 0)
    // TODO: This is now supposed to cause a type checking error?
    XCTAssertEqual(run("tuple5.6c"), 0)
    // XCTAssertEqual(run("tuple_assign.6c"), 0)
    // XCTAssertEqual(run("tuple_equality.6c"), 0)
    // XCTAssertEqual(run("tuple_equality2.6c"), 0)
    // XCTAssertEqual(run("tuple_equality3.6c"), 0)
    XCTAssertEqual(run("tuple_match.6c"), 0)
    XCTAssertEqual(run("tuple_match2.6c"), 0)
    XCTAssertEqual(run("tuple_match3.6c"), 0)
    // XCTAssertEqual(run("type_compute.6c"), 0)
    // XCTAssertEqual(run("type_compute2.6c"), 0)
    // XCTAssertEqual(run("type_compute3.6c"), 0)
    XCTAssertEqual(run("while1.6c"), 0)
    XCTAssertEqual(run("zero.6c"), 0)
  }
}
