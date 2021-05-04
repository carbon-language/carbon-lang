// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

final class TestEvaluateCall: XCTestCase {
  func testMinimal() throws {
    let ast = try checkNoThrow(try "fn main() -> Int {}".parsedAsCarbon())
    let exe = try checkNoThrow(try ExecutableProgram(ast))

    var engine = Interpreter(exe)

    // Allocate an address for the return value.
    let resultAddress = engine.memory.allocate(boundTo: .int, from: .empty)

    guard let mainCall = checkNonNil(exe.entryPoint) else { return }

    let call = EvaluateCall(
      call: mainCall,
      callerContext: engine.functionContext, returnValueStorage: resultAddress)

    engine.pushTodo_testingOnly(call)
    while engine.termination == nil {
      engine.step()
    }
  }
}

final class InterpreterTests: XCTestCase {
  
}

