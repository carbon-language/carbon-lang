// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

final class TestEvaluateCall: XCTestCase {
  func testMinimal() throws {
    let exe = try "fn main() -> Int { return 0; }".checkExecutable()

    var engine = Interpreter(exe)
    engine.start()
  
    while true {
      if case .exited(let exitCode) = engine.step() {
        XCTAssertEqual(0, exitCode)
        break
      }
    }
  }
}

final class InterpreterTests: XCTestCase {
  
}

