// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

final class TestEvaluateCall: XCTestCase {
  func testMinimal() {
    guard let exe = "fn main() -> Int { return 0; }".checkExecutable() else {
      return
    }

    var engine = Interpreter(exe)
    XCTAssertEqual(0, engine.run())
  }
}

final class InterpreterTests: XCTestCase {
  
}

