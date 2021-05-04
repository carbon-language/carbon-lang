// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

final class TestSemanticAnalysis: XCTestCase {
  func testNoMain() throws {
    let executable = try "var Int: x = 3;".checkExecutable()
    XCTAssertNil(executable.entryPoint)
  }
      
  func testMultiMain() {
    let source = """
      fn main() -> Int {}
      var Int: x = 3;
      fn main() -> Void {}
      """
    checkThrows(try ExecutableProgram(source.parsedAsCarbon()))
    {
      (e: ErrorLog) in XCTAssert(e[0].message.contains("already defined"))
    }
  }                      
      
  func testMinimal() throws {
    let exe = try "fn main() -> Int {}".checkExecutable()

    guard let entryPoint = checkNonNil(exe.entryPoint) else {
      XCTFail("Missing unambiguous main()")
      return
    }
    guard case let .name(name) = entryPoint.callee else {
      XCTFail("Callee is not an identifier: \(entryPoint.callee)")
      return
    }
    
    // Nothing interesting to check about exe yet.
    XCTAssertEqual(name.text, "main")
  }
}
