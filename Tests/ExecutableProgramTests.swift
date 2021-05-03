// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

final class TestSemanticAnalysis: XCTestCase {
  func testNoMain() {
    CheckThrows(
      try ExecutableProgram("var Int: x = 3;".parsedAsCarbon())
    ) {
      (e: ErrorLog) in XCTAssert(e[0].message.contains("No nullary main"))
    }
  }                      
      
  func testMultiMain() {
    let source = """
      fn main() -> Int {}
      var Int: x = 3;
      fn main() -> Void {}
      """
    CheckThrows(try ExecutableProgram(source.parsedAsCarbon()))
    {
      (e: ErrorLog) in XCTAssert(e[0].message.contains("already defined"))
    }
  }                      
      
  func testMinimal() {
    guard let exe = CheckNoThrow(
      try ExecutableProgram(
	"fn main() -> Int {}".parsedAsCarbon(fromFile: "main.6c")))
    else { return }
    
    guard case let .name(name) = exe.mainCall.callee else {
      XCTFail("Callee is not an identifier: \(exe.mainCall.callee)")
      return
    }
    
    // Nothing interesting to check about exe yet.
    XCTAssertEqual(name.text, "main")
  }
}
