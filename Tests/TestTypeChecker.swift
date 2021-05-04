// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

final class TypeCheckNominalTypeDeclaration: XCTestCase {

  func testStruct() throws {
    let executable = try "struct X { var Int: y; }".checkExecutable()

    let typeChecker = TypeChecker(executable)
    XCTAssertEqual(typeChecker.errors, [])
  }

  func testStructNonTypeExpression() throws {
    let executable = try "struct X { var 42: y; }".checkExecutable()

    let typeChecker = TypeChecker(executable)
    XCTAssert(
      typeChecker.errors.contains {
        $0.message.contains("Not a type expression") },
      String(describing: typeChecker.errors)
    )
  }

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
