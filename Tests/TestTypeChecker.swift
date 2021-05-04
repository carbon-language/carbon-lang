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

  func testStructStructMember() throws {
    let executable = try """
      struct X { var Int: y; }
      struct Z { var X: a; }
      """.checkExecutable()

    let typeChecker = TypeChecker(executable)
    XCTAssertEqual(typeChecker.errors, [])
  }

  func testStructNonTypeExpression0() throws {
    let executable = try "struct X { var 42: y; }".checkExecutable()

    let typeChecker = TypeChecker(executable)
    typeChecker.errors.checkForMessageExcerpt("Not a type expression")
  }

  func testChoice() throws {
    let executable = try """
      choice X {
        Box,
        Car(Int),
        Children(Int, Bool)
      }
      """.checkExecutable()

    let typeChecker = TypeChecker(executable)
    XCTAssertEqual(typeChecker.errors, [])
  }

  func testChoiceChoiceMember() throws {
    let executable = try """
      choice Y {
        Fork, Knife(X), Spoon(X, X)
      }
      choice X {
        Box,
        Car(Int),
        Children(Int, Bool)
      }
      """.checkExecutable()

    let typeChecker = TypeChecker(executable)
    XCTAssertEqual(typeChecker.errors, [])
  }

  func testChoiceNonTypeExpression() throws {
    let executable = try "choice X { Bog(42) }".checkExecutable()
    let typeChecker = TypeChecker(executable)
    typeChecker.errors.checkForMessageExcerpt("Not a type expression")
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
