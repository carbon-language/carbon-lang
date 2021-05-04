// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter
import Foundation

final class NameResolutionTests: XCTestCase {
  func testNoMain() throws {
    let ast = try "var Int: x = 1;".checkParsed()
    let n = NameResolution(ast)
    XCTAssertEqual(n.definition.count, 0)
    XCTAssertEqual(n.errors, [])
  }

  func testUndeclaredName0() throws {
    let ast = try "var Y: x = 1;".checkParsed()
    let n = NameResolution(ast)
    XCTAssertEqual(n.definition.count, 0)
    n.errors.checkForMessageExcerpt("Un-declared name 'Y'")
  }

  func testDeclaredNameUse() throws {
    let ast = try """
      var Int: x = 1;
      var Int: y = x;
      """.checkParsed()
    let n = NameResolution(ast)
    XCTAssertEqual(n.definition.count, 1)
    XCTAssertEqual(n.errors, [])
  }

  func testOrderIndependence() throws {
    let ast = try """
      var Int: y = x;
      var Int: x = 1;
      """.checkParsed()
    let n = NameResolution(ast)
    XCTAssertEqual(n.definition.count, 1)
    XCTAssertEqual(n.errors, [])
  }

  func testScopeEnds() throws {
    let ast = try """
      struct X { var Int: a; }
      var Int: y = a;
      """.checkParsed()
    let n = NameResolution(ast)
    n.errors.checkForMessageExcerpt("Un-declared name 'a'")
  }

  func testRedeclaredMember() throws {
    let ast = try """
      struct X {
        var Int: a;
        var Bool: a;
      }
      """.checkParsed()
    let n = NameResolution(ast)
    n.errors.checkForMessageExcerpt("'a' already defined")
  }

  func testSelfReference() throws {
    let ast = try """
      struct X {
        var Int: a;
        var fnty ()->X: b;
      }
      """.checkParsed()
    let n = NameResolution(ast)
    XCTAssertEqual(n.definition.count, 1)
    XCTAssertEqual(n.errors, [])
  }

  func testRedeclaredAlternative() throws {
    let ast = try """
      choice X {
        Box,
        Car(Int),
        Children(Int, Bool),
        Car
      }
      """.checkParsed()
    let n = NameResolution(ast)
    n.errors.checkForMessageExcerpt("'Car' already defined")
  }

  func testExamples() throws {
    let testdata = 
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("testdata")

    for f in try! FileManager().contentsOfDirectory(atPath: testdata.path) {
      let p = testdata.appendingPathComponent(f).path

      // Skip experimental syntax for now.
      if f.hasPrefix("experimental_") { continue }

      if !f.hasSuffix("_fail.6c") {
        let ast = try checkNoThrow(
          try String(contentsOfFile: p).parsedAsCarbon(fromFile: p))

        let executable = try checkNoThrow(try ExecutableProgram(ast))
        _ = executable
      }
    }
  }
}
