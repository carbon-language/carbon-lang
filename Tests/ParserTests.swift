// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter
import Foundation

extension String {
  /// Returns `self`, parsed as Carbon.
  func parsedAsCarbon(fromFile sourceFile: String = #filePath, tracing: Bool = false) throws
    -> [TopLevelDeclaration]
  {
    let p = CarbonParser()
    p.isTracingEnabled = tracing
    for t in Tokens(in: self, from: sourceFile) {
      try p.consume(token: t, code: t.kind)
    }
    return try p.endParsing()
  }
}

final class ParserTests: XCTestCase {
  func testInit() {
    // Make sure we can even create one.
    _ = CarbonParser()
  }

  func testBasic0() {
    // Parse a few tiny programs
    guard let p = CheckNoThrow(try "fn main() -> Int;".parsedAsCarbon())
    else { return }
    XCTAssertEqual(p.count, 1)
    let p0 = p[0]
    guard case .function(let d) = p0 else { XCTFail("\(p0)"); return }
    XCTAssertEqual(d.name.text, "main")
    XCTAssertEqual(d.parameters.elements, [])
    if case .intType(_) = d.returnType {} else { XCTFail("\(d.returnType)") }
    XCTAssertNil(d.body)
  }

  func testBasic1() {
    guard let p = CheckNoThrow(try "fn main() -> Int {}".parsedAsCarbon())
    else { return }
    XCTAssertEqual(p.count, 1)
    let p0 = p[0]
    guard case .function(let d) = p0 else { XCTFail("\(p0)"); return }
    XCTAssertEqual(d.name.text, "main")
    XCTAssertEqual(d.parameters.elements, [])
    if case .intType = d.returnType {} else { XCTFail("\(d.returnType)") }
    guard let b = d.body, case .block(let c, _) = b else {
      XCTFail("\(d)")
      return
    }
    XCTAssertEqual(c, [])
  }

  func testBasic2() {
    guard let p = CheckNoThrow(try "var Int: x = 0;".parsedAsCarbon())
    else { return }
    XCTAssertEqual(p.count, 1)
    let p0 = p[0]
    guard case let .initialization(i) = p0 else {
      XCTFail("\(p0)")
      return
    }
    guard case let .variable(b) = i.bindings else {
      XCTFail("\(i)")
      return
    }
    XCTAssertEqual(b.boundName.text, "x")

    if case .literal(.intType) = b.type {} else { XCTFail("\(b.type)") }
    if case let .integerLiteral(n, _) = i.initializer { XCTAssertEqual(n, 0) }
    else { XCTFail("\(i.initializer)") }
  }

  func testParseFailure() {
    XCTAssertThrowsError(try "fn ()".parsedAsCarbon()) { e in
      print(e)
      XCTAssertTrue(
        e is _CitronParserUnexpectedTokenError<Token, TokenID>);
    }

    XCTAssertThrowsError(try "fn f".parsedAsCarbon()) { e in
      XCTAssertTrue(e is CitronParserUnexpectedEndOfInputError);
    }
  }

  func testExamples() {
    let testdata = 
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("testdata")

    for f in try! FileManager().contentsOfDirectory(atPath: testdata.path) {
      let p = testdata.appendingPathComponent(f).path

      // Skip experimental syntax for now.
      if f.hasPrefix("experimental_") { continue }

      if f.hasSuffix("_fail.6c") {
        let s = try! String(contentsOfFile: p)
        XCTAssertThrowsError(try s.parsedAsCarbon(fromFile: p))
      }
      else {
        XCTAssertNoThrow(
          try String(contentsOfFile: p).parsedAsCarbon(fromFile: p))
      }
    }
  }
}
