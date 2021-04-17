// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter
import Foundation

extension String {
  /// Returns `self`, parsed as Carbon.
  func parsedAsCarbon(fromFile sourceFile: String = #filePath) throws
    -> [TopLevelDeclaration]
  {
    let p = CarbonParser()
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
    XCTAssertEqual(d.parameterPattern.elements, [])
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
    XCTAssertEqual(d.parameterPattern.elements, [])
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
    guard case let .variable(v) = p0 else {
      XCTFail("\(p0)")
      return
    }
    XCTAssertEqual(v.name.text, "x")

    if case .intType = v.type {} else { XCTFail("\(v.type)") }

    if case let .integerLiteral(n, _) = v.initializer { XCTAssertEqual(n, 0) }
    else { XCTFail("\(v.initializer)") }
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
      XCTAssertNoThrow(
        try String(contentsOfFile: p).parsedAsCarbon(fromFile: p))
    }
  }
}
