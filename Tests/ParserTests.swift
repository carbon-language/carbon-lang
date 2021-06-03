// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter
import Foundation

final class ParserTests: XCTestCase {
  func testInit() {
    // Make sure we can even create one.
    _ = CarbonParser()
  }

  let o = ASTSite.empty
  
  func testBasic0() {
    // Parse a few tiny programs
    let p = "fn main() -> Int;".checkParsed()

    XCTAssertEqual(
      p,
      [
        .function(
          FunctionDefinition(
            name: Identifier(text: "main", site: o),
            parameters: TupleSyntax([], o),
            returnType: .expression(TypeExpression(.intType(o))),
            body: nil,
            site: o))])
  }

  func testBasic1() {
    let p = "fn main() -> Int {}".checkParsed()

    XCTAssertEqual(
      p,
      [
        .function(
          FunctionDefinition(
            name: Identifier(text: "main", site: o),
            parameters: TupleSyntax([], o),
            returnType: .expression(TypeExpression(.intType(o))),
            body: .block([], o),
            site: o))])
  }

  func testBasic2() {
    let p = "var Int x = 0;".checkParsed()

    XCTAssertEqual(
      p,
      [
        .initialization(
          Initialization(
            bindings: .variable(
              SimpleBinding(
                type: .expression(TypeExpression(.intType(o))),
                name: Identifier(text: "x", site: o))),
            initializer: .integerLiteral(0, o),
            site: o))])
  }

  func testFunctionTypePattern() {
    // Even though the C++ implementation isn't parsing this way, Jeremy likes
    // this parse better, without parens.
    let p = "var fnty()->Type x = 0;".checkParsed()

    XCTAssertEqual(
      p,
      [
        .initialization(
          Initialization(
            bindings: .variable(
              SimpleBinding(
                type:
                  .expression(
                    TypeExpression(
                      .functionType(
                        FunctionTypeSyntax(
                          parameters: TupleSyntax([], o),
                          returnType: TypeExpression(.typeType(o)),
                          site: o)))),
                name: Identifier(text: "x", site: o))),
            initializer: .integerLiteral(0, o),
            site: o))])
  }

  func testParenthesizedPattern() {
    let p = "var fnty()->(Type x) = 0;".checkParsed()

    XCTAssertEqual(
      p,
      [
        .initialization(
          Initialization(
            bindings: .functionType(
              FunctionTypeSyntax(
                parameters: TupleSyntax([], o),
                returnType: .variable(
                  SimpleBinding(
                    type: .expression(TypeExpression(.typeType(o))),
                    name: Identifier(text: "x", site: o))), site: o)),
            initializer: .integerLiteral(0, o),
            site: o))])
  }

  func testParseFailure() {
    XCTAssertThrowsError(try "fn ()".parsedAsCarbon()) { e in
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
