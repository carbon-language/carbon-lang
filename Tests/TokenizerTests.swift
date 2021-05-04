// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

final class TokenizerTests: XCTestCase {
  func testExample() {
    let program =
      """
      an and andey 999 // burp
        ->=-automobile->otto auto Bool break case choice continue=>

        default/\r\nelse
      // excuse me
        ==false fn fnty if\t\tInt match not or return struct true Type var while
      = - +( ){}[]a.b,;:
      //\r\n.
      """    
    let scannedTokens = Array(Tokens(in: program, from: ""))

    /// Returns the token with the given contents.
    ///
    /// This is just a helper to make `expectedTokens` (below) read nicely.
    func token(
      _ k: TokenID, _ text: String,
      from startLine: Int, _ startColumn: Int,
      to endLine: Int, _ endColumn: Int) -> Token
    {
      Token(
        k, text,
        ASTSite(
          devaluing: SourceRegion(
            fileName: "",
            .init(line: startLine, column: startColumn)
              ..< .init(line: endLine, column: endColumn))))
    }
    
    typealias L = SourceRegion
    let expectedTokens: [Token] = [
      token(.Identifier, "an", from: 1, 1, to: 1, 3),
      token(.AND, "and", from: 1, 4, to: 1, 7),
      token(.Identifier, "andey", from: 1, 8, to: 1, 13),
      token(.Integer_literal, "999", from: 1, 14, to: 1, 17),
      token(.ARROW, "->", from: 2, 3, to: 2, 5),
      token(.EQUAL, "=", from: 2, 5, to: 2, 6),
      token(.MINUS, "-", from: 2, 6, to: 2, 7),
      token(.Identifier, "automobile", from: 2, 7, to: 2, 17),
      token(.ARROW, "->", from: 2, 17, to: 2, 19),
      token(.Identifier, "otto", from: 2, 19, to: 2, 23),
      token(.AUTO, "auto", from: 2, 24, to: 2, 28),
      token(.BOOL, "Bool", from: 2, 29, to: 2, 33),
      token(.BREAK, "break", from: 2, 34, to: 2, 39),
      token(.CASE, "case", from: 2, 40, to: 2, 44),
      token(.CHOICE, "choice", from: 2, 45, to: 2, 51),
      token(.CONTINUE, "continue", from: 2, 52, to: 2, 60),
      token(.DBLARROW, "=>", from: 2, 60, to: 2, 62),
      token(.DEFAULT, "default", from: 4, 3, to: 4, 10),
      token(.ILLEGAL_CHARACTER, "/", from: 4, 10, to: 4, 11),
      token(.ELSE, "else", from: 5, 1, to: 5, 5),
      token(.EQUAL_EQUAL, "==", from: 7, 3, to: 7, 5),
      token(.FALSE, "false", from: 7, 5, to: 7, 10),
      token(.FN, "fn", from: 7, 11, to: 7, 13),
      token(.FNTY, "fnty", from: 7, 14, to: 7, 18),
      token(.IF, "if", from: 7, 19, to: 7, 21),
      token(.INT, "Int", from: 7, 23, to: 7, 26),
      token(.MATCH, "match", from: 7, 27, to: 7, 32),
      token(.NOT, "not", from: 7, 33, to: 7, 36),
      token(.OR, "or", from: 7, 37, to: 7, 39),
      token(.RETURN, "return", from: 7, 40, to: 7, 46),
      token(.STRUCT, "struct", from: 7, 47, to: 7, 53),
      token(.TRUE, "true", from: 7, 54, to: 7, 58),
      token(.TYPE, "Type", from: 7, 59, to: 7, 63),
      token(.VAR, "var", from: 7, 64, to: 7, 67),
      token(.WHILE, "while", from: 7, 68, to: 7, 73),
      token(.EQUAL, "=", from: 8, 1, to: 8, 2),
      token(.MINUS, "-", from: 8, 3, to: 8, 4),
      token(.PLUS, "+", from: 8, 5, to: 8, 6),
      token(.LEFT_PARENTHESIS, "(", from: 8, 6, to: 8, 7),
      token(.RIGHT_PARENTHESIS, ")", from: 8, 8, to: 8, 9),
      token(.LEFT_CURLY_BRACE, "{", from: 8, 9, to: 8, 10),
      token(.RIGHT_CURLY_BRACE, "}", from: 8, 10, to: 8, 11),
      token(.LEFT_SQUARE_BRACKET, "[", from: 8, 11, to: 8, 12),
      token(.RIGHT_SQUARE_BRACKET, "]", from: 8, 12, to: 8, 13),
      token(.Identifier, "a", from: 8, 13, to: 8, 14),
      token(.PERIOD, ".", from: 8, 14, to: 8, 15),
      token(.Identifier, "b", from: 8, 15, to: 8, 16),
      token(.COMMA, ",", from: 8, 16, to: 8, 17),
      token(.SEMICOLON, ";", from: 8, 17, to: 8, 18),
      token(.COLON, ":", from: 8, 18, to: 8, 19),
      token(.PERIOD, ".", from: 10, 1, to: 10, 2)
    ]

    let sourceLines = program.split(
      omittingEmptySubsequences: false, whereSeparator: \.isNewline)

    /// Returns the text denoted by `s` in `program`.
    func text(_ l: ASTSite) -> String {
      let (start, end) = (l.region.span.lowerBound, l.region.span.upperBound)
      let startIndex = sourceLines[start.line - 1].index(
        sourceLines[start.line - 1].startIndex, offsetBy: start.column - 1)
      let endIndex = sourceLines[end.line - 1].index(
        sourceLines[end.line - 1].startIndex, offsetBy: end.column - 1)
      return String(program[startIndex..<endIndex])
    }

    for (t, e) in zip(scannedTokens, expectedTokens) {
      XCTAssertEqual(t, e, "Unexpected token value.")
      XCTAssertEqual(t.site, e.site, "Unexpected token region.")
      XCTAssertEqual(t.text, text(t.site),"Token text doesn't match source.")
    }
    let extraScanned = scannedTokens.dropFirst(expectedTokens.count)
    XCTAssert(
      extraScanned.isEmpty,
      "Unexpected tokens at end of input: \(Array(extraScanned))")
    
    let extraExpected = expectedTokens.dropFirst(scannedTokens.count)
    XCTAssert(
      extraExpected.isEmpty,
      "Expected tokens not found at end of input \(Array(extraExpected))")
  }
}
