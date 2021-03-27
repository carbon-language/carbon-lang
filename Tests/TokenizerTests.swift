import XCTest
@testable import CarbonInterpreter

final class TokenizerTests: XCTestCase {
  func testExample() {
    let program =
      """
      an and andey 999 // burp
        ->=-automobile->otto auto bool break case choice continue=>

        default/
      else
      // excuse me
        ==false fn fnty if\t\tint match not or return struct true type var while
      = - +( ){}[]a.b,;:
      """    
    let scannedTokens = Tokens(in: program, from: "")
    
    func tok(_ k: TokenCode, _ text: String) -> Token {
      Token(kind: .init(k), text: text)
    }
    
    func loc(_ start: (line: Int, column: Int), _ end: (line: Int, column: Int))
      -> SourceLocation
    {
      SourceLocation(
        fileName: "",
        .init(line: start.line, column: start.column)
        ..< .init(line: end.line, column: end.column))
    }
    
    typealias L = SourceLocation
    let expectedTokens: [AST<Token>] = [
      (tok(.Identifier, "an"), loc((1, 1), (1, 3))),
      (tok(.AND, "and"), loc((1, 4), (1, 7))),
      (tok(.Identifier, "andey"), loc((1, 8), (1, 13))),
      (tok(.Integer_literal, "999"), loc((1, 14), (1, 17))),
      (tok(.ARROW, "->"), loc((2, 3), (2, 5))),
      (tok(.EQUAL, "="), loc((2, 5), (2, 6))),
      (tok(.MINUS, "-"), loc((2, 6), (2, 7))),
      (tok(.Identifier, "automobile"), loc((2, 7), (2, 17))),
      (tok(.ARROW, "->"), loc((2, 17), (2, 19))),
      (tok(.Identifier, "otto"), loc((2, 19), (2, 23))),
      (tok(.AUTO, "auto"), loc((2, 24), (2, 28))),
      (tok(.BOOL, "bool"), loc((2, 29), (2, 33))),
      (tok(.BREAK, "break"), loc((2, 34), (2, 39))),
      (tok(.CASE, "case"), loc((2, 40), (2, 44))),
      (tok(.CHOICE, "choice"), loc((2, 45), (2, 51))),
      (tok(.CONTINUE, "continue"), loc((2, 52), (2, 60))),
      (tok(.DBLARROW, "=>"), loc((2, 60), (2, 62))),
      (tok(.DEFAULT, "default"), loc((4, 3), (4, 10))),
      (tok(.ILLEGAL_CHARACTER, "/"), loc((4, 10), (4, 11))),
      (tok(.ELSE, "else"), loc((5, 1), (5, 5))),
      (tok(.EQUAL_EQUAL, "=="), loc((7, 3), (7, 5))),
      (tok(.FALSE, "false"), loc((7, 5), (7, 10))),
      (tok(.FN, "fn"), loc((7, 11), (7, 13))),
      (tok(.FNTY, "fnty"), loc((7, 14), (7, 18))),
      (tok(.IF, "if"), loc((7, 19), (7, 21))),
      (tok(.INT, "int"), loc((7, 23), (7, 26))),
      (tok(.MATCH, "match"), loc((7, 27), (7, 32))),
      (tok(.NOT, "not"), loc((7, 33), (7, 36))),
      (tok(.OR, "or"), loc((7, 37), (7, 39))),
      (tok(.RETURN, "return"), loc((7, 40), (7, 46))),
      (tok(.STRUCT, "struct"), loc((7, 47), (7, 53))),
      (tok(.TRUE, "true"), loc((7, 54), (7, 58))),
      (tok(.TYPE, "type"), loc((7, 59), (7, 63))),
      (tok(.VAR, "var"), loc((7, 64), (7, 67))),
      (tok(.WHILE, "while"), loc((7, 68), (7, 73))),
      (tok(.EQUAL, "="), loc((8, 1), (8, 2))),
      (tok(.MINUS, "-"), loc((8, 3), (8, 4))),
      (tok(.PLUS, "+"), loc((8, 5), (8, 6))),
      (tok(.LEFT_PARENTHESIS, "("), loc((8, 6), (8, 7))),
      (tok(.RIGHT_PARENTHESIS, ")"), loc((8, 8), (8, 9))),
      (tok(.LEFT_CURLY_BRACE, "{"), loc((8, 9), (8, 10))),
      (tok(.RIGHT_CURLY_BRACE, "}"), loc((8, 10), (8, 11))),
      (tok(.LEFT_SQUARE_BRACKET, "["), loc((8, 11), (8, 12))),
      (tok(.RIGHT_SQUARE_BRACKET, "]"), loc((8, 12), (8, 13))),
      (tok(.Identifier, "a"), loc((8, 13), (8, 14))),
      (tok(.PERIOD, "."), loc((8, 14), (8, 15))),
      (tok(.Identifier, "b"), loc((8, 15), (8, 16))),
      (tok(.COMMA, ","), loc((8, 16), (8, 17))),
      (tok(.SEMICOLON, ";"), loc((8, 17), (8, 18))),
      (tok(.COLON, ":"), loc((8, 18), (8, 19)))
    ]
    
    for (t, e) in zip(scannedTokens, expectedTokens) {
      XCTAssertEqual(t.body, e.body)
      XCTAssertEqual(t.location, e.location)
    }
  }
}
