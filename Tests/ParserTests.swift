import XCTest
@testable import CarbonInterpreter

prefix operator ^

/// Returns `x`, “lifted” into an `AST` node with dummy location information.
prefix func ^ <T>(x: T) -> AST<T> {
  AST(x, .empty)
}

extension String {
  /// Returns `self`, parsed as Carbon.
  func parsedAsCarbon() throws -> [Declaration] {
    let p = CarbonParser()
    for t in Tokens(in: self, from: "no file") {
      try p.consume(token: t, code: t.body.kind)
    }
    return try p.endParsing()
  }
}

final class ParserTests: XCTestCase {
  func testInit() {
    // Make sure we can even create one.
    _ = CarbonParser()
  }

  func testBasic() {
    // Parse a few tiny programs
    XCTAssertEqual(
      try! "fn main() -> Int;".parsedAsCarbon(),
      [
        ^.function(
          ^FunctionDefinition_(
            name: ^Token(.Identifier, "main"),
            parameterPattern: ^[],
            returnType: ^.intType,
            body: nil))
      ])

    XCTAssertEqual(
      try! "fn main() -> Int {}".parsedAsCarbon(),
      [
        ^.function(
          ^FunctionDefinition_(
            name: ^Token(.Identifier, "main"),
            parameterPattern: ^[],
            returnType: ^.intType,
            body: ^.block([])))
      ])

   XCTAssertEqual(
      try! "var Int: x = 0;".parsedAsCarbon(),
      [
        ^.variable(
          name: ^Token(.Identifier, "x"),
          type: ^.intType,
          initializer: ^.integerLiteral(0))
      ])
  }

  func testParseFailure() {
    XCTAssertThrowsError(try "fn ()".parsedAsCarbon()) { e in
      XCTAssertTrue(
        e is _CitronParserUnexpectedTokenError<Identifier, TokenKind>);
    }

    XCTAssertThrowsError(try "fn f".parsedAsCarbon()) { e in
      XCTAssertTrue(
        e is CitronParserUnexpectedEndOfInputError);
    }
  }
}
