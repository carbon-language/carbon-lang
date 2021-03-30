import XCTest
@testable import CarbonInterpreter
import Foundation

prefix operator ^

/// Returns `x`, “lifted” into an `AST` node with dummy location information.
prefix func ^ <T>(x: T) -> AST<T> {
  AST(x, .empty)
}

extension String {
  /// Returns `self`, parsed as Carbon.
  func parsedAsCarbon(fromFile sourceFile: String = #filePath) throws
    -> [Declaration]
  {
    let p = CarbonParser()
    for t in Tokens(in: self, from: sourceFile) {
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
            name: ^"main",
            parameterPattern: ^[],
            returnType: ^.intType,
            body: nil))
      ])

    XCTAssertEqual(
      try! "fn main() -> Int {}".parsedAsCarbon(),
      [
        ^.function(
          ^FunctionDefinition_(
            name: ^"main",
            parameterPattern: ^[],
            returnType: ^.intType,
            body: ^.block([])))
      ])

   XCTAssertEqual(
     try! "var Int: x = 0;".parsedAsCarbon(),
     [
       ^.variable(name: ^"x", type: ^.intType, initializer: ^.integerLiteral(0))
     ])
  }

  func testParseFailure() {
    XCTAssertThrowsError(try "fn ()".parsedAsCarbon()) { e in
      print(e)
      XCTAssertTrue(
        e is _CitronParserUnexpectedTokenError<AST<Token>, TokenKind>);
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
      let fileName = testdata.appendingPathComponent(f)
      do {
        _ = try String(contentsOf: fileName)
          .parsedAsCarbon(fromFile: fileName.path)
      }
      catch let e as _CitronParserUnexpectedTokenError<AST<Token>, TokenKind> {
        print(e.token, e.tokenCode)
      }
      catch let e {
        print(f)
        print(e)
      }
    }
  }
}
