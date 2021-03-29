import XCTest
@testable import CarbonInterpreter

final class ParserTests: XCTestCase {
  func testInit() {
    // Make sure we can even create one.
    _ = CarbonParser()
  }

  func testBasic() {
    let p = CarbonParser()
    for t in Tokens(in: "fn main() -> Int;", from: "noFile") {
      try! p.consume(token: t, code: t.body.kind)
    }
    print(try! p.endParsing())
  }
}
