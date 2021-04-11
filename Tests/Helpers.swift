// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest

func CheckNonNil<T>(
  _ expression: @autoclosure () -> T?,
  _ message: @autoclosure () -> String = "",
  file: StaticString = (#filePath),
  line: UInt = #line
) -> T? {
  let r = expression()
  XCTAssertNotNil(r, message(), file: file, line: line)
  return r
}

func CheckNoThrow<T>(
  _ expression: @autoclosure () throws -> T,
  _ message: @autoclosure () -> String = "",
  file: StaticString = (#filePath),
  line: UInt = #line
) -> T? {
  var r: T?
  XCTAssertNoThrow(
    try { r = try expression() }(), message(), file: file, line: line)
  return r
}

func CheckThrows<T, E: Error>(
  _ expression: @autoclosure () throws -> T,
  _ message: @autoclosure () -> String = "",
  file: StaticString = (#filePath),
  line: UInt = #line,
  handler: (E) -> Void
) {
  XCTAssertThrowsError(try expression(), message(), file: file, line: line) {
    let e = $0 as? E
    XCTAssertNotNil(e, "Unexpected exception kind: \($0)")
    handler(e!)
  }
}
