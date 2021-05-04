// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest

func checkNonNil<T>(
  _ expression: @autoclosure () -> T?,
  _ message: @autoclosure () -> String = "",
  filePath: StaticString = #filePath,
  line: UInt = #line
) -> T? {
  let r = expression()
  XCTAssertNotNil(r, message(), file: filePath, line: line)
  return r
}

/// Returns `expression()`, creating an XCTest failure and propagating the error
/// if the evaluation throws.
///
/// Use this in cases where the test can't proceed if `expression()` throws.
func checkNoThrow<T>(
  _ expression: @autoclosure () throws -> T,
  _ message: @autoclosure () -> String = "",
  filePath: StaticString = #filePath,
  line: UInt = #line
) throws -> T {
  do {
    return try expression()
  }
  catch {
    XCTAssertNoThrow(
      try { throw error }(), message(), file: filePath, line: line)
    throw XCTSkip() // Don't report another failure
  }
}

func checkThrows<T, E: Error>(
  _ expression: @autoclosure () throws -> T,
  _ message: @autoclosure () -> String = "",
  filePath: StaticString = #filePath,
  line: UInt = #line,
  handler: (E) -> Void
) {
  XCTAssertThrowsError(try expression(), message(), file: filePath, line: line) {
    let e = $0 as? E
    if let e1 = checkNonNil(e, "Unexpected exception kind: \($0)") {
      handler(e1)
    }
  }
}
