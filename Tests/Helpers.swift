// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

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

extension ErrorLog {
  func checkForMessageExcerpt(
    _ excerpt: String,
    _ message: @autoclosure () -> String? = nil,
    filePath: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssert(
      self.contains { $0.message.contains(excerpt) },
      message()
        ?? "expecting message \(String(reflecting: excerpt)) in \(self)",
      file: filePath,
      line: line)
  }
}

extension String {
  /// Returns `self` parsed as Carbon, throwing an error if parsing fails.
  ///
  /// - Parameter sourceFile: the source file name used to label regions in the
  ///   resulting AST.
  /// - Parameter tracing: `true` iff Citron parser tracing should be enabled.
  func parsedAsCarbon(
    fromFile sourceFile: String = #filePath, tracing: Bool = false
  ) throws -> AbstractSyntaxTree {
    let p = CarbonParser()
    p.isTracingEnabled = tracing
    for t in Tokens(in: self, from: sourceFile) {
      try p.consume(token: t, code: t.kind)
    }
    return try p.endParsing()
  }

  /// Returns `self` parsed as Carbon, throwing an error and causing an XCTest
  /// failure if parsing fails.
  ///
  /// - Parameter sourceFile: the source file name used to label regions in the
  ///   resulting AST.
  /// - Parameter tracing: `true` iff Citron parser tracing should be enabled.
  func checkParsed(
    fromFile sourceFile: String = #filePath, tracing: Bool = false,
    _ message: @autoclosure () -> String = "",
    filePath: StaticString = #filePath,
    line: UInt = #line
  ) throws -> AbstractSyntaxTree {
    try checkNoThrow(
      self.parsedAsCarbon(fromFile: sourceFile, tracing: tracing),
      message(), filePath: filePath, line: line)
  }
}


extension String {
  /// Returns the carbon executable corresponding to `self`, without type
  /// checking.
  ///
  /// Throws and causes an XCTest failure if errors occurred in parsing or
  /// name lookup.
  func checkExecutable(
    _ message: @autoclosure () -> String = "",
    filePath: StaticString = #filePath,
    line: UInt = #line
  ) throws -> ExecutableProgram {
    try checkNoThrow(
      try ExecutableProgram(
        self.parsedAsCarbon(fromFile: String(describing: filePath))),
      message(), filePath: filePath, line: line)
  }
}

extension String {
  /// Returns the results of parsing, name lookup, and typechecking `self`.
  ///
  /// Throws and causes an XCTest failure if errors occurred in parsing or
  /// name lookup.
  func typeChecked(
    _ message: @autoclosure () -> String = "",
    filePath: StaticString = #filePath,
    line: UInt = #line
  ) throws -> (ExecutableProgram, typeChecker: TypeChecker, errors: ErrorLog)
  {
    let executable = try self.checkExecutable(
      message(), filePath: filePath, line: line)
    let typeChecker = TypeChecker(executable)
    return (executable, typeChecker, typeChecker.errors)
  }

  /// Causes an XCTest failure if errors occur in parsing or
  /// name lookup, or type checking.
  func checkTypeChecks(
    _ message: @autoclosure () -> String = "",
    filePath: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertEqual(
      try self.typeChecked(
        message(), filePath: filePath, line: line).errors, [])
  }
}
