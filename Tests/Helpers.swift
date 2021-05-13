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
  XCTAssertThrowsError(
    try expression(), message(), file: filePath, line: line
  ) {
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
      file: filePath, line: line)
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

  /// Returns `self` parsed as Carbon, or, if parsing fails, an empty AST while
  /// causing an XCTest failure.
  ///
  /// - Parameter sourceFile: the source file name used to label regions in the
  ///   resulting AST.
  /// - Parameter tracing: `true` iff Citron parser tracing should be enabled.
  func checkParsed(
    fromFile sourceFile: String = #filePath, tracing: Bool = false,
    _ message: @autoclosure () -> String = "",
    filePath: StaticString = #filePath,
    line: UInt = #line, column: Int = #column,
    topLevelCheckFunction: StaticString? = nil
  ) -> AbstractSyntaxTree {
    do {
      return try self.parsedAsCarbon(fromFile: sourceFile, tracing: tracing)
    }
    catch let errors as ErrorLog {
      failWithUnexpectedErrors(
        "Unexpected parsing errors", errors,
        filePath: filePath, line: line, column: column,
        topLevelCheckFunction: topLevelCheckFunction ?? #function)
    }
    catch let e {
      XCTFail(
        "Unexpected error during parsing: \(e)",
        file: (filePath), line: line)
    }
    return []
  }
}


extension String {
  /// Returns the carbon executable corresponding to `self`, without type
  /// checking, or `nil` (logging an XCTest failure) if errors occurred.
  func checkExecutable(
    _ message: @autoclosure () -> String = "",
    filePath: StaticString = #filePath,
    line: UInt = #line, column: Int = #column,
    topLevelCheckFunction: StaticString? = nil
  ) -> ExecutableProgram? {
    do {
      return try ExecutableProgram(
        self.parsedAsCarbon(fromFile: String(describing: filePath)))
    }
    catch let errors as ErrorLog {
      failWithUnexpectedErrors(
        "Unexpected parsing errors", errors,
        filePath: filePath, line: line, column: column,
        topLevelCheckFunction: topLevelCheckFunction ?? #function)
    }
    catch let e {
      XCTFail(
        "Unknown error during parsing: \(e)", file: (filePath), line: line)
    }
    return nil
  }
}


extension String {
  /// Interpreting `self` as string literal source code in a test file followed
  /// by a method call (see checkTypeChecks below), returns a string
  /// representing `errors` formatted in Gnu style, referring to the appropriate
  /// places in the test file.
  fileprivate func failWithUnexpectedErrors(
    _ message: String,
    _ errors: ErrorLog,
    filePath: StaticString,
    line: UInt, column: Int, topLevelCheckFunction: StaticString = #function
  ) {
    let baseNameLength = String(describing: topLevelCheckFunction)
      .enumerated().first { $1 == "(" }!.0

    let lines = self.split(separator: "\n")
    let offset = (
      line: Int(line) - (lines.count == 1 ? 0 : lines.count) - 1,
      column: column - baseNameLength
        - (lines.count == 1 ? self.count + 3 : 5))
    XCTFail(
      "\(message):\n" +
        errors.lazy.map { "\($0 + offset)" }.joined(separator: "\n"),
      file: (filePath), line: line)
  }

  /// Returns the results of parsing, name lookup, and typechecking `self`, or
  /// `nil` (logging an XCTest failure) if there are errors before type checking.
  func typeChecked(
    _ message: @autoclosure () -> String = "",
    filePath: StaticString = #filePath,
    line: UInt = #line, column: Int = #column,
    topLevelCheckFunction: StaticString? = nil
  ) -> (ExecutableProgram, typeChecker: TypeChecker, errors: ErrorLog)?
  {
    guard let executable = self.checkExecutable(
            message(), filePath: filePath, line: line, column: column,
            topLevelCheckFunction: topLevelCheckFunction ?? #function
          ) else { return nil }

    let typeChecker = TypeChecker(executable)
    return (executable, typeChecker, typeChecker.errors)
  }

  /// Causes an XCTest failure if errors occur in parsing, name lookup, or type
  /// checking.
  ///
  /// For high-quality failure messages, should be run on a single line string
  /// literal like this:
  ///
  ///     "code()".checkTypeChecks()
  ///
  /// or on a multi-line string literal like this:
  ///
  ///     """
  ///     code()
  ///     code()
  ///     """.checkTypeChecks()
  ///
  func checkTypeChecks(
    _ message: @autoclosure () -> String = "",
    filePath: StaticString = #filePath,
    line: UInt = #line, column: Int = #column,
    topLevelCheckFunction: StaticString? = nil
  ) {
    if case let .some((_, _, errors)) = self.typeChecked(
         filePath: filePath, line: line, column: column,
         topLevelCheckFunction: topLevelCheckFunction ?? #function),
       !errors.isEmpty
    {
      failWithUnexpectedErrors(
        "Unexpected compilation errors", errors,
        filePath: filePath, line: line, column: column,
        topLevelCheckFunction: topLevelCheckFunction ?? #function)
    }
  }

  /// Causes an XCTest failure if `excerpt` does not occur as part of a type
  /// checking error.
  ///
  /// For high-quality failure messages, should be run on a single line string
  /// literal like this:
  ///
  ///     "code()".checkFailsToTypeCheck(withMessage: "No such thang")
  ///
  /// or on a multi-line string literal like this:
  ///
  ///     """
  ///     code()
  ///     code()
  ///     """.checkFailsToTypeCheck(withMessage: "No such thang")
  ///
  func checkFailsToTypeCheck(
    withMessage excerpt: String,
    filePath: StaticString = #filePath,
    line: UInt = #line, column: Int = #column
  ) {
    guard case let .some((_, _, errors)) = self.typeChecked(
            filePath: filePath, line: line, column: #column,
            topLevelCheckFunction: #function),
          !(errors.contains { e in e.message.contains(excerpt) })
    else { return }

    let prefix
      = "Expected type checking error message \(String(reflecting: excerpt)) "

    if errors.isEmpty {
      XCTFail(prefix + "but none found", file: (filePath), line: line)
    }
    else {
      failWithUnexpectedErrors(
        prefix + "not found. Actual errors",
        errors, filePath: filePath, line: line, column: column)
    }
  }
}
