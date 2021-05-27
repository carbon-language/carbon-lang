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
    fromFile sourceFile: String? = nil, tracing: Bool = false,
    _ message: @autoclosure () -> String = "",
    filePath: StaticString = #filePath,
    line: UInt = #line, column: Int = #column,
    topLevelCheckFunction: StaticString? = nil
  ) -> AbstractSyntaxTree {
    do {
      return try self.parsedAsCarbon(
        fromFile: sourceFile ?? filePath.description, tracing: tracing)
    }
    catch let errors as ErrorLog {
      failWithUnexpectedErrors(
        fromFile: sourceFile,
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
  /// Returns the carbon executable corresponding to `self`, or `nil` (logging
  /// an XCTest failure) if errors occurred.
  func checkExecutable(
    fromFile sourceFile: String? = nil,
    _ message: @autoclosure () -> String = "",
    filePath: StaticString = #filePath,
    line: UInt = #line, column: Int = #column,
    topLevelCheckFunction: StaticString? = nil
  ) -> ExecutableProgram? {
    do {
      return try ExecutableProgram(
        self.parsedAsCarbon(fromFile: sourceFile ?? filePath.description))
    }
    catch let errors as ErrorLog {
      failWithUnexpectedErrors(
        fromFile: sourceFile,
        "Unexpected errors", errors,
        filePath: filePath, line: line, column: column,
        topLevelCheckFunction: topLevelCheckFunction ?? #function)
    }
    catch let e {
      XCTFail(
        "Unknown error: \(e)", file: (filePath), line: line)
    }
    return nil
  }

  /// Returns the AST corresponding to `self` with its name resolution results,
  /// or `nil` (logging an XCTest failure) if errors occurred.
  func checkNameResolution(
    fromFile sourceFile: String? = nil,
    _ message: @autoclosure () -> String = "",
    filePath: StaticString = #filePath,
    line: UInt = #line, column: Int = #column,
    topLevelCheckFunction: StaticString? = nil
  ) -> (parsedProgram: AbstractSyntaxTree, nameResolution: NameResolution)? {
    do {
      let parsedProgram
        = try self.parsedAsCarbon(fromFile: sourceFile ?? filePath.description)
      let nameResolution = NameResolution(parsedProgram)
      if !nameResolution.errors.isEmpty { throw nameResolution.errors }
      return (parsedProgram, nameResolution)
    }
    catch let errors as ErrorLog {
      failWithUnexpectedErrors(
        fromFile: sourceFile,
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
  /// Returns a string representing `errors` formatted in Gnu style, referring
  /// to the appropriate places in filePath.
  ///
  fileprivate func failWithUnexpectedErrors(
    fromFile sourceFile: String? = nil,
    _ message: String,
    _ errors: ErrorLog,
    filePath: StaticString,
    line: UInt, column: Int,
    topLevelCheckFunction: StaticString = #function
  ) {
    var offset = (line: 0, column: 0)

    if sourceFile == nil {
      let baseNameLength = String(describing: topLevelCheckFunction)
        .enumerated().first { $1 == "(" }!.0
      let lineCount = self.split(separator: "\n").count
      offset = lineCount <= 1
        ? (Int(line) - 1,             column - baseNameLength - self.count + 3)
        : (Int(line) - lineCount - 1, column - baseNameLength - 5)
    }

    XCTFail(
      "\(message):\n" +
        errors.lazy.map { "\($0 + offset)" }.joined(separator: "\n"),
      file: (filePath), line: line)
  }

  /// Returns the errors from type checking, or `nil` (logging an XCTest
  /// failure) if there are errors before type checking.
  func typeCheckingErrors(
    fromFile sourceFile: String? = nil,
    _ message: @autoclosure () -> String = "",
    filePath: StaticString = #filePath,
    line: UInt = #line, column: Int = #column,
    topLevelCheckFunction: StaticString? = nil
  ) -> ErrorLog?
  {
    guard let (parsedProgram, nameLookup) = self.checkNameResolution(
            fromFile: sourceFile,
            message(), filePath: filePath, line: line, column: column,
            topLevelCheckFunction: topLevelCheckFunction ?? #function
          ) else { return nil }

    let typeChecker = TypeChecker(parsedProgram, nameLookup: nameLookup)
    return typeChecker.errors
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
    fromFile sourceFile: String? = nil,
    _ message: @autoclosure () -> String = "",
    filePath: StaticString = #filePath,
    line: UInt = #line, column: Int = #column,
    topLevelCheckFunction: StaticString? = nil
  ) {
    if let errors = self.typeCheckingErrors(
         fromFile: sourceFile,
         filePath: filePath, line: line, column: column,
         topLevelCheckFunction: topLevelCheckFunction ?? #function),
       !errors.isEmpty
    {
      failWithUnexpectedErrors(
        fromFile: sourceFile,
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
    fromFile sourceFile: String? = nil,
    withMessage excerpt: String,
    filePath: StaticString = #filePath,
    line: UInt = #line, column: Int = #column
  ) {
    guard let errors = self.typeCheckingErrors(
            fromFile: sourceFile,
            filePath: filePath, line: line, column: #column),
          !(errors.contains { e in
              excerpt.isEmpty // Work around wrong semantics of Foundation.
                || e.message.contains(excerpt) })
    else { return }

    let prefix
      = "Expected type checking error message \(String(reflecting: excerpt)) "

    if errors.isEmpty {
      XCTFail(
        prefix + "but none found" + (sourceFile.map { " in \($0)" } ?? ""),
        file: (filePath), line: line)
    }
    else {
      failWithUnexpectedErrors(
        fromFile: sourceFile,
        prefix + "not found. Actual errors",
        errors, filePath: filePath, line: line, column: column)
    }
  }
}

// TODO: Large-scale cleanups in this file: factor out the expected/unexpected
// error handling; remove checkTypeChecks, ...
// TODO: Translate parsing exceptions into CarbonErrors.
