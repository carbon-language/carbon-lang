// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A program whose semantic analysis/compilation is complete, containing all
/// the information necessary for execution.
struct ExecutableProgram {
  /// The result of running the parser
  let ast: AbstractSyntaxTree

  /// The entry point for this program.
  let mainCall: FunctionCall<Expression>

  /// A mapping from identifier to its definition.
  let definition: ASTDictionary<Identifier, Declaration>

  /// Constructs an instance for the given parser output, or throws `ErrorLog`
  /// if the program is ill-formed.
  init(_ program: AbstractSyntaxTree) throws {
    self.ast = program
    let r = NameResolution(program)

    if !r.errors.isEmpty { throw r.errors }
    switch unambiguousMain(in: ast) {
    case let .failure(e): throw [e]
    case let .success(main):
      // Synthesize a call to `main` to act as the entry point for the program
      let mainID = Identifier(text: "main", site: topColumns(1...4))
      var definition = r.definition
      definition[mainID] = main
      self.definition = definition
      let mainExpression = Expression.name(mainID)
      let arguments = TupleLiteral([], topColumns(5...6))
      self.mainCall = FunctionCall<Expression>(callee: mainExpression, arguments: arguments)
    }
  }
}

/// Return the site covering the given columns of line 1 in a mythical
/// file called "<TOP>".
fileprivate func topColumns(_ r: ClosedRange<Int>) -> ASTSite {
  ASTSite(
    devaluing: SourceRegion(
      fileName: "<TOP>",
      .init(line: 1, column: r.lowerBound)
        ..< .init(line: 1, column: r.upperBound + 1)))
}

/// Returns the unique top-level nullary main() function defined in
/// `parsedProgram`, or reports a suitable CompileError if that doesn't exist.
fileprivate func unambiguousMain(
  in parsedProgram: AbstractSyntaxTree
) -> Result<FunctionDefinition, CompileError> {
  let mainCandidates: [FunctionDefinition] = parsedProgram.compactMap {
    if case .function(let f) = $0,
       f.name.text == "main",
       f.parameters.isEmpty
    { return f } else { return nil }
  }

  assert(
    mainCandidates.count <= 1,
    "name resolution should have ruled out duplicate definitions.")

  guard let r = mainCandidates.first else {
    return .failure(CompileError("No nullary main() found.", at: .empty))
  }
  return .success(r)
}
