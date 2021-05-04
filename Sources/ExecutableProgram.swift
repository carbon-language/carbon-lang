// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A program whose semantic analysis/compilation is complete, containing all
/// the information necessary for execution.
struct ExecutableProgram {
  /// The result of running the parser
  let ast: AbstractSyntaxTree

  /// A synthesized call to this program's unique main, or nil if none can be
  /// found.
  let entryPoint: FunctionCall<Expression>?

  /// A mapping from identifier to its definition.
  let definition: ASTDictionary<Identifier, Declaration>

  /// Constructs an instance for the given parser output, or throws `ErrorLog`
  /// if the program is ill-formed.
  init(_ program: AbstractSyntaxTree) throws {
    self.ast = program
    let r = NameResolution(program)
    if !r.errors.isEmpty { throw r.errors }
    var definitions = r.definition

    if let main = unambiguousMain(in: ast) {
      // Synthesize a call to main()
      let mainID = Identifier(text: "main", site: topColumns(1...4))
      let arguments = TupleLiteral([], topColumns(5...6))
      entryPoint = FunctionCall(callee: .name(mainID), arguments: arguments)
      // Make sure the identifier can be looked up.
      definitions[mainID] = main
    }
    else {
      entryPoint = nil
    }
    self.definition = definitions
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
  in parsedProgram: AbstractSyntaxTree) -> FunctionDefinition?
{
  // The nullary main functions defined at global scope
  var candidates = parsedProgram.compactMap { (x)->FunctionDefinition? in
    if case .function(let f) = x, f.name.text == "main", f.parameters.isEmpty
    { return f } else { return nil }
  }[...]

  guard let main = candidates.popFirst() else {
    return nil
  }
  assert(
    candidates.isEmpty,
    "Duplicate definitions should have been ruled out by name resolution.")
  return main
}
