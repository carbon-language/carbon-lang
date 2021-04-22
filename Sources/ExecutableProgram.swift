// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A program whose semantic analysis/compilation is complete, containing all
/// the information necessary for execution.
struct ExecutableProgram {
  /// The result of running the parser
  let ast: [TopLevelDeclaration]

  /// The entry point for this program.
  let main: FunctionDefinition

  /// A mapping from identifier to its declaration
  var //let
    declaration = ASTDictionary<Identifier, TopLevelDeclaration>()

  /// Constructs an instance for the given parser output, or throws `ErrorLog`
  /// if the program is ill-formed.
  // should be fileprivate - internal for testing purposes.
  init(_parsedProgram ast: [TopLevelDeclaration]) throws {
    self.ast = ast
    switch unambiguousMain(in: ast) {
    case let .failure(e): throw [e]
    case let .success(main): self.main = main
    }
  }
}

/// Returns the unique top-level nullary main() function defined in
/// `parsedProgram`, or reports a suitable CompileError if that doesn't exist.
fileprivate func unambiguousMain(
  in parsedProgram: [TopLevelDeclaration]
) -> Result<FunctionDefinition, CompileError> {
  let mainCandidates: [FunctionDefinition] = parsedProgram.compactMap {
    if case .function(let f) = $0,
       f.name.text == "main",
       f.parameters.isEmpty
    { return f } else { return nil }
  }

  guard let r = mainCandidates.first else {
    return .failure(CompileError("No nullary main() found.", at: .empty))
  }

  if mainCandidates.count > 1 {
    return .failure(
      CompileError(
        "Multiple main() candidates found.", at: r.name.site,
        notes: mainCandidates.dropFirst().map { ("candidate", $0.name.site) }))
  }

  return .success(r)
}
