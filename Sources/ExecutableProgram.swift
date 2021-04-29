// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A program whose semantic analysis/compilation is complete, containing all
/// the information necessary for execution.
struct ExecutableProgram {
  /// The result of running the parser
  let ast: AbstractSyntaxTree

  /// The entry point for this program.
  let main: FunctionDefinition

  /// A mapping from identifier to its definition.
  let definition: ASTDictionary<Identifier, Declaration>

  /// Constructs an instance for the given parser output, or throws `ErrorLog`
  /// if the program is ill-formed.
  init(_ program: AbstractSyntaxTree) throws {
    self.ast = program
    let r = NameResolution(program)
    definition = r.definition

    if !r.errors.isEmpty { throw r.errors }
    switch unambiguousMain(in: ast) {
    case let .failure(e): throw [e]
    case let .success(main): self.main = main
    }
  }
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
