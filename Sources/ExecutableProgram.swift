// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A program whose semantic analysis/compilation is complete, containing all
/// the information necessary for execution.
struct ExecutableProgram {
  /// The result of running the parser
  let ast: [Declaration]

  /// The entry point for this program.
  let main: FunctionDefinition

  /// A mapping from identifier to its declaration
  let declaration = PropertyMap<Identifier.Body, Declaration>()

  /// A mapping from (non-function) declarations to their addresses.
  let declarationAddress = PropertyMap<Declaration.Body, RelativeAddress>()

  /// A mapping from (non-function) expressions to their addresses.
  let expressionAddress = PropertyMap<Expression.Body, RelativeAddress>()

  /// An element of the frame layout
  typealias FrameElement = (type: Type, mutable: Bool, site: SourceRegion)
  
  /// A mapping from function definition to the list of types that are
  /// allocated into its frame.
  let frameLayout = PropertyMap<FunctionDefinition.Body, [FrameElement]>()

  /// Constructs an instance for the given parser output, or throws some sort of
  /// compilation error if the program is ill-formed.
  init(_ parsedProgram: [Declaration]) throws {
    self.ast = parsedProgram
    let mainCandidates: [FunctionDefinition] = parsedProgram.compactMap { d in
      if case .function(let f) = d.body,
         f.body.name.body == "main",
         f.body.parameterPattern.body.isEmpty { return f }
      return nil
    }

    guard let main0 = mainCandidates.first else {
      throw CompileError("No nullary main() found.", at: .empty)
    }

    if mainCandidates.count > 1 {
      throw CompileError(
        "Multiple main() candidates found.", at: main0.body.name.site,
        notes: mainCandidates.dropFirst().map {
          ("other candidate:", $0.body.name.site)
        })
    }
    self.main = main0
  }
}
