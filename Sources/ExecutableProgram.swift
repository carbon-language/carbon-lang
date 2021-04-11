// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

typealias FrameElement = (type: Type, mutable: Bool, site: SourceRegion)

/// A program whose semantic analysis/compilation is complete, containing all
/// the information necessary for execution.
struct ExecutableProgram {
  /// The result of running the parser
  let ast: [Declaration]

  /// The entry point for this program.
  let main: FunctionDefinition

  /// A mapping from identifier to its declaration
  let declaration: PropertyMap<Identifier.Body, Declaration>

  /// A mapping from (non-function) declarations to their addresses.
  let declarationAddress: PropertyMap<Declaration.Body, RelativeAddress>

  /// A mapping from (non-function) expressions to their addresses.
  let expressionAddress: PropertyMap<Expression.Body, RelativeAddress>
  
  /// A mapping from function definition to the list of types that are
  /// allocated into its frame.
  let frameLayout: PropertyMap<FunctionDefinition.Body, [FrameElement]>
}
