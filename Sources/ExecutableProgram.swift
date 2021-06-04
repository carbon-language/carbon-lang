// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A program whose semantic analysis/compilation is complete, containing all
/// the information necessary for execution.
struct ExecutableProgram {
  /// The result of running the parser
  let ast: AbstractSyntaxTree

  /// A mapping from identifier to its definition.
  let definition: ASTDictionary<Identifier, Declaration>

  /// Mapping from expression to the static type of that expression.
  let staticType: ASTDictionary<Expression, Type>

  /// The payload tuple type for each alternative.
  let payloadType: [ASTIdentity<Alternative>: TupleType]

  /// Mapping from alternative declaration to the choice in which it is defined.
  let enclosingChoice: ASTDictionary<Alternative, ChoiceDefinition>

  /// The type of the expression consisting of the name of each declared entity.
  let typeOfNameDeclaredBy: Dictionary<Declaration.Identity, Memo<Type>>

  /// The unique top-level nullary main() function defined in `ast`,
  /// or `nil` if that doesn't exist.
  var main: FunctionDefinition? {
    // The nullary main functions defined at global scope
    let candidates = ast.compactMap { (x)->FunctionDefinition? in
      if case .function(let f) = x, f.name.text == "main", f.parameters.isEmpty
      { return f } else { return nil }
    }
    if candidates.isEmpty { return nil }

    assert(
      candidates.count == 1,
      "Duplicate definitions should have been ruled out by name resolution.")
    return candidates[0]
  }

  /// Constructs an instance for the given parser output, or throws `ErrorLog`
  /// if the program is ill-formed.
  init(_ parsedProgram: AbstractSyntaxTree) throws {
    self.ast = parsedProgram
    let nameLookup = NameResolution(ast)
    if !nameLookup.errors.isEmpty { throw nameLookup.errors }
    self.definition = nameLookup.definition
    let typeChecking = TypeChecker(parsedProgram, nameLookup: nameLookup)
    if !typeChecking.errors.isEmpty { throw typeChecking.errors }
    staticType = typeChecking.expressionType
    payloadType = typeChecking.payloadType
    enclosingChoice = typeChecking.enclosingChoice
    typeOfNameDeclaredBy = typeChecking.typeOfNameDeclaredBy
  }
}
