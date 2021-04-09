// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A syntactically valid program fragment in non-textual form, annotated with
/// its site in an input source file.
///
/// - Note: the source site is *incidental* information that is **not
/// considered part of the AST's value.** In other words, two ASTs whose
/// contents differ only by source sites will compare as equal.
struct AST<Body: Hashable>: Hashable {
  init(_ body: Body, _ site: SourceRegion) {
    self.body = body
    self.site = site
  }

  /// Returns `true` iff `l` and `r` are equivalent, i.e. have the same `body`
  /// value.
  static func == (l: Self, r: Self) -> Bool { l.body == r.body }

  /// Accumulates the hash value of `self` into `accumulator`.
  func hash(into accumulator: inout Hasher) { body.hash(into: &accumulator) }

  /// The content of this fragment.
  var body: Body

  /// The textual range of this fragment in the source.
  var site: SourceRegion
}

/// An unqualified name.
typealias Identifier = AST<String>

typealias Declaration = AST<Declaration_>
indirect enum Declaration_: Hashable {
  case
    function(FunctionDefinition),
    `struct`(StructDefinition),
    choice(ChoiceDefinition),
    variable(name: Identifier, type: Expression, initializer: Expression)
}

typealias FunctionDefinition = AST<FunctionDefinition_>
struct FunctionDefinition_: Hashable {
  var name: Identifier
  var parameterPattern: TupleLiteral
  var returnType: Expression
  var body: Statement?
}

typealias MemberDesignator = Identifier

typealias Alternative = AST<Alternative_>
struct Alternative_: Hashable {
  var name: Identifier;
  var payload: TupleLiteral
}

struct StructDefinition: Hashable {
  var name: Identifier
  var members: [VariableDeclaration]
}

struct ChoiceDefinition: Hashable {
  var name: Identifier
  var alternatives: [Alternative]
}


typealias Statement = AST<Statement_>
indirect enum Statement_: Hashable {
  case
    expressionStatement(Expression),
    assignment(target: Expression, source: Expression),
    variableDefinition(pattern: Expression, initializer: Expression),
    `if`(condition: Expression, thenClause: Statement, elseClause: Statement?),
    `return`(Expression),
    sequence(Statement, Statement),
    block([Statement]),
    `while`(condition: Expression, body: Statement),
    match(subject: Expression, clauses: [MatchClause]),
    `break`,
    `continue`
}

typealias MatchClauseList = AST<[MatchClause]>
typealias MatchClause = AST<MatchClause_>
struct MatchClause_: Hashable {
  /// A `nil` `pattern` means this is a default clause.
  var pattern: Expression?
  var action: Statement
}


typealias TupleLiteral = AST<TupleLiteral_>
typealias TupleLiteral_ = [TupleLiteralElement]
struct TupleLiteralElement: Hashable {
  var name: Identifier?
  var value: Expression
}


typealias Expression = AST<Expression_>
indirect enum Expression_: Hashable {
  case
    variable(Identifier),
    getField(target: Expression, fieldName: Identifier),
    index(target: Expression, offset: Expression),
    patternVariable(name: Identifier, type: Expression),
    integerLiteral(Int),
    booleanLiteral(Bool),
    tupleLiteral(TupleLiteral_),
    unaryOperator(operation: Token, operand: Expression),
    binaryOperator(operation: Token, lhs: Expression, rhs: Expression),
    functionCall(callee: Expression, arguments: TupleLiteral),
    intType,
    boolType,
    typeType,
    autoType,
    functionType(parameterTypes: TupleLiteral, returnType: Expression)
};

typealias Field = AST<Field_>
struct Field_: Hashable {
  var first: Identifier?
  var second: Expression
}

typealias FieldList = AST<FieldList_>
struct FieldList_: Hashable {
  var fields: [Field]
  var hasExplicitComma: Bool
}

typealias VariableDeclaration = AST<VariableDeclaration_>
struct VariableDeclaration_: Hashable {
  var name: Identifier
  var type: Expression
}
