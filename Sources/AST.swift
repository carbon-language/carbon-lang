typealias AST<Node> = (body: Node, location: SourceLocation)
typealias Identifier = AST<Token>

indirect enum Declaration_ {
  case
    function(FunctionDefinition),
    `struct`(StructDefinition),
    choice(name: Identifier, alternatives: [Alternative]),
    variable(name: Identifier, type: Expression, initializer: Expression)
}
typealias Declaration = AST<Declaration_>

struct FunctionDefinition_ {
  var name: Identifier
  var parameterPattern: TupleLiteral
  var returnType: Expression
  var body: Statement?
}
typealias FunctionDefinition = AST<FunctionDefinition_>

typealias MemberDesignator = Identifier

typealias Alternative = AST<(name: Identifier, payload: TupleLiteral)>

struct StructDefinition {
  var name: Identifier
  var members: [VariableDeclaration]
}

indirect enum Statement_ {
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
typealias Statement = AST<Statement_>

/// A `nil` `pattern` means this is a default clause.
typealias MatchClause = AST<(pattern: Expression?, action: Statement)>
typealias MatchClauseList = AST<[MatchClause]>
typealias TupleLiteral_ = [(name: Identifier?, value: Expression)]
typealias TupleLiteral = AST<TupleLiteral_>

indirect enum Expression_ {
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
typealias Expression = AST<Expression_>

typealias Field = AST<(Identifier?, Expression)>
typealias FieldList = AST<(fields: [Field], hasExplicitComma: Bool)>
typealias VariableDeclaration = AST<(name: Identifier, type: Expression)>
