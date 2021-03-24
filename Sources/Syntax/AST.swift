public typealias AST<Node> = (Node, location: SourceLocation)
public typealias Identifier = AST<String>

public indirect enum Declaration_ {
  case
    function(FunctionDefinition),
    `struct`(StructDefinition),
    choice(name: Identifier, [(Identifier, Expression)]),
    variable(name: Identifier, type: TypeExpression, initializer: Expression)
}
public typealias Declaration = AST<Declaration_>

public struct FunctionDefinition {
  public var name: Identifier
  public var parameterPattern: TupleLiteral
  public var returnType: Expression
  public var body: Statement
}

typealias MemberDesignator = Identifier

typealias Alternative = AST<(name: Identifier, payload: TupleLiteral)>
public enum Member { case name(Identifier), type(TypeExpression) }

public struct StructDefinition {
  public var name: Identifier
  public var members: [Member]
}

public indirect enum Statement_ {
  case
    expressionStatement(Expression),
    assignment(target: Expression, source: Expression),
    variableDefinition(pattern: Expression, initializer: Expression),
    `if`(condition: Expression, thenClause: Statement, elseClause: Statement),
    `return`(Expression),
    sequence(Statement, Statement),
    block(Statement),
    `while`(condition: Expression, body: Statement),
    match(clauses: [MatchClause])
}
public typealias Statement = AST<Statement_>

public enum TypeExpression_ {
  case
    int,
    bool,
    typetype,
    auto,
    function(parameterTypes: TupleLiteral, returnType: Expression)
}
public typealias TypeExpression = AST<TypeExpression_>

public typealias MatchClause = AST<(pattern: Expression, action: Statement)>
public typealias TupleLiteral = AST<[(name: Identifier?, value: Expression)]>

public indirect enum Expression_ {
  case
    variable(Identifier),
    getField(target: Expression, fieldName: Identifier),
    index(target: Expression, offset: Expression),
    patternVariable(name: Identifier, type: TypeExpression),
    integerLiteral(Int),
    booleanLiteral(Bool),
    tupleLiteral(TupleLiteral),
    unaryOperator(operation: Token, operand: Expression),
    binaryOperator(operation: Token, lhs: Expression, rhs: Expression),
    functionCall(callee: Expression, arguments: TupleLiteral),
    type(TypeExpression)
};
public typealias Expression = AST<Expression_>
