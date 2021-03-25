public typealias AST<Node> = (body: Node, location: SourceLocation)
public typealias Identifier = AST<Token>

public indirect enum Declaration_ {
  case
    function(FunctionDefinition),
    `struct`(StructDefinition),
    choice(name: Identifier, [(Identifier, Expression)]),
    variable(name: Identifier, type: Expression, initializer: Expression)
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
public enum Member { case name(Identifier), type(Expression) }

public struct StructDefinition {
  public var name: Identifier
  public var members: [Member]
}

public indirect enum Statement_ {
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
public typealias Statement = AST<Statement_>

/// A `nil` `pattern` means this is a default clause.
public typealias MatchClause = AST<(pattern: Expression?, action: Statement)>
public typealias MatchClauseList = AST<[MatchClause]>
public typealias TupleLiteral_ = [(name: Identifier?, value: Expression)]
public typealias TupleLiteral = AST<TupleLiteral_>

public indirect enum Expression_ {
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
public typealias Expression = AST<Expression_>

public typealias Field = AST<(Identifier?, Expression)>
public typealias FieldList = AST<(fields: [Field], hasExplicitComma: Bool)>
