public indirect enum Declaration {
  case
    function(FunctionDefinition),
    `struct`(StructDefinition),
    choice(name: String, [(String, Expression)]),
    variable(name: String, type: TypeExpression, initializer: Expression)
}

public struct FunctionDefinition {
  public var name: String
  public var parameterPattern: TupleLiteral
  public var returnType: Expression
  public var body: Statement
}

typealias MemberDesignator = String

typealias Alternative (name: String, payload: TupleLiteral)
public enum Member { case name(String), type(TypeExpression) }

public struct StructDefinition {
  public var name: String
  public var members: [Member]
}

public indirect enum Statement {
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

public enum TypeExpression {
  case
    int,
    bool,
    typetype,
    auto,
    function(parameterTypes: TupleLiteral, returnType: Expression)
}

public typealias MatchClause = (pattern: Expression, action: Statement)
public typealias TupleLiteral = [(name: String?, value: Expression)]

public indirect enum Expression {
  case
    variable(String),
    getField(target: Expression, fieldName: String),
    index(target: Expression, offset: Expression),
    patternVariable(name: String, type: TypeExpression),
    integerLiteral(Int),
    booleanLiteral(Bool),
    tupleLiteral(TupleLiteral),
    unaryOperator(operation: Token, operand: Expression),
    binaryOperator(operation: Token, lhs: Expression, rhs: Expression),
    functionCall(callee: Expression, arguments: TupleLiteral),
    type(TypeExpression)
};
