struct AST<Node: Hashable>: Hashable {
  init(_ body: Node, _ location: SourceLocation) {
    self.body = body
    self.location = location
  }
  
  var body: Node
  var location: SourceLocation
}

typealias Identifier = AST<Token>

indirect enum Declaration_: Hashable {
  case
    function(FunctionDefinition),
    `struct`(StructDefinition),
    choice(name: Identifier, alternatives: [Alternative]),
    variable(name: Identifier, type: Expression, initializer: Expression)
}
typealias Declaration = AST<Declaration_>

struct FunctionDefinition_: Hashable {
  var name: Identifier
  var parameterPattern: TupleLiteral
  var returnType: Expression
  var body: Statement?
}
typealias FunctionDefinition = AST<FunctionDefinition_>

typealias MemberDesignator = Identifier

struct Alternative_: Hashable {
  var name: Identifier;
  var payload: TupleLiteral
}

typealias Alternative = AST<Alternative_>

struct StructDefinition: Hashable {
  var name: Identifier
  var members: [VariableDeclaration]
}

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
typealias Statement = AST<Statement_>

struct MatchClause_: Hashable {
  /// A `nil` `pattern` means this is a default clause.
  var pattern: Expression?
  var action: Statement
}

typealias MatchClause = AST<MatchClause_>
typealias MatchClauseList = AST<[MatchClause]>

struct TupleLiteralElement: Hashable {
  var name: Identifier?
  var value: Expression
}
typealias TupleLiteral_ = [TupleLiteralElement]
typealias TupleLiteral = AST<TupleLiteral_>

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
typealias Expression = AST<Expression_>

struct Field_: Hashable {
  var first: Identifier?
  var second: Expression
}

typealias Field = AST<Field_>
struct FieldList_: Hashable {
  var fields: [Field]
  var hasExplicitComma: Bool
}
typealias FieldList = AST<FieldList_>

struct VariableDeclaration_: Hashable {
  var name: Identifier
  var type: Expression  
}
typealias VariableDeclaration = AST<VariableDeclaration_>
