// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A syntactically valid program fragment in non-textual form, annotated with
/// its site in an input source file.
///
/// - Note: the value of an AST node is determined by its type and source
///   region.  Any other content is along for the ride, and assumed to be
///   uniquely identified by the node's value.
protocol AST: Hashable {
  typealias Site = SourceRegion
  /// The textual range of this fragment in the source.
  var site: Site { get }
}

extension AST {
  /// Returns `true` iff `l` and `r` are equivalent, i.e. have the same `content`
  /// value.
  static func == (l: Self, r: Self) -> Bool {
    l.site == r.site
  }

  /// Accumulates the hash value of `self` into `accumulator`.
  func hash(into accumulator: inout Hasher) {
    site.hash(into: &accumulator)
  }
}

/// An unqualified name.
struct Identifier: AST {
  let text: String
  let site: Site
}

/// A declaration, except for pattern variables, struct members, and function
/// parameters.
indirect enum TopLevelDeclaration: AST {
  case
    function(FunctionDefinition),
    `struct`(StructDefinition),
    choice(ChoiceDefinition),
    variable(VariableDefinition)

  var site: Site {
    switch self {
    case let .function(f): return f.site
    case let .struct(s): return s.site
    case let .choice(c): return c.site
    case let .variable(v): return v.site
    }
  }
}

enum VariableDefinition: AST {
  case
    uninitialized(Binding, Site),
    simple(Binding, initializer: Expression, Site),
    tuplePattern(TuplePattern, initializer: Expression, Site),
    recordPattern(RecordPattern, initializer: Expression, Site),
    functionTypePattern(FunctionTypePattern, initializer: Expression, Site)

  var site: Site {
    switch self {
    case let .uninitialized(_, r): return r
    case let .simple(_, initializer: _, r): return r
    case let .tuplePattern(_, initializer: _, r): return r
    case let .recordPattern(_, initializer: _, r): return r
    case let .functionTypePattern(_, initializer: _, r): return r
    }
  }
}

struct FunctionDefinition: AST {
  let name: Identifier
  let parameters: List<Binding>
  let returnType: Expression
  let body: Statement?
  let site: Site
}

typealias MemberDesignator = Identifier

struct Alternative: AST {
  let name: Identifier;
  let payload: List<Expression>
  let site: Site
}

struct StructDefinition: AST {
  let name: Identifier
  let members: [StructMemberDeclaration]
  let site: Site
}

struct ChoiceDefinition: AST {
  let name: Identifier
  let alternatives: [Alternative]
  let site: Site
}

indirect enum Statement: AST {
  case
    expressionStatement(Expression, Site),
    assignment(target: Expression, source: Expression, Site),
    variableDefinition(VariableDefinition),
    `if`(condition: Expression, thenClause: Statement, elseClause: Statement?, Site),
    `return`(Expression, Site),
    sequence(Statement, Statement, Site),
    block([Statement], Site),
    `while`(condition: Expression, body: Statement, Site),
    match(subject: Expression, clauses: [MatchClause], Site),
    `break`(Site),
    `continue`(Site)

  var site: Site {
    switch self {
    case let .expressionStatement(_, r): return r
    case let .assignment(target: _, source: _, r): return r
    case let .variableDefinition(v): return v.site
    case let .if(condition: _, thenClause: _, elseClause: _, r): return r
    case let .return(_, r): return r
    case let .sequence(_, _, r): return r
    case let .block(_, r): return r
    case let .while(condition: _, body: _, r): return r
    case let .match(subject: _, clauses: _, r): return r
    case let .break(r): return r
    case let .continue(r): return r
    }
  }
}

struct List<T>: AST {
  init(_ elements: [T], _ site: Site) {
    self.elements = elements
    self.site = site
  }

  let elements: [T]
  let site: Site
}

extension List: RandomAccessCollection {
  var startIndex: Int { 0 }
  var endIndex: Int { elements.count }
  subscript(i: Int) -> T { elements[i] }
}

struct MatchClause: AST {
  /// A `nil` `pattern` means this is a default clause.
  let pattern: Pattern?
  let action: Statement
  let site: Site
}
typealias MatchClauseList = List<MatchClause>
typealias TupleLiteral = List<Expression>
typealias RecordLiteral = List<(fieldName: Identifier, value: Expression)>

struct Binding: AST {
  let type: Expression
  let boundName: Identifier

  var site: Site { type.site...boundName.site }
}

enum TuplePatternElement {
  case binding(Binding)
  case literal(Expression)
}

typealias TuplePattern = List<TuplePatternElement>

struct FunctionTypePattern {
  let parameters: TuplePattern
  let returnType: TuplePatternElement
}

enum RecordPatternElement {
  case binding(fieldName: Identifier, type: Expression, boundName: Identifier)
  case literal(fieldName: Identifier, value: Expression)
}

typealias RecordPattern = List<RecordPatternElement>

indirect enum Expression: AST {
  case
    name(Identifier),
    getField(target: Expression, fieldName: Identifier, Site),
    index(target: Expression, offset: Expression, Site),
    patternVariable(type: Expression, name: Identifier, Site),
    integerLiteral(Int, Site),
    booleanLiteral(Bool, Site),
    tupleLiteral(List<Expression>),
    recordLiteral(RecordLiteral),
    unaryOperator(operation: Token, operand: Expression, Site),
    binaryOperator(operation: Token, lhs: Expression, rhs: Expression, Site),
    functionCall(callee: Expression, arguments: List<Expression>, Site),
    structInitialization(
      type: Expression, fieldInitializers: RecordLiteral, Site),
    intType(Site),
    boolType(Site),
    typeType(Site),
    autoType(Site),
    functionType(parameterTypes: List<Expression>, returnType: Expression, Site)

  var site: Site {
    switch self {
    case let .name(v): return v.site
    case let .getField(_, _, r): return r
    case let .index(target: _, offset: _, r): return r
    case let .patternVariable(type: _, name: _, r): return r
    case let .integerLiteral(_, r): return r
    case let .booleanLiteral(_, r): return r
    case let .tupleLiteral(t): return t.site
    case let .recordLiteral(l): return l.site
    case let .unaryOperator(operation: _, operand: _, r): return r
    case let .binaryOperator(operation: _, lhs: _, rhs: _, r): return r
    case let .functionCall(callee: _, arguments: _, r): return r
    case let .structInitialization(type: _, fieldInitializers: _, r): return r
    case let .intType(r): return r
    case let .boolType(r): return r
    case let .typeType(r): return r
    case let .autoType(r): return r
    case let .functionType(parameterTypes: _, returnType: _, r):
      return r
    }
  }
};

enum Pattern {
  case
    literal(Expression),
    unary(Binding),
    tuple(TuplePattern),
    record(RecordPattern),
    alternative(identity: Expression, payload: TuplePattern),
    functionType(FunctionTypePattern)
}

struct StructMemberDeclaration: AST {
  let name: Identifier
  let type: Expression
  let site: Site
}

/// Unifies all declarations
///
/// This doesn't actually appear in the AST, but is used by the typechecker and
/// interpreter as the value type of a name-use -> declaration dictionary.
enum AnyDeclaration: AST { 
  case
    function(FunctionDefinition),
    `struct`(StructDefinition),
    choice(ChoiceDefinition),
    variable(VariableDefinition),
    // Function parameters, variable declarations...
    binding(Binding),
    structMember(StructMemberDeclaration),
    alternative(Alternative)

  init(_ x: TopLevelDeclaration) {
    switch x {
    case let .function(f): self = .function(f)
    case let .struct(s): self = .struct(s)
    case let .choice(c): self = .choice(c)
    case let .variable(v): self = .variable(v)
    }
  }
  
  var site: SourceRegion {
    switch self {
    case let .function(f): return f.site
    case let .struct(s): return s.site
    case let .choice(c): return c.site
    case let .variable(v): return v.site
    case .binding(let p): return p.site
    case .structMember(let m): return m.site
    case .alternative(let a): return a.site
    }
  }
}

