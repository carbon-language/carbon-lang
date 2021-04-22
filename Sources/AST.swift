// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A syntactically valid program fragment in non-textual form, annotated with
/// its site in an input source file.
///
/// - Note: the value of an AST node is determined by its structure.  The site
///   of the node in source is incidental/non-salient information.
protocol AST: Hashable {
  /// A value-free representation of `self`'s source region
  typealias Site = ASTSite

  /// A value-ful representation of `self`'s source region, carrying `self` as
  /// non-salient data for debugging purposes.
  typealias Identity = ASTIdentity<Self>

  /// The textual range of this fragment in the source.
  var site: Site { get }
}

extension AST {
  /// An identifier of this node by its source location.
  var identity: Identity { Identity(of: self) }
}

/// An unqualified name.
struct Identifier: AST {
  let text: String
  let site: Site
}

/// A declaration that can appear at file scope.
enum TopLevelDeclaration: AST {
  case
    function(FunctionDefinition),
    `struct`(StructDefinition),
    choice(ChoiceDefinition),
    initialization(Initialization)

  var site: Site {
    switch self {
    case let .function(f): return f.site
    case let .struct(s): return s.site
    case let .choice(c): return c.site
    case let .initialization(v): return v.site
    }
  }
}

/// A destructurable pattern.
indirect enum Pattern: AST {
  case
    atom(Expression),             // A non-destructurable expression
    variable(SimpleBinding),     // <Type>: <name>
    tuple(TuplePattern),
    functionCall(FunctionCall<PatternElement>),
    functionType(parameterTypes: TuplePattern, returnType: Pattern, Site)

  init(_ e: Expression) {
    // Upcast all destructurable things into appropriate pattern buckets
    switch e {
    case let .tupleLiteral(t):
      self = .tuple(TuplePattern(t))
    case let .functionCall(f):
      self = .functionCall(.init(f))
    case let .functionType(parameterTypes: p, returnType: r, site):
      self = .functionType(
        parameterTypes: TuplePattern(p), returnType: Pattern(r), site)
    default: self = .atom(e)
    }
  }

  var site: Site {
    switch self {
    case let .atom(x): return x.site
    case let .variable(x): return x.site
    case let .tuple(x): return x.site
    case let .functionCall(x): return x.site
    case let .functionType(parameterTypes: _, returnType: _, r): return r
    }
  }
}

struct TypeSpecifier: AST {
  let type: Expression
  init(_ type: Expression) { self.type = type }

  var site: Site { type.site }
}
enum LHSTypeSpecifier: AST {
  case
    auto(Site),
    literal(Expression)

  init(_ e: Expression) { self = .literal(e) }
  
  var site: Site {
    switch self {
    case let .auto(r): return r
    case let .literal(x): return x.site
    }
  }
}

struct SimpleBinding: AST {
  let type: LHSTypeSpecifier
  let boundName: Identifier
  var site: Site { type.site...boundName.site }
}

struct FunctionCall<Argument: Hashable>: AST {
  let callee: Expression
  let arguments: Tuple<Argument>

  var site: Site { return callee.site...arguments.site }
}

extension FunctionCall where Argument == PatternElement {
  /// "Upcast" from literal to pattern
  init(_ literal: FunctionCall<LiteralElement>) {
    self.init(callee: literal.callee, arguments: .init(literal.arguments))
  }
}

struct LiteralElement: Hashable {
  init(label: Identifier? = nil, _ value: Expression) {
    self.label = label
    self.value = value
  }
  let label: Identifier?
  let value: Expression
}
typealias TupleLiteral = Tuple<LiteralElement>

struct PatternElement: Hashable {
  init(label: Identifier? = nil, _ value: Pattern) {
    self.label = label
    self.value = value
  }
  // "Upcast" from literal element
  init(_ l: LiteralElement) {
    self.init(label: l.label, Pattern(l.value))
  }
  
  let label: Identifier?
  let value: Pattern
}

typealias TuplePattern = Tuple<PatternElement>
extension TuplePattern {
  // "Upcast" from tuple literal.
  init(_ l: TupleLiteral) {
    self.init(l.elements.map { PatternElement($0) }, l.site)
  }
}

struct FunctionDefinition: AST {
  let name: Identifier
  let parameters: TuplePattern
  let returnType: Expression
  let body: Statement?
  let site: Site
}

typealias MemberDesignator = Identifier

struct Alternative: AST {
  let name: Identifier;
  let payload: TupleLiteral
  let site: Site
}

struct StructDefinition: AST {
  let name: Identifier
  let members: [StructMember]
  let site: Site
}

struct ChoiceDefinition: AST {
  let name: Identifier
  let alternatives: [Alternative]
  let site: Site
}

struct StructMember: AST {
  let type: TypeSpecifier
  let name: Identifier
  let site: Site
}

struct Initialization: AST {
  let bindings: Pattern
  let initializer: Expression
  let site: Site
}

indirect enum Statement: AST {
  case
    expressionStatement(Expression, Site),
    assignment(target: Expression, source: Expression, Site),
    initialization(Initialization),
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
    case let .initialization(v): return v.site
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

struct Tuple<T: Hashable>: AST {
  init(_ elements: [T], _ site: Site) {
    self.elements = elements
    self.site = site
  }

  let elements: [T]
  let site: Site
}

extension Tuple: RandomAccessCollection {
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
typealias MatchClauseList = [MatchClause]

struct FunctionType<Parameter: Hashable, Return: Hashable>: AST {
  let parameters: Tuple<Parameter>
  let returnType: Return
  let site: Site
}
extension FunctionType where Parameter == PatternElement, Return == Pattern {
  /// "Upcast" from literal to pattern
  init(_ literal: FunctionType<LiteralElement, Expression>) {
    self.init(
      parameters: .init(literal.parameters),
      returnType: .init(literal.returnType),
      site: literal.site)
  }
}


indirect enum Expression: AST {
  case
    name(Identifier),
    getField(target: Expression, fieldName: Identifier, Site),
    index(target: Expression, offset: Expression, Site),
    patternVariable(type: Expression, name: Identifier, Site),
    integerLiteral(Int, Site),
    booleanLiteral(Bool, Site),
    tupleLiteral(TupleLiteral),
    unaryOperator(operation: Token, operand: Expression, Site),
    binaryOperator(operation: Token, lhs: Expression, rhs: Expression, Site),
    functionCall(FunctionCall<LiteralElement>),
    intType(Site),
    boolType(Site),
    typeType(Site),
    autoType(Site),
    functionType(parameterTypes: TupleLiteral, returnType: Expression, Site)

  var site: Site {
    switch self {
    case let .name(v): return v.site
    case let .getField(_, _, r): return r
    case let .index(target: _, offset: _, r): return r
    case let .patternVariable(type: _, name: _, r): return r
    case let .integerLiteral(_, r): return r
    case let .booleanLiteral(_, r): return r
    case let .tupleLiteral(t): return t.site
    case let .unaryOperator(operation: _, operand: _, r): return r
    case let .binaryOperator(operation: _, lhs: _, rhs: _, r): return r
    case let .functionCall(f): return f.site
    case let .intType(r): return r
    case let .boolType(r): return r
    case let .typeType(r): return r
    case let .autoType(r): return r
    case let .functionType(parameterTypes: _, returnType: _, r):
      return r
    }
  }
};

struct StructMemberDeclaration: AST {
  let name: Identifier
  let type: TypeSpecifier
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
    structMember(StructMember),
    initialization(Initialization),
    alternative(Alternative)

  init(_ x: TopLevelDeclaration) {
    switch x {
    case let .function(f): self = .function(f)
    case let .struct(s): self = .struct(s)
    case let .choice(c): self = .choice(c)
    case let .initialization(v): self = .initialization(v)
    }
  }
  
  var site: Site {
    switch self {
    case let .function(f): return f.site
    case let .struct(s): return s.site
    case let .choice(c): return c.site
    case let .structMember(v): return v.site
    case let .initialization(v): return v.site
    case let .alternative(a): return a.site
    }
  }
}

/// An annotation that indicates the SourceRegion of an AST node.
///
/// Instances of ASTSite always compare ==, allowing us to include location
/// information in the AST while still letting the compiler synthesize node
/// equality based on structure.
struct ASTSite: Hashable {
  /// Creates an instance storing `r` without making it part of the value of
  /// `self`.
  init(devaluing r: SourceRegion) { self.region = r }

  let region: SourceRegion

  static func == (_: Self, _: Self) -> Bool { true }
  func hash(into _: inout Hasher) {}

  static var empty: ASTSite { ASTSite(devaluing: SourceRegion.empty) }

  /// Returns the site from the beginning of `first` to the end of `last`,
  /// unless one of `first` or `last` is empty, in which case the other one is
  /// returned.
  ///
  /// - Requires first or last is empty, or `site.fileName ==
  ///   last.fileName && first.span.lowerBound < last.span.upperBound`.
  static func ... (first: Self, last: Self) -> Self {
    .init(devaluing: first.region...last.region)
  }
}

/// The identity of an AST node in a program's source, based on its region.
///
/// NodeIdentity additionally carries the structure of the AST node as
/// auxilliary information for debugging purposes.  Otherwise it is equivalent
/// to SourceRegion.
struct ASTIdentity<Node: AST>: Hashable {
  fileprivate init(of n: Node) {
    self.structure = n
  }
  let structure: Node

  static func == (l: Self, r: Self) -> Bool {
    l.structure.site.region == r.structure.site.region
  }

  func hash(into h: inout Hasher) {
    structure.site.region.hash(into: &h)
  }
}
