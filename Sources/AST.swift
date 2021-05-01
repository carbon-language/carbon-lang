// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Types representing syntactically valid program fragments in non-textual
/// form.
///
/// - Note: the value of an AST node is determined by its structure.  The site
///   of the node in source is an incidental/non-salient annotation that does
///   not contribute to its value.
protocol AST: Equatable {
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

/// The whole abstractsyntax tree is just a bunch of top-level declarations.
typealias AbstractSyntaxTree = [TopLevelDeclaration]

/// A destructurable pattern.
indirect enum Pattern: AST {
  case
    atom(Expression),             // A non-destructurable expression
    variable(SimpleBinding),     // <Type>: <name>
    tuple(TuplePattern),
    functionCall(FunctionCall<Pattern>),
    functionType(FunctionTypePattern)

  init(_ e: Expression) {
    // Upcast all destructurable things into appropriate pattern buckets
    switch e {
    case let .tupleLiteral(t): self = .tuple(TuplePattern(t))
    case let .functionCall(f): self = .functionCall(.init(f))
    case let .functionType(f): self = .functionType(.init(f))
    default: self = .atom(e)
    }
  }

  var site: Site {
    switch self {
    case let .atom(x): return x.site
    case let .variable(x): return x.site
    case let .tuple(x): return x.site
    case let .functionCall(x): return x.site
    case let .functionType(x): return x.site
    }
  }
}

/// Either a literal type expression, or `.auto(site)` which denotes a type that
/// should be deduced from another expression in an initialization or pattern
/// match.
enum TypeSpecifier: AST {
  case
    auto(Site),
    literal(TypeExpression)

  init(_ e: TypeExpression) { self = .literal(e) }
  
  var site: Site {
    switch self {
    case let .auto(r): return r
    case let .literal(x): return x.site
    }
  }
}

struct SimpleBinding: AST, Declaration {
  let type: TypeSpecifier
  let name: Identifier
  var site: Site { type.site...name.site }
}

struct FunctionCall<Argument: AST>: AST {
  let callee: Expression
  let arguments: Tuple<Argument>

  var site: Site { return callee.site...arguments.site }
}

extension FunctionCall where Argument == Pattern {
  /// "Upcast" from literal to pattern
  init(_ literal: FunctionCall<Expression>) {
    self.init(callee: literal.callee, arguments: .init(literal.arguments))
  }
}

typealias TupleLiteral = Tuple<Expression>
typealias TuplePattern = Tuple<Pattern>

extension TuplePattern {
  // "Upcast" from tuple literal.
  init(_ l: TupleLiteral) {
    self.init(l.elements.map { PatternElement($0) }, l.site)
  }

  // "Upcast" from tuple literal.
  init(_ l: TypeTuple) {
    self.init(l.elements.map { PatternElement($0) }, l.site)
  }
}

struct FunctionDefinition: AST, Declaration {
  let name: Identifier
  let parameters: TuplePattern
  let returnType: TypeSpecifier
  let body: Statement?
  let site: Site
}

typealias MemberDesignator = Identifier

struct Alternative: AST, Declaration {
  let name: Identifier;
  let payload: TupleLiteral
  let site: Site
}

struct StructDefinition: AST, TypeDeclaration {
  let name: Identifier
  let members: [StructMember]
  let site: Site

  var declaredType: Type { .struct(self) }
}

struct ChoiceDefinition: AST, TypeDeclaration {
  let name: Identifier
  let alternatives: [Alternative]
  let site: Site

  var declaredType: Type { .choice(self) }
}

struct StructMember: AST, Declaration {
  let type: TypeExpression
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
    case let .block(_, r): return r
    case let .while(condition: _, body: _, r): return r
    case let .match(subject: _, clauses: _, r): return r
    case let .break(r): return r
    case let .continue(r): return r
    }
  }
}

struct Tuple<Payload: AST>: AST {
  struct Element: AST {
    init(label: Identifier? = nil, _ payload: Payload) {
      self.label = label
      self.payload = payload
    }
    let label: Identifier?
    let payload: Payload

    var site: Site { label.map { $0.site...payload.site } ?? payload.site }
  }

  init(_ elements: [Element], _ site: Site) {
    self.elements = elements
    self.site = site
  }

  let elements: [Element]
  let site: Site
}
typealias LiteralElement = Tuple<Expression>.Element
typealias PatternElement = Tuple<Pattern>.Element

typealias TypeTuple = Tuple<TypeExpression>

extension PatternElement {
  // "Upcast" from literal element
  init(_ l: LiteralElement) {
    self.init(label: l.label, Pattern(l.payload))
  }

  // "Upcast" from literal element
  init(_ l: Tuple<TypeExpression>.Element) {
    self.init(label: l.label, Pattern(l.payload.body))
  }
}

extension Tuple: RandomAccessCollection {
  var startIndex: Int { 0 }
  var endIndex: Int { elements.count }
  subscript(i: Int) -> Element { elements[i] }
}

struct MatchClause: AST {
  /// A `nil` `pattern` means this is a default clause.
  let pattern: Pattern?
  let action: Statement
  let site: Site
}
typealias MatchClauseList = [MatchClause]

struct FunctionType<ParameterType: AST>: AST {
  let parameters: Tuple<ParameterType>
  let returnType: ParameterType
  let site: Site
}
typealias FunctionTypePattern = FunctionType<Pattern>
typealias FunctionTypeLiteral = FunctionType<Expression>

extension FunctionTypePattern {
  /// "Upcast" from literal to pattern
  init(_ source: FunctionTypeLiteral) {
    self.init(
      parameters: .init(source.parameters),
      returnType: .init(source.returnType),
      site: source.site)
  }
}

indirect enum Expression: AST {
  case
    name(Identifier),
    getField(target: Expression, fieldName: Identifier, Site),
    index(target: Expression, offset: Expression, Site),
    integerLiteral(Int, Site),
    booleanLiteral(Bool, Site),
    tupleLiteral(TupleLiteral),
    unaryOperator(operation: Token, operand: Expression, Site),
    binaryOperator(operation: Token, lhs: Expression, rhs: Expression, Site),
    functionCall(FunctionCall<Expression>),
    intType(Site),
    boolType(Site),
    typeType(Site),
    functionType(FunctionTypeLiteral)

  var site: Site {
    switch self {
    case let .name(v): return v.site
    case let .getField(_, _, r): return r
    case let .index(target: _, offset: _, r): return r
    case let .integerLiteral(_, r): return r
    case let .booleanLiteral(_, r): return r
    case let .tupleLiteral(t): return t.site
    case let .unaryOperator(operation: _, operand: _, r): return r
    case let .binaryOperator(operation: _, lhs: _, rhs: _, r): return r
    case let .functionCall(f): return f.site
    case let .intType(r): return r
    case let .boolType(r): return r
    case let .typeType(r): return r
    case let .functionType(t): return t.site
    }
  }
};

/// An expression whose value will be used as a type in type-checking.
///
/// You can think of this as an un-canonicalized `Type` value.  An instance is
/// created by the parser for all expressions in a syntactic position that
/// indicates a type.
///
/// TypeExpression(s) must be evaluated at compile-time.
struct TypeExpression: AST {
  /// Creates an instance containing `body`.
  init(_ body: Expression) {
    self.body = body
  }

  /// The computation that produces the type value.
  let body: Expression

  var site: Site { body.site }
}

/// The declaration of a name.
protocol Declaration {
  /// A type that can be used to uniquely identify declarations in the source.
  typealias Identity = AnyASTIdentity

  /// A unique identifier of this declaration, not based on its structure.
  var identity: Identity { get }

  /// The name being declared.
  var name: Identifier { get }

  /// The region of source covered by this declaration.
  var site: ASTSite { get }
}

extension Declaration where Self: AST {
  var identity: AnyASTIdentity { AnyASTIdentity(of: self) }
}

protocol TypeDeclaration: Declaration {
  /// Returns the type value created by this declaration
  var declaredType: Type { get }
}

/// An annotation that indicates the SourceRegion of an AST node.
///
/// Instances of ASTSite always compare ==, allowing us to include location
/// information in the AST while still letting the compiler synthesize node
/// equality based on structure.
struct ASTSite: Equatable {
  /// Creates an instance storing `r` without making it part of the value of
  /// `self`.
  init(devaluing r: SourceRegion) { self.region = r }

  let region: SourceRegion

  static func == (_: Self, _: Self) -> Bool { true }

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

/// The identity, in a program's source, of any `AST` node.
///
/// A useful dictionary key for mappings from the identities of heterogeneous
/// `AST` nodes.
struct AnyASTIdentity: Hashable {
  init<Node: AST>(of node: Node) {
    self.placement = node.site.region
    self.structure = node
  }

  /// The source code covered by the identified node.
  let placement: SourceRegion

  /// The type and (incidental) structure of the identified node.
  let structure: Any

  static func == (l: Self, r: Self) -> Bool {
    l.placement == r.placement && type(of: l.structure) == type(of: r.structure)
  }

  func hash(into h: inout Hasher) {
    placement.hash(into: &h)
    // Can't hash types directly; "upcast" to ObjectIdentifer first.
    ObjectIdentifier(type(of: structure)).hash(into: &h)
  }
}

/// The identity, in a program's source, of a `Node` instance.
///
/// `ASTIdentity` additionally carries the structure of the AST node as
/// auxilliary information for debugging purposes.  Otherwise it is equivalent
/// to SourceRegion.
struct ASTIdentity<Node: AST>: Hashable {
  init(of n: Node) {
    self.structure = n
  }
  /// The (incidental) structure of the identified node.
  let structure: Node

  static func == (l: Self, r: Self) -> Bool {
    l.structure.site.region == r.structure.site.region
  }

  func hash(into h: inout Hasher) {
    structure.site.region.hash(into: &h)
  }
}
