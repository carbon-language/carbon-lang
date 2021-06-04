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
struct Identifier: AST, Hashable {
  let text: String
  let site: Site
}

extension Identifier: CustomStringConvertible {
  var description: String { text }
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
    atom(Expression),             // An expression containing no bindings.
    variable(SimpleBinding),     // <Type>: <name>
    tuple(TuplePattern),
    functionCall(FunctionCall<Pattern>),
    functionType(FunctionTypePattern)

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
    expression(TypeExpression)

  init(_ e: TypeExpression) { self = .expression(e) }
  
  var site: Site {
    switch self {
    case let .auto(r): return r
    case let .expression(x): return x.site
    }
  }

  var expression: TypeExpression? {
    if case .expression(let x) = self { return x } else { return nil }
  }
  var isAuto: Bool {
    if case .auto = self { return true } else { return false }
  }
}

struct SimpleBinding: AST, Declaration {
  let type: TypeSpecifier
  let name: Identifier
  var site: Site { type.site...name.site }
}

struct FunctionCall<Argument: AST>: AST {
  let callee: Expression
  let arguments: TupleSyntax<Argument>

  var site: Site { return callee.site...arguments.site }
}

extension FunctionCall where Argument == Pattern {
  /// "Upcast" from literal to pattern
  init(_ literal: FunctionCall<Expression>) {
    self.init(callee: literal.callee, arguments: .init(literal.arguments))
  }
}

typealias TupleLiteral = TupleSyntax<Expression>
typealias TuplePattern = TupleSyntax<Pattern>
typealias TupleTypeLiteral = TupleSyntax<TypeExpression>

extension TupleLiteral {
  /// "Upcast" from tuple type literal.
  init(_ l: TupleTypeLiteral) {
    // This tuple has been discovered in type position.
    self.init(l.elements.map { Element($0) }, l.site)
  }
}
extension TuplePattern {
  /// "Upcast" from tuple literal.
  init(_ l: TupleLiteral) {
    self.init(l.elements.map { PatternElement($0) }, l.site)
  }

  /// "Upcast" from tuple type literal.
  init(_ l: TupleTypeLiteral) {
    self.init(l.elements.map { PatternElement($0) }, l.site)
  }
}

extension TupleTypeLiteral {
  /// "Downcast" from tuple literal.
  init(_ l: TupleLiteral) {
    // This tuple has been discovered in type position.
    self.init(l.elements.map { Element($0) }, l.site)
  }
}

extension TypeLiteralElement {
  /// "Downcast" from tuple literal.
  init(_ e: LiteralElement) {
    self.init(label: e.label, TypeExpression(e.payload))
  }
}

struct FunctionDefinition: AST, Declaration {
  let name: Identifier
  let parameters: TuplePattern
  let returnType: TypeSpecifier
  let body: Statement?
  let site: Site
  // Why no declaredType? -Jeremy
}

typealias MemberDesignator = Identifier

struct Alternative: AST, Declaration {
  let name: Identifier;
  let payload: TupleTypeLiteral
  let site: Site

  var dynamic_type: Type { .type }
}

struct StructDefinition: AST, TypeDeclaration {
  let name: Identifier
  let members: [StructMember]
  let site: Site

  /// The parameter type tuple used to initialize instances of the struct being
  /// defined.
  ///
  /// - Note: this node is synthesized; its `site` will match that of `self`.
  var initializerTuple: TupleSyntax<TypeExpression> {
    return TupleSyntax(
      members.map { .init(label: $0.name, $0.type) }, site)
  }

  var declaredType: Type { .struct(self.identity) }
  var dynamic_type: Type { .type }
}

struct ChoiceDefinition: AST, TypeDeclaration {
  let name: Identifier
  let alternatives: [Alternative]
  let site: Site

  var declaredType: Type { .choice(self.identity) }

  /// Returns the alternative with the given name, or `nil` if no such
  /// alternative exists.
  subscript(alternativeName: Identifier) -> Alternative? {
    alternatives.first { $0.name == alternativeName }
  }
  var dynamic_type: Type { .type }
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
    `if`(Expression, Statement, else: Statement?, Site),
    `return`(Expression, Site),
    block([Statement], Site),
    `while`(Expression, Statement, Site),
    match(subject: Expression, clauses: [MatchClause], Site),
    `break`(Site),
    `continue`(Site)

  var site: Site {
    switch self {
    case let .expressionStatement(_, r): return r
    case let .assignment(target: _, source: _, r): return r
    case let .initialization(v): return v.site
    case let .if(_, _, else: _, r): return r
    case let .return(_, r): return r
    case let .block(_, r): return r
    case let .while(_, _, r): return r
    case let .match(subject: _, clauses: _, r): return r
    case let .break(r): return r
    case let .continue(r): return r
    }
  }
}

struct TupleSyntax<Payload: AST>: AST {
  struct Element: AST {
    init(label: Identifier? = nil, _ payload: Payload) {
      self.label = label
      self.payload = payload
    }
    let label: Identifier?
    let payload: Payload

    var site: Site { label.map { $0.site...payload.site } ?? payload.site }
  }

  init<E: Collection>(_ elements: E, _ site: Site)
    where E.Element == Element
  {
    self.elements = .init(elements)
    self.site = site
  }

  let elements: [Element]
  let site: Site
}

typealias TypeLiteralElement = TupleSyntax<TypeExpression>.Element
typealias LiteralElement = TupleSyntax<Expression>.Element
typealias PatternElement = TupleSyntax<Pattern>.Element

extension LiteralElement {
  // "Downcast" from type literal element
  init(_ l: TypeLiteralElement) {
    self.init(label: l.label, l.payload.body)
  }
}

extension PatternElement {
  // "Upcast" from literal element
  init(_ l: LiteralElement) {
    self.init(label: l.label, .atom(l.payload))
  }

  // "Upcast" from literal element
  init(_ l: TypeLiteralElement) {
    self.init(label: l.label, .atom(l.payload.body))
  }
}

extension TupleSyntax: RandomAccessCollection {
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

struct FunctionTypeSyntax<Parameter: AST>: AST {
  let parameters: TupleSyntax<Parameter>
  let returnType: Parameter
  let site: Site
}
typealias FunctionTypePattern = FunctionTypeSyntax<Pattern>
typealias FunctionTypeLiteral = FunctionTypeSyntax<TypeExpression>

extension FunctionTypePattern {
  /// "Upcast" from literal to pattern
  init(_ source: FunctionTypeLiteral) {
    self.init(
      parameters: .init(source.parameters),
      returnType: .atom(source.returnType.body),
      site: source.site)
  }
}

indirect enum Expression: AST {
  case
    name(Identifier),
    memberAccess(MemberAccessExpression),
    index(target: Expression, offset: Expression, Site),
    integerLiteral(Int, Site),
    booleanLiteral(Bool, Site),
    tupleLiteral(TupleLiteral),
    unaryOperator(UnaryOperatorExpression),
    binaryOperator(BinaryOperatorExpression),
    functionCall(FunctionCall<Expression>),
    intType(Site),
    boolType(Site),
    typeType(Site),
    functionType(FunctionTypeLiteral)

  var site: Site {
    switch self {
    case let .name(v): return v.site
    case let .memberAccess(x): return x.site
    case let .index(target: _, offset: _, r): return r
    case let .integerLiteral(_, r): return r
    case let .booleanLiteral(_, r): return r
    case let .tupleLiteral(t): return t.site
    case let .unaryOperator(x): return x.site
    case let .binaryOperator(x): return x.site
    case let .functionCall(f): return f.site
    case let .intType(r): return r
    case let .boolType(r): return r
    case let .typeType(r): return r
    case let .functionType(t): return t.site
    }
  }
};

struct UnaryOperatorExpression: AST {
  let operation: Token, operand: Expression
  var site: Site { operation.site...operand.site }
}

struct BinaryOperatorExpression: AST {
  let operation: Token, lhs: Expression, rhs: Expression
  var site: Site { lhs.site...rhs.site }
}

struct MemberAccessExpression: AST {
  let base: Expression
  let member: Identifier

  var site: ASTSite { base.site...member.site }
}

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

  /// Creates an instance from a tuple of TypeExpressions.
  init(_ t: TupleTypeLiteral) {
    self.body = .tupleLiteral(TupleLiteral(t))
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
/// equality and hashability based on structure.
struct ASTSite: Hashable {
  /// Creates an instance storing `r` without making it part of the value of
  /// `self`.
  init(devaluing r: SourceRegion) { self.region = r }

  let region: SourceRegion

  static func == (_: Self, _: Self) -> Bool { true }
  func hash(into h: inout Hasher) {}

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

extension ASTSite: CustomStringConvertible {
  var description: String { "\(region)" }
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
