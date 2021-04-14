// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

var UNIMPLEMENTED: Never { fatalError("unimplemented") }

struct TypeChecker {
  init(_ program: [Declaration]) {
    activeScopes = Stack()
    activeScopes.push([]) // Prepare the global scope

    for d in program {
      visit(d)
    }
  }

  /// Unify all declarations (workaround for
  /// https://github.com/carbon-language/carbon-lang/issues/456).
  enum Decl {
    case declaration(Declaration),
         functionParameter(Identifier),
         structMember(StructMemberDeclaration)

    var site: SourceRegion {
      switch self {
      case .declaration(let d): return d.site
      case .functionParameter(let p): return p.site
      case .structMember(let m): return m.site
      }
    }
  }

  var toDeclaration = PropertyMap<Identifier, Decl>()
  var toType = PropertyMap<Typed, Type>()

  /// A mapping from names to the stack of declcarations they reference, with
  /// the top of each stack being the declaration referenced in the current
  /// scope.
  var symbolTable: [String: Stack<Decl>] = [:]

  /// The set of names defined in each scope, with the current scope at the top.
  var activeScopes: Stack<Set<String>>

  /// A record of the collected errors.
  var errors: ErrorLog = []
}

private extension TypeChecker {
  /// Adds the given error to the error log.
  mutating func error(
    _ message: String, at site: SourceRegion, notes: [CompileError.Note] = []
  ) {
    errors.append(CompileError(message, at: site, notes: notes))
  }

  /// Returns the result of running `body(&self` in a new sub-scope of the
  /// current one.
  mutating func inNewScope<R>(do body: (inout TypeChecker)->R) -> R {
    activeScopes.push([])
    let r = body(&self)
    for name in activeScopes.pop()! {
      _ = symbolTable[name]!.pop()
    }
    return r
  }

  /// Records that `name.text` refers to `definition` in the current scope.
  mutating func define(_ name: Identifier, _ definition: Decl) {
    if activeScopes.top.contains(name.text) {
      error(
        "'\(name.text)' already defined in this scope", at: name.site,
        notes: [("previous definition", symbolTable[name.text]!.top.site)])
    }

    activeScopes.top.insert(name.text)
    symbolTable[name.text, default: Stack()].push(definition)
  }

  /// Records and returns the declaration associated with the given use, or
  /// `nil` if the name is not defined in the current scope.
  ///
  /// Every identifier should either be declaring a new name, or should be
  /// looked up during type checking.
  mutating func lookup(_ use: Identifier) -> Decl? {
    guard let d = symbolTable[use.text]?.elements.last else {
      error("Un-declared name '\(use.text)'", at: use.site)
      return nil
    }
    toDeclaration[use] = d
    return d
  }

  /// Typechecks `d` in the current context.
  mutating func visit(_ d: Declaration) {
    switch d {
    case let .function(f):
      // TODO: handle forward declarations (they're in the grammar).
      guard let body = f.body else { UNIMPLEMENTED }

      define(f.name, .declaration(d))
      inNewScope {
        for p in f.parameterPattern.elements {
          $0.visit(asFunctionParameter: p)
        }
        $0.visit(body)
      }

    case let .struct(s):
      define(s.name, .declaration(d))
      inNewScope {
        for m in s.members { $0.visit(m) }
      }

    case let .choice(c):
      define(c.name, .declaration(d))
      inNewScope {
        for a in c.alternatives { $0.visit(a) }
      }

    case let .variable(name: n, type: t, initializer: i, _):
      define(n, .declaration(d))
      visit(t)
      visit(i)
      let t1 = evaluateTypeExpression(
        t, initializingFrom: toType[.expression(i)])
      toType[.declaration(d)] = t1
    }
  }

  /// Returns the concrete type deduced for the type expression `e` given
  /// an initialization from an expression of type `rhs`.
  ///
  /// Most of the work of `evaluateTypeExpression` happens here but a final
  /// validation step is needed.
  private mutating func deducedType(
    _ e: Expression, initializingFrom rhs: Type? = nil
  ) -> Type {
    func nils(_ n: Int) -> [Type?] { Array(repeating: nil, count: n) }

    switch e {
    case let .name(n):
      guard let d = lookup(n) else { return .error } // Name not defined
      switch d {
      case .declaration(.struct(let d)):
        return .struct(d)

      case .declaration(.choice(let d)):
        return .choice(d)

      case .declaration(.function), .declaration(.variable),
           .functionParameter, .structMember:
        error(
          "'\(n.text)' does not refer to a type", at: n.site,
          notes: [("actual definition: \(d)", d.site)])
        return .error
      }

      case .intType: return .int
      case .boolType: return .bool
      case .typeType: return .type
      case .autoType:
        if let r = rhs { return r }
        error("No initializer from which to deduce type.", at: e.site)
        return .error

      case .functionType(parameterTypes: let p0, returnType: let r0, _):
        if rhs != nil && rhs!.function == nil { return .error }
        let (p1, r1) = rhs?.function ?? (nil, nil)

        return .function(
          parameterTypes: mapDeducedType(p0.lazy.map(\.value), p1),
          returnType: evaluateTypeExpression(r0, initializingFrom: r1))

      case .tupleLiteral(let t0):
        if rhs != nil && rhs!.tuple == nil { return .error }
        return .tuple(mapDeducedType(t0.lazy.map(\.value), rhs?.tuple))

      case .getField, .index, .patternVariable, .integerLiteral,
           .booleanLiteral, .unaryOperator, .binaryOperator, .functionCall:
        error("Type expression expected", at: e.site)
        return .error
    }
  }

  /// Returns the result of mapping `deducedType` over `zip(e, rhs)`, or over
  /// `e` if `rhs` is `nil`.
  private mutating func mapDeducedType<E: Collection, R: Collection>(
    _ e: E, _ rhs: R?
  ) -> [Type]
    where E.Element == Expression, R.Element == Type
  {
    guard let r = rhs else { return e.map { deducedType($0) } }
    return zip(e, r).map { deducedType($0, initializingFrom: $1) }
  }


  /// Returns the type described by `e`, or `.error` if `e` doesn't describe a
  /// type, using `rhs`, if supplied, to do any type deduction using the
  /// semantics of:
  ///
  ///   var <e>: x = <rhs>
  ///
  /// where <e> `e`, and <rhs> is an expression of type `rhs`.
  mutating func evaluateTypeExpression(
    _ e: Expression, initializingFrom rhs: Type? = nil
  ) -> Type {
    let r = deducedType(e, initializingFrom: rhs)

    // Final validation.
    if let r1 = rhs, r1 != r {
      error("Initialization value has wrong type: \(r1)", at: e.site)
    }
    return r
  }

  mutating func visit(asFunctionParameter p: TupleLiteralElement) {
  }

  mutating func visit(_ node: FunctionDefinition) {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: Alternative) {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: Statement) {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: MatchClauseList) {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: MatchClause) {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: TupleLiteral) {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: Expression) {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: Field) {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: FieldList) {
    UNIMPLEMENTED
  }

  /// Typechecks `m` in the current context.
  mutating func visit(_ m: StructMemberDeclaration) {
    UNIMPLEMENTED
  }
}
