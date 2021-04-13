// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

var UNIMPLEMENTED: Never { fatalError("unimplemented") }

struct TypeChecker {
  init(_ program: [Declaration]) {
    scope = Stack()
    enterScope() // global scope.

    for d in program {
      visit(d)
    }
  }

  /// Unify all declarations (workaround for
  /// https://github.com/carbon-language/carbon-lang/issues/456).
  enum Decl {
    case decl(Declaration), param(Identifier)

    var site: SourceRegion {
      switch self {
      case .decl(let d): return d.site
      case .param(let p): return p.site
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
  var scope: Stack<Set<String>>

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

  mutating func enterScope() {
    scope.push([]) // Prepare the global scope
  }

  mutating func leaveScope() {
    for name in scope.pop()! {
      _ = symbolTable[name]!.pop()
    }
  }

  mutating func inNewScope<R>(do body: (inout TypeChecker)->R) -> R {
    enterScope()
    defer { leaveScope() }
    return body(&self)
  }

  mutating func define(_ name: Identifier, _ definition: Decl) {
    if scope.top.contains(name.text) {
      error(
        "'\(name.text)' already defined in this scope", at: name.site,
        notes: [("previous definition", symbolTable[name.text]!.top.site)])
    }

    scope.top.insert(name.text)
    symbolTable[name.text, default: Stack()].push(definition)
  }

  mutating func visit(_ name: Identifier) {
    guard let d = symbolTable[name.text]?.elements.last else {
      error("Un-declared name '\(name.text)'", at: name.site)
      return
    }
    toDeclaration[name] = d
  }

  mutating func visit(_ node: Declaration) {
    switch node {
    case let .function(f):
      // TODO: handle forward declarations (they're in the grammar).
      guard let body = f.body else { UNIMPLEMENTED }

      define(f.name, .decl(node))
      inNewScope {
        for p in f.parameterPattern.elements {
          $0.visit(asFunctionParameter: p)
        }
        $0.visit(body)
      }

    case let .struct(s):
      define(s.name, .decl(node))
      inNewScope {
        for m in s.members { $0.visit(asStructMember: m) }
      }

    case let .choice(c):
      define(c.name, .decl(node))
      inNewScope {
        for a in c.alternatives { $0.visit(a) }
      }

    case let .variable(name: n, type: t, initializer: i, _):
      define(n, .decl(node))
      visit(t)
      visit(i)
      let t1 = evaluateTypeExpression(t) ?? toType[.expression(i)]
    }
  }

  func evaluateTypeExpression(_ e: Expression) -> Type {
    UNIMPLEMENTED
    /*
    switch e {
    case let .variable(v): return v.site
    case let .getField(_, _, r): return r
    case let .index(target: _, offset: _, r): return r
    case let .patternVariable(name: _, type: _, r): return r
    case let .integerLiteral(_, r): return r
    case let .booleanLiteral(_, r): return r
    case let .tupleLiteral(t): return t.site
    case let .unaryOperator(operation: _, operand: _, r): return r
    case let .binaryOperator(operation: _, lhs: _, rhs: _, r): return r
    case let .functionCall(callee: _, arguments: _, r): return r
    case let .intType(r): return r
    case let .boolType(r): return r
    case let .typeType(r): return r
    case let .autoType(r): return r
    case let .functionType(parameterTypes: _, returnType: _, r):
    }
     */
  }

  mutating func visit(asStructMember m: VariableDeclaration) {
    UNIMPLEMENTED
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

  mutating func visit(_ node: VariableDeclaration) {
    UNIMPLEMENTED
  }
}
