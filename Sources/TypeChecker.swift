// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

var UNIMPLEMENTED: Never { fatalError("unimplemented") }

struct TypeChecker {
  init(_ program: [Declaration]) throws {
    scope = Stack()
    enterScope() // global scope.

    for d in program {
      try visit(d)
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
}

private extension TypeChecker {
  mutating func enterScope() {
    scope.push([]) // Prepare the global scope
  }

  mutating func leaveScope() {
    for name in scope.pop()! {
      _ = symbolTable[name]!.pop()
    }
  }

  mutating func define(_ name: Identifier, _ definition: Decl) throws {
    if scope.top.contains(name.text) {
      throw CompileError(
        "'\(name.text)' already defined in this scope", at: name.site,
        notes: [("previous definition", symbolTable[name.text]!.top.site)])
    }

    scope.top.insert(name.text)
    symbolTable[name.text, default: Stack()].push(definition)
  }

  mutating func visit(_ name: Identifier) throws -> Decl {
    guard let d = symbolTable[name.text]?.elements.last else {
      throw CompileError("Unknown name '\(name.text)'", at: name.site)
    }
    toDeclaration[name] = d
    return d
  }

  mutating func visit(_ node: Declaration) throws {
    switch node {
    case let .function(f):
      // Handle forward declarations (they're in the grammar).
      guard let body = f.body else { UNIMPLEMENTED }

      try define(f.name, .decl(node))
      enterScope()
      for p in f.parameterPattern.elements {
        try visit(asFunctionParameter: p)
      }
      try visit(body)
      leaveScope()

    case let .struct(s):
      try define(s.name, .decl(node))
      enterScope()
      for m in s.members {
        try visit(asStructMember: m)
      }
      leaveScope()

    case let .choice(c):
      try define(c.name, .decl(node))
      enterScope()

      leaveScope()
    case let .variable(name: n, type: t, initializer: i, _): UNIMPLEMENTED
    }
  }

  mutating func visit(asStructMember m: VariableDeclaration) throws {
    UNIMPLEMENTED
  }

  mutating func visit(asFunctionParameter p: TupleLiteralElement) throws {
  }

  mutating func visit(_ node: FunctionDefinition) throws {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: Alternative) throws {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: Statement) throws {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: MatchClauseList) throws {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: MatchClause) throws {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: TupleLiteral) throws {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: Expression) throws {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: Field) throws {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: FieldList) throws {
    UNIMPLEMENTED
  }

  mutating func visit(_ node: VariableDeclaration) throws {
    UNIMPLEMENTED
  }
}
