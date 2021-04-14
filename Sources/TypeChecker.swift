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

  /// Records the declaration associated with the given use.
  mutating func visit(asUse use: Identifier) {
    guard let d = symbolTable[use.text]?.elements.last else {
      error("Un-declared name '\(use.text)'", at: use.site)
      return
    }
    toDeclaration[use] = d
  }

  /// Typechecks `d` in the current context.
  mutating func visit(_ d: Declaration) {
    switch d {
    case let .function(f):
      // TODO: handle forward declarations (they're in the grammar).
      guard let body = f.body else { UNIMPLEMENTED }

      define(f.name, .decl(d))
      inNewScope {
        for p in f.parameterPattern.elements {
          $0.visit(asFunctionParameter: p)
        }
        $0.visit(body)
      }

    case let .struct(s):
      define(s.name, .decl(d))
      inNewScope {
        for m in s.members { $0.visit(asStructMember: m) }
      }

    case let .choice(c):
      define(c.name, .decl(d))
      inNewScope {
        for a in c.alternatives { $0.visit(a) }
      }

    case let .variable(name: n, type: t, initializer: i, _):
      define(n, .decl(d))
      visit(t)
      visit(i)
      let t1 = evaluateTypeExpression(t) ?? toType[.expression(i)]
    }
  }

  /// Returns the type described by `e`, or `nil` if e is `auto`, or .error if
  /// `e` doesn't describe a type.
  mutating func evaluateTypeExpression(_ e: Expression) -> Type? {
    switch e {
    case let .name(n):
      visit(asUse: n)
      return nil
    case let .getField(_, _, _): return .error
    case let .index(target: _, offset: _, _): return .error
    case let .patternVariable(name: _, type: _, _): return .error
    case let .integerLiteral(_, _): return .error
    case let .booleanLiteral(_, _): return .error
    case let .tupleLiteral(t): return .error
    case let .unaryOperator(operation: _, operand: _, _): return .error
    case let .binaryOperator(operation: _, lhs: _, rhs: _, _): return .error
    case let .functionCall(callee: _, arguments: _, _): return .error
    case let .intType(_): return .error
    case let .boolType(_): return .error
    case let .typeType(_): return .error
    case let .autoType(_): return nil
    case let .functionType(parameterTypes: _, returnType: _, _): return .error
    }
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
