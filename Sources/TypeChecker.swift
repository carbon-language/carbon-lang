// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

var unimplemented: Never { fatalError("unimplemented") }

struct TypeChecker {
  enum Decl { case decl(Declaration), param(Identifier) }
  var toDeclaration = PropertyMap<Identifier, Decl>()
  var toType = PropertyMap<Typed, Type>()

  var symbolTable: [String: Stack<Decl>] = [:]
  var scope: [Set<String>] = [[]]

  init(_ program: [Declaration]) throws {
    for d in program {
      try visit(d)
    }
  }

  mutating func visit(_ node: Identifier) throws {
    guard let d = symbolTable[node^]?.elements.last else {
      throw CompileError("Unknown name node^.name", at: node.site)
    }
    toDeclaration[node] = d
  }

  mutating func visit(_ node: Declaration) throws {
    switch node^ {
    case let .function(f): try visit(f)
    case let .struct(s): unimplemented
    case let .choice(c): unimplemented
    case let .variable(name: n, type: t, initializer: i): unimplemented
    }
  }

  mutating func visit(_ node: FunctionDefinition) throws {
  }

  mutating func visit(_ node: Alternative) throws {
  }

  mutating func visit(_ node: Statement) throws {
  }

  mutating func visit(_ node: MatchClauseList) throws {
  }

  mutating func visit(_ node: MatchClause) throws {
  }

  mutating func visit(_ node: TupleLiteral) throws {
  }

  mutating func visit(_ node: Expression) throws {
  }

  mutating func visit(_ node: Field) throws {
  }

  mutating func visit(_ node: FieldList) throws {
  }

  mutating func visit(_ node: VariableDeclaration) throws {
  }
}
