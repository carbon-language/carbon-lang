// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// The name resolution algorithm and associated data.
struct NameResolution {
  init(_ program: AbstractSyntaxTree) {
    activeScopes = Stack()
    activeScopes.push([]) // Prepare the global scope

    // Collect the definitions of top-level entities.
    for d in program {
      defineOuterScopeEntities(declaredBy: d)
    }

    for d in program {
      resolveNames(usedIn: d)
    }
  }

  /// The ultimate result of name resolution; a mapping from identifier use to
  /// the declared entity referenced.
  private(set) var definition = ASTDictionary<Identifier, Declaration>()

  /// A mapping from names to the declarations they reference in the current
  /// scope.
  private var symbolTable: StackDictionary<String, Declaration> = .init()

  /// The set of names defined in each scope, with the current scope at the top.
  private var activeScopes: Stack<Set<String>>

  /// A record of any collected errors.
  private(set) var errors: ErrorLog = []
}

private extension NameResolution {
  /// Returns the result of running `body(&self)` in a new sub-scope of the
  /// current one.
  mutating func inNewScope<R>(do body: (inout NameResolution)->R) -> R {
    activeScopes.push([])
    let r = body(&self)
    let newlyDefined = activeScopes.pop()!
    
    for name in newlyDefined {
      symbolTable.pop(at: name)
    }
    return r
  }

  /// Records that `name.text` refers to `definition` in the current scope.
  mutating func define(_ name: Identifier, _ definition: Declaration) {
    if activeScopes.top.contains(name.text) {
      error(
        name, "'\(name)' already defined in this scope",
        notes:
          [("previous definition", symbolTable[name.text].site)])
    }

    activeScopes.top.insert(name.text)
    symbolTable.push(definition, at: name.text)
  }

  /// Records the declaration associated with the given name use, or records an
  /// error if the name is not defined in the current scope.
  mutating func use(_ name: Identifier) {
    if let d = symbolTable[query: name.text] {
      definition[name] = d
      return
    }
    error(name, "Un-declared name '\(name)'")
  }

  /// Adds an error at the site of `offender` to the error log.
  mutating func error<Node: AST>(
    _ offender: Node, _ message: String , notes: [CarbonError.Note] = []
  ) {
    errors.append(CarbonError(message, at: offender.site, notes: notes))
  }
}

/// Registering definitions and checking for redefinitions.
private extension NameResolution {
  /// Records the names of the global-scope entity/ies declared by `d`, and
  /// associates them with their AST fragments, reporting any redefinitions of a
  /// name at global scope as errors.
  mutating func defineOuterScopeEntities(declaredBy d: TopLevelDeclaration) {
    switch d {
    case .function(let f): define(f.name, f)
    case .struct(let s): define(s.name, s)
    case .choice(let c): define(c.name, c)
    case .initialization(let i): defineVariables(declaredBy: i.bindings)
    }
  }

  mutating func defineVariables(declaredBy bindings: Pattern) {
    switch bindings {
    case .atom:
      return
    case .variable(let b):
      define(b.name, b)
    case .tuple(let t):
      defineVariables(declaredBy: t)
    case .functionCall(let f):
      defineVariables(declaredBy: f.arguments)
    case .functionType(let t):
      defineVariables(declaredBy: t.parameters)
      defineVariables(declaredBy: t.returnType)
    }
  }

  /// Associates every use of a name in `d` with the definition it references.
  mutating func resolveNames(usedIn d: TopLevelDeclaration) {
    switch d {
    case .function(let f): resolveNames(usedIn: f)
    case .struct(let s): resolveNames(usedIn: s)
    case .choice(let c): resolveNames(usedIn: c)
    case .initialization(let i): resolveNames(usedIn: i)
    }
  }

  mutating func defineVariables(declaredBy bindings: TuplePattern) {
    for p in bindings { defineVariables(declaredBy: p.payload) }
  }

  mutating func resolveNames(usedIn f: FunctionDefinition) {
    inNewScope { me in
      me.defineVariables(declaredBy: f.parameters)
      me.resolveNames(usedIn: f.parameters)
      me.resolveNames(usedIn: f.returnType)

      if let body = f.body {
        me.resolveNames(usedIn: body)
      }
    }
  }

  mutating func resolveNames(usedIn s: StructDefinition) {
    inNewScope { me in
      for m in s.members {
        me.resolveNames(usedIn: m.type.body)
        me.define(m.name, m)
      }
    }
  }

  mutating func resolveNames(usedIn c: ChoiceDefinition) {
    inNewScope { me in
      for a in c.alternatives {
        me.define(a.name, a)
        me.resolveNames(usedIn: a.payload)
      }
    }
  }

  mutating func resolveNames(usedIn t: TupleLiteral) {
    for e in t { resolveNames(usedIn: e.payload) }
  }

  mutating func resolveNames(usedIn t: TupleSyntax<TypeExpression>) {
    for e in t { resolveNames(usedIn: e.payload) }
  }

  mutating func resolveNames(usedIn t: TuplePattern) {
    for e in t { resolveNames(usedIn: e.payload) }
  }

  mutating func resolveNames(usedIn i: Initialization) {
    resolveNames(usedIn: i.bindings)
    resolveNames(usedIn: i.initializer)
  }

  mutating func resolveNames(usedIn e: Expression) {
    switch e {
    case let .name(v):
      use(v)

    case let .memberAccess(e):
      resolveNames(usedIn: e.base)
      // Only unqualified names get resolved; member lookups can only be
      // done by the typechecker, once the type of the base is known.

    case let .index(target: t, offset: o, _):
      resolveNames(usedIn: t)
      resolveNames(usedIn: o)

    case let .tupleLiteral(t):
      resolveNames(usedIn: t)

    case let .unaryOperator(x):
      resolveNames(usedIn: x.operand)

    case let .binaryOperator(x):
      resolveNames(usedIn: x.lhs)
      resolveNames(usedIn: x.rhs)

    case let .functionCall(f):
      resolveNames(usedIn: f)

    case let .functionType(t):
      resolveNames(usedIn: t)

    case .integerLiteral, .booleanLiteral,
         .intType, .boolType, .typeType: ()
    }
  }

  mutating func resolveNames(usedIn t: TypeExpression) {
    resolveNames(usedIn: t.body)
  }

  mutating func resolveNames(usedIn t: TypeSpecifier) {
    if case let .expression(e) = t { resolveNames(usedIn: e) }
  }

  mutating func resolveNames(usedIn s: Statement) {
    switch s {
    case let .expressionStatement(e, _):
      resolveNames(usedIn: e)
    case let .assignment(target: t, source: s, _):
      resolveNames(usedIn: t)
      resolveNames(usedIn: s)
    case let .initialization(v):
      resolveNames(usedIn: v)
    case let .if(condition, s0, else: s1, _):
      resolveNames(usedIn: condition)
      resolveNames(usedIn: s0)
      if let s1 = s1 {
         resolveNames(usedIn: s1)
      }
    case let .return(e, _):
      resolveNames(usedIn: e)
    case let .block(b, _):
      inNewScope { me in
        // TODO: prohibit the use of initializations AS then or else clauses,
        // etc.(?)
        for substatement in b {
          // The only kind of statement that introduces new names.
          if case .initialization(let i) = substatement {
            me.defineVariables(declaredBy: i.bindings)
          }
          me.resolveNames(usedIn: substatement)
        }
      }
    case let .while(condition: c, body: b, _):
      resolveNames(usedIn: c)
      resolveNames(usedIn: b)
    case let .match(subject: s, clauses: clauses, _):
      resolveNames(usedIn: s)
      for clause in clauses { resolveNames(usedIn: clause) }
    case .break, .continue: ()
    }
  }

  mutating func resolveNames(usedIn p: Pattern) {
    switch p {
    case let .atom(x): resolveNames(usedIn: x)
    case let .variable(x): resolveNames(usedIn: x)
    case let .tuple(x): resolveNames(usedIn: x)
    case let .functionCall(x): resolveNames(usedIn: x)
    case let .functionType(x): resolveNames(usedIn: x)
    }
  }

  mutating func resolveNames(usedIn b: SimpleBinding) {
    resolveNames(usedIn: b.type)
  }

  mutating func resolveNames(usedIn m: MatchClause) {
    inNewScope { me in
      if let p = m.pattern {
        me.defineVariables(declaredBy: p)
        me.resolveNames(usedIn: p)
      }
      me.resolveNames(usedIn: m.action)
    }
  }

  mutating func resolveNames(usedIn f: FunctionCall<Expression>) {
    resolveNames(usedIn: f.callee)
    resolveNames(usedIn: f.arguments)
  }

  mutating func resolveNames(usedIn f: FunctionCall<Pattern>) {
    resolveNames(usedIn: f.callee)
    resolveNames(usedIn: f.arguments)
  }

  mutating func resolveNames(usedIn f: FunctionTypePattern) {
    resolveNames(usedIn: f.parameters)
    resolveNames(usedIn: f.returnType)
  }

  mutating func resolveNames(usedIn f: FunctionTypeLiteral) {
    resolveNames(usedIn: f.parameters)
    resolveNames(usedIn: f.returnType)
  }
}
