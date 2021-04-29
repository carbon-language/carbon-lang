// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

var UNIMPLEMENTED: Never { fatalError("unimplemented") }

struct TypeChecker {
  init(_ program: [TopLevelDeclaration]) {
    activeScopes = Stack()
    activeScopes.push([]) // Prepare the global scope

    for d in program {
      visit(d)
    }
  }

  /// The function currently being typechecked.
  var currentFunction: FunctionDefinition?

  var declaration = ASTDictionary<Identifier, Declaration>()
  var declaredType = Dictionary<Declaration.Identity, Type>()
  var expressionType = ASTDictionary<Expression, Type>()

  /// A mapping from names to the declarations they reference in the current
  /// scope.
  var symbolTable: StackDictionary<String, Declaration> = .init()

  /// The set of names defined in each scope, with the current scope at the top.
  var activeScopes: Stack<Set<String>>

  /// A record of the collected errors.
  var errors: ErrorLog = []
}

private extension TypeChecker {
  /// Adds the given error to the error log.
  mutating func error(
    _ message: String, at site: ASTSite, notes: [CompileError.Note] = []
  ) {
    errors.append(CompileError(message, at: site, notes: notes))
  }

  /// Returns the result of running `body(&self)` in a new sub-scope of the
  /// current one.
  mutating func inNewScope<R>(do body: (inout TypeChecker)->R) -> R {
    activeScopes.push([])
    let r = body(&self)
    for name in activeScopes.pop()! {
      symbolTable.pop(at: name)
    }
    return r
  }

  /// Records that `name.text` refers to `definition` in the current scope.
  mutating func define(_ name: Identifier, _ definition: Declaration) {
    if activeScopes.top.contains(name.text) {
      error(
        "'\(name.text)' already defined in this scope", at: name.site,
        notes:
          [("previous definition", symbolTable[name.text].site)])
    }

    activeScopes.top.insert(name.text)
    symbolTable.push(definition, at: name.text)
  }

  /// Records and returns the declaration associated with the given use, or
  /// `nil` if the name is not defined in the current scope.
  ///
  /// Every identifier should either be declaring a new name, or should be
  /// looked up during type checking.
  mutating func lookup(_ use: Identifier) -> Declaration? {
    guard let d = symbolTable[query: use.text] else {
      error("Un-declared name '\(use.text)'", at: use.site)
      return nil
    }
    declaration[use] = d
    return d
  }

  /// Typechecks `d` in the current context.
  mutating func visit(_ d: TopLevelDeclaration) {
    switch d {
    case let .function(f): visit(f)
    case let .struct(s): visit(s)
    case let .choice(c):
      define(c.name, c)
      inNewScope {
        for a in c.alternatives { $0.visit(a) }
      }

    case let .initialization(v):
      _ = v
      /*
      define(v.name, .init(d))
      visit(v.type)
      visit(v.initializer)
      let t1 = evaluateTypeExpression(
        v.type, initializingFrom: expressionType[v.initializer])
      declaredType[.init(d)] = t1
       */
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
      if let r = (d as? TypeDeclaration)?.declaredType { return r }
      error(
        "'\(n.text)' does not refer to a type",
        at: n.site, notes: [("actual definition: \(d)", d.site)])
      return .error

    case .intType: return .int
    case .boolType: return .bool
    case .typeType: return .type
    case .autoType:
      if let r = rhs { return r }
      error("No initializer from which to deduce type.", at: e.site)
      return .error

    case .functionType(let f0):
      if rhs != nil && rhs!.function == nil { return .error }
      let (p1, r1) = rhs?.function ?? (nil, nil)

      return .function(
        parameterTypes: mapDeducedType(f0.parameters, p1),
        returnType: evaluateTypeExpression(f0.returnType, initializingFrom: r1))

    case .tupleLiteral(let t0):
      if rhs != nil && rhs!.tuple == nil { return .error }
      let types = mapDeducedType(t0, rhs?.tuple)
      return .tuple(types)

    case .getField, .index, .integerLiteral,
         .booleanLiteral, .unaryOperator, .binaryOperator, .functionCall:
      error("Type expression expected", at: e.site)
      return .error
    }
  }

  /// Returns the result of mapping `deducedType` over `zip(e, rhs)`, or over
  /// `e` if `rhs` is `nil`.
  private mutating func mapDeducedType<E: Collection>(
    _ e: E, _ rhs: [Type]? = nil
  ) -> [Type]
    where E.Element == LiteralElement
  {
    guard let r = rhs else { return e.map { deducedType($0.value) } }
    return zip(e, r).map { deducedType($0.value, initializingFrom: $1) }
  }


  /// Returns the type described by `e`, or `.error` if `e` doesn't describe a
  /// type, using `rhs`, if supplied, to do any type deduction using the
  /// semantics of:
  ///
  ///   var <e>: x = <rhs>
  ///
  /// where <e> `e`, and <rhs> is an expression of type `rhs`.
  mutating func evaluateTypeExpression(
    _ e: TypeExpression, initializingFrom rhs: Type? = nil
  ) -> Type {
    let r = deducedType(e.body, initializingFrom: rhs)

    // Final validation.
    if let r1 = rhs, r1 != r {
      error("Initialization value has wrong type: \(r1)", at: e.site)
    }
    return r
  }

  mutating func visit(_ f: FunctionDefinition) {
    // TODO: handle forward declarations (they're in the grammar).
    guard let body = f.body else { UNIMPLEMENTED }

    define(f.name, f)
    
    inNewScope { me in
      var parameterTypes: [Type] = []
      parameterTypes.reserveCapacity(0)

      for p in f.parameters {
        _ = p
        /*
        me.define(p.name, .binding(p))
        let t = me.evaluateTypeExpression(p.type)
        me.declaredType[.binding(p)] = t
        parameterTypes.append(t)

         */
      }
//      let r = me.evaluateTypeExpression(f.returnType)
//      me.declaredType[f.identity] = .function(parameterTypes: parameterTypes, returnType: r)
      me.currentFunction = f
      me.visit(body)
      me.currentFunction = nil
      UNIMPLEMENTED
    }
  }

  mutating func visit(_ s: StructDefinition) {
    define(s.name, s)
    inNewScope { me in
      for m in s.members {
        me.define(m.name, m)
        me.declaredType[m.identity] = me.evaluateTypeExpression(m.type)
      }
    }
  }

  
  mutating func visit(_ a: Alternative) {
    define(a.name, a)
    declaredType[a.identity] = .tuple(mapDeducedType(a.payload, nil))
  }

  mutating func visit(_ s: Statement) {
    switch s {
    case let .expressionStatement(e, _): visit(e)
    case let .assignment(target: t, source: s, _):
      visit(t)
      visit(s)
      let targetType = expressionType[t]!
      let sourceType = expressionType[s]!
      // TODO: check LHS for lvalue-ness.
      if targetType != sourceType {
        error(
          "Can't assign expression of type \(sourceType)"
            + "to lvalue of type \(targetType)", at: t.site...s.site)
      }
    case let .initialization(v):
      _ = v
      /*
      visit(p)
      visit(i)
      let t = evaluateTypeExpression(
        p, initializingFrom: expressionType[i])
        */

    case let .if(condition: c, thenClause: then, elseClause: maybeElse, _):
      visit(c)
      let conditionType = expressionType[c]!
      if conditionType != .bool {
        error(
          "Expecting a bool expression in 'if', got \(conditionType)",
          at: c.site)
      }
      visit(then)
      if let e = maybeElse { visit(e) }
    case let .return(e, _):
      visit(e)
      guard case .function(_, returnType: let r)
              = declaredType[currentFunction!.identity]
      else {
        fatalError("function without function type")
      }
      let t = expressionType[e]!
      if t != r {
        error("Expected return type \(r); got \(t)", at: e.site)
      }
    case /*let*/ .sequence(_, _, _): UNIMPLEMENTED
    case /*let*/ .block(_, _): UNIMPLEMENTED
    case /*let*/ .while(condition: _, body: _, _): UNIMPLEMENTED
    case /*let*/ .match(subject: _, clauses: _, _): UNIMPLEMENTED
    case /*let*/ .break(_): UNIMPLEMENTED
    case /*let*/ .continue(_): UNIMPLEMENTED
    }
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

  /// Typechecks `m` in the current context.
  mutating func visit(_ m: StructMemberDeclaration) {
    UNIMPLEMENTED
  }
}
