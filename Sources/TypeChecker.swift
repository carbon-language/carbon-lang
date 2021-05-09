// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A marker for code that needs to be implemented.  Eventually all of these
/// should be eliminated from the codebase.
var UNIMPLEMENTED: Never { fatalError("unimplemented") }

/// A marker for code that should never be reached.
var UNREACHABLE: Never { fatalError("unreachable.") }

struct TypeChecker {
  init(_ program: ExecutableProgram) {
    self.program = program

    for d in program.ast {
      registerParentage(d)
    }
    for d in program.ast {
      checkNominalTypeBody(d)
    }
    for d in program.ast {
      if case .function(let f) = d { _ = type(f) }
    }
    /*
    for d in program.ast {
      checkFunctionBody(d)
    }
    */
  }

  private let program: ExecutableProgram

  /// Mapping from alternative declaration to the choice in which it is defined.
  private var parent = ASTDictionary<Alternative, ChoiceDefinition>()

  /// Mapping from Declaration to type of thing it declares.
  private(set) var types = Dictionary<Declaration.Identity, Type>()

  /// Mapping from struct to the parameter tuple type that initializes it.
  private var initializerTuples = ASTDictionary<StructDefinition, TupleType>()

  /// Return type of function currently being checked, if any.
  private var returnType: Type?

  /// A record of the collected errors.
  var errors: ErrorLog = []
}

private extension TypeChecker {
  /// Adds an error at the site of `offender` to the error log, returning
  /// `Type.error` for convenience.
  @discardableResult
  mutating func error<Node: AST>(
    _ offender: Node, _ message: String , notes: [CompileError.Note] = []
  ) -> Type {
    errors.append(CompileError(message, at: offender.site, notes: notes))
    return .error
  }
}

private extension TypeChecker {
  /// Records references from child declarations to their enclosing parents.
  mutating func registerParentage(_ d: TopLevelDeclaration) {
    switch d {
    case let .choice(c):
      for a in c.alternatives { parent[a] = c }
    case .struct, .function, .initialization: ()
    }
  }

  /// Typechecks the body of `d` if it declares a nominal type, recording the
  /// types of any interior declarations in `self.types` (and any errors in
  /// `self.errors`).
  mutating func checkNominalTypeBody(_ d: TopLevelDeclaration) {
    // Note: when nominal types gain methods and/or initializations, we'll need to
    // change the name of this method because those must be checked later.
    switch d {
    case let .struct(s):
      for m in s.members { _ = type(m) }
    case let .choice(c):
      for a in c.alternatives {
        parent[a] = c
        _ = type(a)
      }
    case .function, .initialization: ()
    }
  }

  /// Returns the type defined by `t` or `.error` if `d` doesn't define a type.
  mutating func evaluate(_ e: TypeExpression) -> Type {
    let v = evaluate(e.body)
    if let r = Type(v) { return r }
    return error(e, "Not a type expression (value has type \(v.type)).")
  }

  /// Returns the result of evaluating `e`, logging an error if `e` doesn't
  /// have a value that can be computed at compile-time.
  mutating func evaluate(_ e: Expression) -> Value {
    // Temporarily evaluating the easy subset of type expressions until we have
    // an interpreter.
    switch e {
    case let .name(v):
      if let r = Type(program.definition[v]!) {
        return r
      }
      UNIMPLEMENTED
    case .memberAccess(_):
      UNIMPLEMENTED
    case .index(target: _, offset: _, _):
      UNIMPLEMENTED
    case let .integerLiteral(r, _):
      return r
    case let .booleanLiteral(r, _):
      return r
    case let .tupleLiteral(t):
      return t.fields(reportingDuplicatesIn: &errors)
        .mapValues { self.evaluate($0) }
    case .unaryOperator(operation: _, operand: _, _):
      UNIMPLEMENTED
    case .binaryOperator(operation: _, lhs: _, rhs: _, _):
      UNIMPLEMENTED
    case .functionCall(_):
      UNIMPLEMENTED
    case .intType:
      return Type.int
    case .boolType:
      return Type.bool
    case .typeType:
      return Type.type
    case let .functionType(f):
      // Evaluate `f.parameters` as a type expression so we'll get a diagnostic
      // if it isn't a type.
      let p = evaluate(TypeExpression(f.parameters)).tuple!
      return Type.function(
        parameterTypes: p, returnType: evaluate(f.returnType))
    }
  }

  /// Registers `t` as the type declared by `d`, returning `t`
  mutating func memoizedType(of d: Declaration, _ t: Type) -> Type {
    types[d.identity] = t
    return t
  }

  /// Returns the type of the entity declared by `d`.
  ///
  /// - Requires: if `d` declares a binding, its type has already been memoized
  ///   or is declared as a type expression rather than with `auto`.
  mutating func type(_ d: Declaration) -> Type {
    if let r = types[d.identity] { return r }

    let r: Type
    switch d {
    case is TypeDeclaration:
      r = .type

    case let x as SimpleBinding:
      r = evaluate(x.type.expression!)

    case let x as FunctionDefinition:
      r = type(x)

    case let a as Alternative:
      let payload = evaluate(TypeExpression(a.payload))
      let payloadTuple = payload == .error ? .void : payload.tuple!
      r = .alternative(
        parent: ASTIdentity(of: parent[a]!), payload: payloadTuple)

    case let x as StructMember:
      r = evaluate(x.type)

    default: UNREACHABLE // All possible cases should be handled.
    }
    return memoizedType(of: d, r)
  }

  /// Returns the type of the value computed by `e`, logging errors if `e`
  /// doesn't typecheck.
  mutating func type(_ e: Expression) -> Type {
    switch e {
    case .name(let v):
      return type(program.definition[v]!)

    case let .functionType(f):
      // PARTIALLY UNIMPLEMENTED
      // _ = type(FunctionType<Pattern>(f))

      return .type

    case .intType, .boolType, .typeType:
      return .type

    case .memberAccess(let e):
      return type(e)

    case .index(target: _, offset: _, _):
      UNIMPLEMENTED

    case .integerLiteral:
      return .int

    case .booleanLiteral:
      return .bool

    case let .tupleLiteral(t):
      return .tuple(
        t.fields(reportingDuplicatesIn: &errors).mapValues { type($0) })

    case .unaryOperator(operation: _, operand: _, _):
      UNIMPLEMENTED

    case .binaryOperator(operation: _, lhs: _, rhs: _, _):
      UNIMPLEMENTED

    case .functionCall(let f):
      return type(f)
    }
  }

  /// Returns the type of the value computed by `e`, logging errors if `e`
  /// doesn't typecheck.
  mutating func type(_ e: FunctionCall<Expression>) -> Type {
    let calleeType = type(e.callee)
    let argumentTypes = type(.tupleLiteral(e.arguments))
    switch calleeType {
    case let .function(parameterTypes: p, returnType: r):
      if argumentTypes != .tuple(p) {
        error(
          e.arguments,
          "argument types \(argumentTypes) do not match parameter types \(p)")
      }
      return r

    case let .alternative(parent: resultID, payload: payload):
      if argumentTypes != .tuple(payload) {
        error(
          e.arguments,
          "argument types \(argumentTypes)"
            + " do not match payload type \(payload)")
      }
      return .choice(resultID)

    case .type:
      let calleeValue = evaluate(TypeExpression(e.callee))
      guard case .struct(let s) = calleeValue else {
        return error(e.callee, "type \(calleeValue) is not callable.")
      }

      let initializerType = initializerParameters(s)

      if argumentTypes != .tuple(initializerType) {
        error(
          e.arguments,
          "argument types \(argumentTypes) do"
            + " not match required initializer parameters \(initializerType)")
      }
      return calleeValue

    default:
      return error(e.callee, "value of type \(calleeType) is not callable.")
    }
  }

  mutating func type(_ e: MemberAccessExpression) -> Type {
    let baseType = type(e.base)

    switch baseType {
    case let .struct(baseID):
      let s = baseID.structure
      if let m = s.members.first(where: { $0.name == e.member }) {
        return type(m)
      }
      return error(e.member, "struct \(s.name) has no member '\(e.member.text)'")

    case let .tuple(t):
      if let r = t[e.member] { return r }
      return error(e.member, "tuple type \(t) has no field '\(e.member.text)'")

    case .type:
      // Handle access to a type member, like a static member in C++.
      if case let .choice(id) = evaluate(TypeExpression(e.base)) {
        let c: ChoiceDefinition = id.structure
        return c.alternatives
          .first(where: { $0.name == e.member }).map { type($0) }
          ?? error(
            e.member,
            "choice \(c.name.text) has no alternative '\(e.member.text)'")
      }
      // No other types have members.
      fallthrough
    default:
      return error(
        e.base, "expression of type \(baseType) does not have named members")
    }
  }

  mutating func type(_ f: FunctionDefinition) -> Type {
    if let r = types[f.identity] { return r }
    let parameterTypes = self.parameterTypes(f.parameters)

    let returnType: Type
    if case .expression(let t) = f.returnType {
      returnType = evaluate(t)
    }
    else if case .some(.return(let e, _)) = f.body {
      returnType = type(e)
    }
    else { UNREACHABLE } // auto return type without return statement body(?)

    let r = Type.function(parameterTypes: parameterTypes, returnType: returnType)
    types[f.identity] = r
    return r
  }

  mutating func parameterTypes(_ p: TuplePattern) -> TupleType {
    return p.fields(reportingDuplicatesIn: &errors).mapValues { type($0) }
  }

  mutating func type(_ p: Pattern) -> Type {
    switch p {
    case let .atom(e):
      return type(e)
    case let .variable(v):
      return evaluate(v.type.expression!)
    case let .tuple(t):
      return .tuple(
        t.fields(reportingDuplicatesIn: &errors).mapValues { type($0) })
    case let .functionCall(c):
      return type(c)
    case let .functionType(f):
      return type(f)
    }
  }

  // FIXME: There is significant code duplication between pattern and expression
  // type deduction.  Perhaps upgrade expressions to patterns and use the same
  // code to check?

  mutating func type(_ p: FunctionCall<Pattern>) -> Type {
    // Because p is a pattern, it must be a destructurable thing containing
    // bindings, which means the callee can only be a choice alternative
    // or struct type.
    let calleeType = type(p.callee)
    let argumentTypes = type(.tuple(p.arguments)).tuple!

    switch calleeType {
    case .type:
      let calleeValue = Type(evaluate(p.callee))!

      guard case .struct(let resultID) = calleeValue else {
        return error(
          p.callee, "Called type must be a struct, not '\(calleeValue)'.")
      }

      let parameterTypes = initializerParameters(resultID)
      if argumentTypes != parameterTypes {
        error(
          p.arguments,
          "Argument tuple type \(argumentTypes) doesn't match"
            + " struct initializer type \(parameterTypes)")
      }
      return calleeValue

    case let .alternative(parent: resultID, payload: payload):
      if argumentTypes != payload {
        error(
          p.arguments,
          "Argument tuple type \(argumentTypes) doesn't match"
            + " alternative payload type \(payload)")
      }
      return .choice(resultID)

    default:
      return error(p.callee, "instance of type \(calleeType) is not callable.")
    }
  }

  /// Returns the initializer parameter list for the given struct
  mutating func initializerParameters(
    _ s: ASTIdentity<StructDefinition>
  ) -> TupleType {
    if let r = initializerTuples[s.structure] { return r }
    let r = s.structure.initializerTuple.fields(reportingDuplicatesIn: &errors)
      .mapValues { evaluate($0) }
    initializerTuples[s.structure] = r
    return r
  }

  mutating func type(_ t: FunctionType<Pattern>) -> Type {
    UNIMPLEMENTED
    /*
    if Type(type(t.parameters)) == nil {
      return error(
        t.parameters, "function type parameter list is not a tuple of types")
    }
    if Type(type(t.returnType)) == nil {
      return error(
        t.returnType, "function type return type is not a type.")
    }
    return .type

     */
  }
}
