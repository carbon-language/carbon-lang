// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

var UNIMPLEMENTED: Never { fatalError("unimplemented") }

/// A marker for code that should never be reached.
var UNREACHABLE: Never { fatalError("unreachable.") }

struct TypeChecker {
  init(_ program: ExecutableProgram) {
    self.program = program

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

  /// Mapping from Declarations to type of thing they declare.
  private(set) var types = Dictionary<Declaration.Identity, Type>()

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
      for a in c.alternatives { _ = type(a) }
    case .function, .initialization: ()
    }
  }

  /// Returns the type defined by `t` or `.error` if `d` doesn't define a type.
  mutating func evaluate(_ e: TypeExpression) -> Type {
    let v = evaluate(e.body)
    if let r = (v as? Type) { return r }

    // If the value is a tuple, check that all its elements are types.
    if let elements = (v as? TupleValue) {
      let typeElements = elements.compactMapValues { $0 as? Type }
      if typeElements.count == elements.count {
        return .tuple(typeElements)
      }
    }

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
    case .getField(_): UNIMPLEMENTED
    case .index(target: _, offset: _, _): UNIMPLEMENTED
    case let .integerLiteral(r, _): return r
    case let .booleanLiteral(r, _): return r
    case let .tupleLiteral(t):
      return t.fields(reportingDuplicatesIn: &errors)
        .mapValues { self.evaluate($0) }
    case .unaryOperator(operation: _, operand: _, _): UNIMPLEMENTED
    case .binaryOperator(operation: _, lhs: _, rhs: _, _): UNIMPLEMENTED
    case .functionCall(_): UNIMPLEMENTED
    case .intType: return Type.int
    case .boolType: return Type.bool
    case .typeType: return Type.type
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
    case let x as TypeDeclaration:  r = x.declaredType
    case let x as SimpleBinding: r = evaluate(x.type.expression!)
    case let x as FunctionDefinition: r = type(x)
    case let x as Alternative: r = evaluate(TypeExpression(x.payload))
    case let x as StructMember: r = evaluate(x.type)
    default: UNREACHABLE // All possible cases should be handled.
    }
    return memoizedType(of: d, r)
  }

  /// Returns the type of the value computed by `e`, logging errors if `e`
  /// doesn't typecheck.
  mutating func type(_ e: Expression) -> Type {
    switch e {
    case .name(let v): return type(program.definition[v]!)

    case .intType, .boolType, .typeType, .functionType:
      return .type

    case .getField(let e): return type(e)

    case .index(target: _, offset: _, _): UNIMPLEMENTED

    case .integerLiteral: return .int
    case .booleanLiteral: return .bool

    case let .tupleLiteral(t):
      return .tuple(
        t.fields(reportingDuplicatesIn: &errors).mapValues { type($0) })

    case .unaryOperator(operation: _, operand: _, _): UNIMPLEMENTED
    case .binaryOperator(operation: _, lhs: _, rhs: _, _): UNIMPLEMENTED

    case .functionCall(let f): return type(f)
    }
  }

  /// Returns the type of the value computed by `e`, logging errors if `e`
  /// doesn't typecheck.
  mutating func type(_ call: FunctionCall<Expression>) -> Type {
    let calleeType = type(call.callee)
    let argumentTypes = type(.tupleLiteral(call.arguments))
    switch calleeType {
    case let .function(parameterTypes: p, returnType: r):
      if argumentTypes != .tuple(p) {
        error(
          call.arguments,
          "argument types \(argumentTypes) do not match parameter types \(p)")
      }
      return r
    default:
      return error(call.callee, "value of type \(calleeType) is not callable.")
    }
  }

  mutating func type(_ access: GetFieldExpression) -> Type {
    let target = access.target
    let fieldName = access.fieldName

    let targetType = type(target)

    switch targetType {
    case let .struct(targetID):
      let s = targetID.structure
      if let m = s.members.first(where: { $0.name == fieldName }) {
        return type(m)
      }
      return error(fieldName, "struct \(s.name) has no field \(fieldName.text)")

    case let .tuple(t):
      if let r = t[.label(fieldName)] { return r }
      return error(fieldName, "tuple type \(t) has no field \(fieldName.text)")

    case .type:
      // See if this is a choice member (or more generally, a member of the
      // type rather than of instances of the type, e.g. static members in
      // C++).
      if case let .choice(id) = evaluate(TypeExpression(target)) {
        let c = id.structure
        if let a = c.alternatives.first(
             where: { $0.name == fieldName }) { return type(a) }
        return error(
          fieldName, "choice \(c.name) has no alternative \(fieldName.text)")
      }
      fallthrough
    default:
      return error(
        target, "expression of type \(targetType) does not have named fields")
    }
  }

  mutating func type(_ f: FunctionDefinition) -> Type {
    if let r = types[f.identity] { return r }
    let parameterTypes = parameterTypes(f.parameters)

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
    case let .atom(e): return type(e)
    case let .variable(v): return evaluate(v.type.expression!)
    case let .tuple(t):
      return .tuple(
        t.fields(reportingDuplicatesIn: &errors).mapValues { type($0) })
    case let .functionCall(c):  return type(c)
    case let .functionType(f): return type(f)
    }
  }

  mutating func type(_ p: FunctionCall<Pattern>) -> Type {
    // Because p is a pattern, it must be a destructurable thing containing
    // bindings, which means the callee can only be a choice alternative or
    // struct type.
    UNIMPLEMENTED
  }

  mutating func type(_ c: FunctionType<Pattern>) -> Type {
    UNIMPLEMENTED
  }
}
