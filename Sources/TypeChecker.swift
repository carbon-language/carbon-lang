// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// The type-checking algorithm and associated data.
struct TypeChecker {
  /// Type-checks `parsedProgram` given its name resolution results.
  init(_ parsedProgram: AbstractSyntaxTree, nameLookup: NameResolution) {
    self.definition = nameLookup.definition

    // Create "external parent links" for the AST in our parentXXX properties.
    for d in parsedProgram { registerParentage(in: d) }

    // Check the bodies of nominal types.
    for d in parsedProgram { checkNominalTypeBody(d) }

    // Check function signatures.
    for d in parsedProgram {
      if case .function(let f) = d {
        _ = typeOfName(declaredBy: f as Declaration)
      }
    }
    // Check top-level initializations
    for d in parsedProgram {
      if case .initialization(let i) = d { check(i) }
    }
    // Check function bodies.
    for d in parsedProgram {
      if case .function(let f) = d { checkBody(f) }
    }
  }


  // Mapping from names to their definitions
  private let definition: ASTDictionary<Identifier, Declaration>

  /// The static type of each expression
  private(set) var expressionType = ASTDictionary<Expression, Type>()

  /// Mapping from alternative declaration to the choice in which it is defined.
  private(set) var enclosingChoice
    = ASTDictionary<Alternative, ChoiceDefinition>()

  /// Mapping from variable declaration to the initialization in which it is
  /// defined.
  private var enclosingInitialization
    = ASTDictionary<SimpleBinding, Initialization>()

  /// Memoized result of computing the type of the expression consisting of the
  /// name of each declared entity.
  private(set) var typeOfNameDeclaredBy
    = Dictionary<Declaration.Identity, Memo<Type>>()

  /// The payload tuple type for each alternative.
  private(set) var payloadType: [ASTIdentity<Alternative>: TupleType] = [:]

  /// The set of initializations that have been completely typechecked.
  private var checkedInitializations = Set<Initialization.Identity>()

  /// Mapping from struct to the parameter tuple type that initializes it.
  private var initializerTuples = ASTDictionary<StructDefinition, TupleType>()

  /// Return type of function currently being checked, if any.
  private var expectedReturnType: Type? = nil

  /// True iff currently checking the body of a loop.
  private var inLoop: Bool = false

  /// True iff tracing output
  public var tracing: Bool = false

  /// A record of the errors encountered during type-checking.
  var errors: ErrorLog = []
}

/// Diagnostic utilities.
private extension TypeChecker {
  /// Adds an error at the site of `offender` to the error log, returning
  /// `Type.error` for convenience.
  @discardableResult
  mutating func error<Node: AST>(
    _ offender: Node, _ message: String , notes: [CarbonError.Note] = []
  ) -> Type {
    let e = CarbonError(message, at: offender.site, notes: notes)
    if tracing { print(e) }
    errors.append(e)
    return .error
  }

  /// Logs an error pointing at `source` unless `t` is a metatype.
  mutating func expectMetatype<Node: AST>(_ t: Type, at source: Node) {
    if !t.isMetatype {
      error(
        source,
        "Pattern in this context must match type values, not \(t) values")
    }
  }

  /// Logs an error unless the type of `e` is `t`.
  mutating func expectType(of e: Expression, toBe expected: Type) {
    let actual = type(e)
    if actual != expected {
      error(e, "Expected expression of type \(expected), not \(actual).")
    }
  }
}

/// Computation of notional “ancestor node links” for the AST.
private extension TypeChecker {
  /// Records references from child declarations to their enclosing parents.
  mutating func registerParentage(in d: TopLevelDeclaration) {
    switch d {
    case let .choice(c):
      for a in c.alternatives { enclosingChoice[a] = c }
    case let .initialization(i):
      registerParent(in: i.bindings, as: i)
    case .struct, .function: ()
    }
  }

  /// Records references from variable declarations in children to the given
  /// parent initialization.
  mutating func registerParent(in children: Pattern, as parent: Initialization) {
    switch children {
    case .atom: return
    case let .variable(x): enclosingInitialization[x] = parent
    case let .tuple(x):
      for a in x { registerParent(in: a.payload, as: parent) }
    case let .functionCall(x):
      for a in x.arguments { registerParent(in: a.payload, as: parent) }
    case let .functionType(x):
      for a in x.parameters { registerParent(in: a.payload, as: parent) }
      registerParent(in: x.returnType, as: parent)
    }
  }
}

private extension TypeChecker {
  /// Typechecks the body of `d` if it declares a nominal type, recording the
  /// types of any interior declarations in `self.types` (and any errors in
  /// `self.errors`).
  mutating func checkNominalTypeBody(_ d: TopLevelDeclaration) {
    // Note: when nominal types gain methods and/or initializations, we'll need
    // to change the name of this method because those must be checked later.
    switch d {
    case let .struct(s):
      for m in s.members { _ = typeOfName(declaredBy: m) }
    case let .choice(c):
      for a in c.alternatives { _ = typeOfName(declaredBy: a) }
    case .function, .initialization: break
    }
  }
}

/// Compile-time expression evaluation.
private extension TypeChecker {
  /// Type-checks `e` and returns its compile-time type value, or `Type.error`
  /// if `e` does not describe a type that can be computed at compile-time.
  mutating func value(_ e: TypeExpression) -> Type {
    let t = type(e.body)
    if !t.isMetatype {
      return error(e, "Not a type expression (value has type \(t))")
    }
    return Type(value(e.body, checkType: false))!
  }

  /// Returns the compile-time value of `e` (typechecking it too iff `checkType
  /// == true`) or logs an error if `e` can't be evaluated at compile-time.
  mutating func value(_ e: Expression, checkType: Bool = true) -> Value {
    if checkType { _ = type(e) }

    // Temporarily evaluating the easy subset of type expressions until we have
    // an interpreter.
    switch e {
    case let .name(v):
      if let r = Type(definition[v]!) {
        return r
      }
      UNIMPLEMENTED()
    case .memberAccess(_):
      UNIMPLEMENTED()
    case .index(target: _, offset: _, _):
      UNIMPLEMENTED()
    case let .integerLiteral(r, _):
      return r
    case let .booleanLiteral(r, _):
      return r
    case let .tupleLiteral(t):
      return t.fields(reportingDuplicatesIn: &errors).mapFields { value($0) }
    case .unaryOperator(_):
      UNIMPLEMENTED()
    case .binaryOperator(_):
      UNIMPLEMENTED()
    case .functionCall(_):
      UNIMPLEMENTED()
    case .intType:
      return Type.int
    case .boolType:
      return Type.bool
    case .typeType:
      return Type.type
    case let .functionType(f):
      // Evaluate `f.parameters` as a type expression so we'll get a diagnostic
      // if it isn't a type.  Fall back to void if the result was `Type.error`.
      let p = value(TypeExpression(f.parameters)).tuple ?? .void
      return Type.function(
        .init(parameterTypes: p, returnType: value(f.returnType)))
    }
  }
}

private extension TypeChecker {
  /// Returns the type of the entity declared by `d`.
  ///
  /// - Requires: if `d` declares a binding, its type has already been memoized
  ///   or is declared as a type expression rather than with `auto`.
  mutating func typeOfName(declaredBy d: Declaration) -> Type {
    // Check the memo
    switch typeOfNameDeclaredBy[d.identity] {
    case .beingComputed:
      return error(d.name, "type dependency loop")
    case let .final(t):
      return t
    case nil: ()
    }
    if tracing { print("\(d.site): info: \(#function)") }
    typeOfNameDeclaredBy[d.identity] = .beingComputed

    let r: Type
    switch d { // Initialize r.
    case is TypeDeclaration:
      r = .type

    case let x as SimpleBinding:
      if let e = x.type.expression {
        r = value(e)
      }
      else {
        check(enclosingInitialization[x]!)
        if case let .final(r0) = typeOfNameDeclaredBy[d.identity]! { r = r0 }
        else { UNREACHABLE() }
      }

    case let x as FunctionDefinition:
      r = typeOfName(declaredBy: x)

    case let a as Alternative:
      let payload = value(TypeExpression(a.payload))
      payloadType[a.identity] = payload == .error ? .void : payload.tuple!
      r = .alternative(a.identity)

    case let x as StructMember:
      r = value(x.type)

    default: UNREACHABLE()
    }

    // memoize the result.
    typeOfNameDeclaredBy[d.identity] = .final(r)
    if tracing { print("\(d.site): info: \(#function) = \(r)") }
    return r
  }

  /// Returns the type of the function declared by `f`, logging any errors in
  /// its signature, and, if `f` was declared with `=>`, in its body expression.
  mutating func typeOfName(declaredBy f: FunctionDefinition) -> Type {
    // Don't bypass memoization in case we are called directly.
    if typeOfNameDeclaredBy[f.identity] != .beingComputed {
      return typeOfName(declaredBy: f as Declaration)
    }

    let parameterTypes = f.parameters.fields(reportingDuplicatesIn: &errors)
      .mapFields { patternType($0) }

    let returnType: Type
    if case .expression(let t) = f.returnType {
      returnType = value(t)
    }
    else if case .some(.return(let e, _)) = f.body {
      returnType = type(e)
    }
    else { UNREACHABLE("auto return type without return statement body") }

    return .function(
      .init(parameterTypes: parameterTypes, returnType: returnType))
  }
}

/// Expression checking.
private extension TypeChecker {
  /// Returns the initializer parameter list for the given struct
  mutating func initializerParameters(
    _ s: ASTIdentity<StructDefinition>
  ) -> TupleType {
    if let r = initializerTuples[s.structure] { return r }
    let r = s.structure.initializerTuple.fields(reportingDuplicatesIn: &errors)
      .mapFields { value($0) }
    initializerTuples[s.structure] = r
    return r
  }

  /// Returns the type of the value computed by `e`, logging errors if `e`
  /// doesn't typecheck.
  ///
  /// - Parameter isCallee: indicates `e` is being evaluated in callee position
  ///   of a function call expression.
  mutating func type(_ e: Expression, isCallee: Bool = false) -> Type {
    if let r = expressionType[e] { return r }
    if tracing { print("\(e.site): info: type") }

    func rawResult() -> Type {
      switch e {
      case .name(let v):
        return typeOfName(declaredBy: definition[v]!)

      case let .functionType(f):
        let p = value(TypeExpression(f.parameters))
        if p != .error { assert(p.tuple != nil) }
        _ = value(f.returnType)
        return .type


      case let .index(target: base, offset: index, _):
        let baseType = type(base)
        guard case .tuple(let types) = baseType else {
          return error(base, "Can't index non-tuple type \(baseType)")
        }
        let indexType = type(index)
        guard indexType == .int else {
          return error(index, "Index type must be Int, not \(indexType)")
        }
        let indexValue = value(index) as! Int
        return types[indexValue] ?? error(
          index, "Tuple type \(types) has no value at position \(indexValue)")

      case let .tupleLiteral(t):
        return .tuple(
          t.fields(reportingDuplicatesIn: &errors).mapFields { type($0) })

      case .intType, .boolType, .typeType: return .type
      case .memberAccess(let e): return type(e)
      case .integerLiteral: return .int
      case .booleanLiteral: return .bool
      case let .unaryOperator(u): return type(u)
      case let .binaryOperator(b): return type(b)
      case .functionCall(let f): return type(f)
      }
    }

    var r  = rawResult()
    // Adjust the result: nullary alternative types are implicitly converted to
    // their parent `choice` type unless they are in callee position.
    if !isCallee, case let .alternative(a) = r, a.structure.payload.isEmpty
    {
      r = .choice(enclosingChoice[a.structure]!.identity)
    }

    if tracing { print("\(e.site): info: type = \(r)") }
    expressionType[e] = r
    return r
  }

  mutating func type(_ u: UnaryOperatorExpression) -> Type {
    switch u.operation.kind {
    case .MINUS:
      expectType(of: u.operand, toBe: .int)
      return .int
    case .NOT:
      expectType(of: u.operand, toBe: .bool)
      return .bool
    default:
      UNREACHABLE(u.operation.text)
    }
  }
  
  mutating func type(_ b: BinaryOperatorExpression) -> Type {
    switch b.operation.kind {
    case .EQUAL_EQUAL:
      expectType(of: b.rhs, toBe: type(b.lhs))
      return .bool

    case .PLUS, .MINUS:
      expectType(of: b.lhs, toBe: .int)
      expectType(of: b.rhs, toBe: .int)
      return .int

    case .AND, .OR:
      expectType(of: b.lhs, toBe: .bool)
      expectType(of: b.rhs, toBe: .bool)
      return .bool

    default:
      UNREACHABLE(b.operation.text)
    }
  }

  /// Returns the type of the value computed by `e`, logging errors if `e`
  /// doesn't typecheck.
  mutating func type(_ e: FunctionCall<Expression>) -> Type {
    let callee = type(e.callee, isCallee: true)
    let argumentTypes = type(.tupleLiteral(e.arguments))

    switch callee {
    case let .function(f):
      if argumentTypes != .tuple(f.parameterTypes) {
        error(
          e.arguments,
          "argument types \(argumentTypes)"
            + " do not match parameter types \(f.parameterTypes)")
      }
      return f.returnType

    case let .alternative(a):
      let payload = payloadType[a]!
      if argumentTypes != .tuple(payload) {
        error(
          e.arguments, "argument types \(argumentTypes)"
            + " do not match payload type \(payload)")
      }
      return .choice(enclosingChoice[a.structure]!.identity)

    case .type:
      let calleeValue = value(TypeExpression(e.callee))
      guard case .struct(let s) = calleeValue else {
        return error(e.callee, "type \(calleeValue) is not callable.")
      }

      let initializerType = initializerParameters(s)

      if argumentTypes != .tuple(initializerType) {
        error(
          e.arguments, "argument types \(argumentTypes) do not match"
            + " required initializer parameters \(initializerType)")
      }
      return calleeValue

    default:
      return error(e.callee, "value of type \(callee) is not callable.")
    }
  }

  mutating func type(_ e: MemberAccessExpression) -> Type {
    let baseType = type(e.base)

    switch baseType {
    case let .struct(baseID):
      let s = baseID.structure
      if let m = s.members.first(where: { $0.name == e.member }) {
        return typeOfName(declaredBy: m)
      }
      return error(e.member, "struct \(s.name) has no member '\(e.member)'")

    case let .tuple(t):
      return t[e.member]
        ?? error(e.member, "tuple type \(t) has no field '\(e.member)'")

    case .type:
      // Handle access to a type member, like a static member in C++.
      if case let .choice(id) = value(TypeExpression(e.base)) {
        let c: ChoiceDefinition = id.structure
        return c[e.member].map { typeOfName(declaredBy: $0) }
          ?? error(
            e.member, "choice \(c.name) has no alternative '\(e.member)'")
      }
      // No other types have members.
      fallthrough
    default:
      return error(
        e.base, "expression of type \(baseType) does not have named members")
    }
  }
}

// FIXME: There is significant code duplication between pattern and expression
// type computation.  Perhaps upgrade expressions to patterns and use the same
// code to check?


/// Pattern checking and type deduction.
private extension TypeChecker {
  /// Returns the type matched by `p`, using `rhs`, if supplied, to deduce `auto`
  /// types, and logging any errors.
  ///
  /// - Note: DOES NOT verify that `rhs` is a subtype of the result; you must
  ///   check that separately.
  mutating func patternType(
    _ p: Pattern, initializerType rhs: Type? = nil) -> Type
  {
    if tracing { print("\(p.site): info: pattern type") }

    switch (p) {
    case let .atom(e):
      return type(e)

    case let .variable(binding):
      let r = binding.type.expression.map { value($0) }
        ?? rhs ?? error(
          binding.type, "No initializer available to deduce type for auto")
      typeOfNameDeclaredBy[binding.identity] = .final(r)
      return r
      // Hack for metatype subtyping---replace with real subtyping.
      // return rhs.map { r == .type && $0.isMetatype ? $0 : r } ?? r
      
    case let .tuple(t):
      return .tuple(
        t.fields(reportingDuplicatesIn: &errors).mapElements { (id, f) in
          patternType(f, initializerType: rhs?.tuple?.elements[id])
        })

    case let .functionCall(c):
      return patternType(c, initializerType: rhs)

    case let .functionType(f):
      return patternType(f, initializerType: rhs?.function)
    }
  }

  /// Returns the type matched by `p`, using `rhs`, if supplied, to deduce `auto`
  /// types, and logging any errors.
  ///
  /// - Note: DOES NOT verify that `rhs` is a subtype of the result; you must
  ///   check that separately.
  mutating func patternType(
    _ p: TupleSyntax<Pattern>, initializerType rhs: TupleType?,
    requireMetatype: Bool = false
  ) -> Type {
    return .tuple(
      p.fields(reportingDuplicatesIn: &errors).mapElements { (id, f) in
        let t = patternType(f, initializerType: rhs?.elements[id])
        if requireMetatype { expectMetatype(t, at: f) }
        return t
      })
  }

  /// Returns the type matched by `p`, using `rhs`, if supplied, to deduce `auto`
  /// types, and logging any errors.
  ///
  /// - Note: DOES NOT verify that `rhs` is a subtype of the result; you must
  ///   check that separately.
  mutating func patternType(
    _ p: FunctionCall<Pattern>, initializerType rhs: Type?) -> Type
  {
    // Because p is a pattern, it must be a destructurable thing containing
    // bindings, which means the callee can only be a choice alternative
    // or struct type.
    let calleeType = type(p.callee)

    switch calleeType {
    case .type:
      let calleeValue = Type(value(p.callee))!

      guard case .struct(let resultID) = calleeValue else {
        return error(
          p.callee, "Called type must be a struct, not '\(calleeValue)'.")
      }
      let parameterTypes = initializerParameters(resultID)
      let argumentTypes = patternType(
        p.arguments, initializerType: parameterTypes).tuple!

      if argumentTypes != parameterTypes {
        error(
          p.arguments,
          "Argument tuple type \(argumentTypes) doesn't match"
            + " struct initializer type \(parameterTypes)")
      }
      return calleeValue

    case let .alternative(a):
      let payload = payloadType[a]!
      let argumentTypes = patternType(
        p.arguments, initializerType: payload).tuple!
      if argumentTypes != payload {
        error(
          p.arguments,
          "Argument tuple type \(argumentTypes) doesn't match"
            + " alternative payload type \(payload)")
      }
      return .choice(enclosingChoice[a.structure]!.identity)

    default:
      return error(p.callee, "instance of type \(calleeType) is not callable.")
    }
  }

  /// Returns the type matched by `p`, using `rhs`, if supplied, to deduce `auto`
  /// types, and logging any errors.
  ///
  /// - Note: DOES NOT verify that `rhs` is a subtype of the result; you must
  ///   check that separately.
  mutating func patternType(
    _ t: FunctionTypeSyntax<Pattern>, initializerType rhs: FunctionType?
  ) -> Type {
    _ = patternType(
      t.parameters, initializerType: rhs?.parameterTypes,
      requireMetatype: true)

    let r = patternType(t.returnType, initializerType: rhs?.returnType)
    expectMetatype(r, at: t.returnType)
    return .type
  }
}

/// Statement and top-level initialization checking.
private extension TypeChecker {
  /// pass `self` to action with `inLoop` temporarily set to `newValue`
  mutating func withInLoop<R>(
    setTo newValue: Bool, _ action: (inout Self)->R) -> R
  {
    let saved = inLoop
    self.inLoop = newValue
    defer { self.inLoop = saved }

    return action(&self)
  }

  /// Typechecks the body of `f`.
  mutating func checkBody(_ f: FunctionDefinition) {
    // auto-returning bodies have already been checked as part of function
    // signature checking.
    if case .auto = f.returnType { return }

    guard let body = f.body else { return }

    self.expectedReturnType = typeOfName(declaredBy: f).function!.returnType
    withInLoop(setTo: false) { me in
      me.check(body)
    }
  }

  /// Typechecks `s`.
  mutating func check(_ s: Statement) {
    if tracing { print("\(s.site): info: check(_:Statement)") }
    switch s {
    case let .expressionStatement(e, _):
      _ = type(e)

    case let .assignment(target: t, source: s, _):
      // TODO: formalize rvalues vs. lvalues and check that type(t) is an
      // lvalue.
      expectType(of: t, toBe: type(s))

    case let .initialization(i):
      check(i)

    case let .if(c, s0, else: s1, _):
      expectType(of: c, toBe: .bool)
      check(s0)
      if let s1 = s1 { check(s1) }

    case let .return(e, _):
      expectType(of: e, toBe: expectedReturnType!)

    case let .block(statements, _):
      for s in statements {
        check(s)
      }

    case let .while(condition, body, _):
      expectType(of: condition, toBe: .bool)
      withInLoop(setTo: true) { me in
        me.check(body)
      }

    case let .match(subject: subject, clauses: clauses, _):
      let subjectType = type(subject)
      for c in clauses {
        if let nonDefault = c.pattern {
          let p = patternType(nonDefault, initializerType: subjectType)
          if p != subjectType {
            error(
              subject, "Pattern type \(p) incompatible with matched "
                   + "expression type \(subjectType).",
              notes: [("Matched expression", subject.site)])
          }
        }
        check(c.action)
      }

    case .break, .continue:
      if !inLoop {
        error(s, "invalid outside loop body")
      }
    }
  }

  /// Ensures that `i` has been type-checked.
  mutating func check(_ i: Initialization) {
    if checkedInitializations.contains(i.identity) { return }
    defer { checkedInitializations.insert(i.identity) }

    let rhs = type(i.initializer)
    let lhs = patternType(i.bindings, initializerType: rhs)
    if lhs != rhs {
      error(i, "Pattern type \(lhs) does not match initializer type \(rhs).")
    }
  }
}

/// A marker for code that needs to be implemented.  Eventually all of these
/// should be eliminated from the codebase.
func UNIMPLEMENTED(
  _ vars: Any...,
  filePath: StaticString = #filePath,
  line: UInt = #line
) -> Never {
  fatalError(
    "UNIMPLEMENTED\n"
      + vars.lazy.map(String.init(describing:)).joined(separator: "\n"),
    file: (filePath), line: line)
}

/// A marker for code that should never be reached.
func UNREACHABLE(
  _ message: String? = nil,
  filePath: StaticString = #filePath, line: UInt = #line) -> Never {
  fatalError(message ?? "unreachable", file: (filePath), line: line)
}
