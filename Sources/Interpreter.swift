// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A notional “self-returning function” type.
///
/// Swift doesn't allow that function type to be declared directly, but we can
/// indirect through this struct.
struct Task {
  /// A type representing the underlying implementation.
  typealias Code = (inout Interpreter)->Task

  /// Creates an instance with the semantics of `implementation`.
  init(_ code: @escaping Code) {
    self.code = code
  }

  /// Executes `self` in `context`.
  func callAsFunction(_ context: inout Interpreter) -> Task {
    code(&context)
  }

  /// The underlying implementation.
  let code: Code
}

/// All the data that needs to be saved and restored across function call
/// boundaries.
struct CallFrame {
  /// The locations of temporaries that persist to the ends of their scopes and
  /// thus can't be cleaned up in the course of expression evaluation.
  var persistentAllocations: Stack<Address> = .init()

  /// The set of all allocated temporary addresses, with an associated
  /// expression tag for diagnostic purposes.
  ///
  /// Used to ensure we don't forget to clean up temporaries when they are no
  /// longer used.
  var ephemeralAllocations: [Address: Expression] = [:]

  /// A mapping from local bindings to addresses.
  var locals: ASTDictionary<SimpleBinding, Address> = .init()

  /// The place where the result of this call is to be written.
  var resultAddress: Address

  /// The code to execute when this call exits
  var onReturn: Task

  /// The code to execute when the current loop exits.
  var onBreak: Task? = nil

  /// Code that returns to the top of the current loop, if any.
  var onContinue: Task? = nil
}

/// The engine that executes the program.
struct Interpreter {
  /// The program being executed.
  let program: ExecutableProgram

  /// The frame for the current function.
  var frame: CallFrame

  /// Mapping from global bindings to address.
  var globals: ASTDictionary<SimpleBinding, Address> = .init()

  /// Storage for all addressable values.
  var memory = Memory()

  /// The next execution step.
  var nextStep: Task

  /// True iff the program is still running.
  var running: Bool = true

  /// A record of any errors encountered.
  var errors: ErrorLog = []

  /// True iff we are printing an evaluation trace to stdout
  var tracing: Bool = false

  /// Creates an instance that runs `p`.
  ///
  /// - Requires: `p.main != nil`
  init(_ p: ExecutableProgram) {
    self.program = p

    frame = CallFrame(
      resultAddress: memory.allocate(),
      onReturn: Task { me in me.terminate() })

    // First step runs the body of `main`
    nextStep = Task { [main = program.main!] me in
      me.run(main.body!, then: me.frame.onReturn)
    }
  }

  /// Advances execution by one unit of work, returning `true` iff the program
  /// is still running and `false` otherwise.
  mutating func step() -> Bool {
    if running {
      nextStep = nextStep(&self)
    }
    return running
  }

  /// Runs the program to completion and returns the exit code, if any.
  mutating func run() -> Int? {
    while step() {}
    return memory.value(at: frame.resultAddress) as? Int
  }

  /// Exits the running program.
  mutating func terminate() -> Task {
    running = false
    return Task { _ in fatalError("Terminated program can't continue.") }
  }

  /// Adds an error at the site of `offender` to the error log and marks the
  /// program as terminated.
  ///
  /// Returns a non-executable task for convenience.
  @discardableResult
  mutating func error<Node: AST>(
    _ offender: Node, _ message: String , notes: [CarbonError.Note] = []
  ) -> Task {
    errors.append(CarbonError(message, at: offender.site, notes: notes))
    return terminate()
  }

  /// Accesses the value in `memory` at `a`, or halts the interpreted program
  /// with an error if `a` is not an initialized address, returning Type.error.
  subscript(a: Address) -> Value {
    memory[a]
  }
}

extension Interpreter {
  /// Executes `s`, and then, absent interesting control flow,
  /// `followup`.
  ///
  /// An example of interesting control flow is a return statement, which
  /// ignores any `followup` and exits the current function instead.
  ///
  /// In fact this function only executes one “unit of work” and packages the
  /// rest of executing `s` (if any), and whatever follows that, into the
  /// returned `Task`.
  mutating func run(_ s: Statement, then followup: Task) -> Task {
    sanityCheck(
      frame.ephemeralAllocations.isEmpty,
      "leaked \(frame.ephemeralAllocations)")

    if tracing {
      print("\(s.site): info: running statement")
    }
    switch s {
    case let .expressionStatement(e, _):
      UNIMPLEMENTED(e)

    case let .assignment(target: t, source: s, _):
      return evaluate(t) { target, me in
        sanityCheck(
          me.frame.ephemeralAllocations.isEmpty, "\(t) not an lvalue?")
        // Can't evaluate source into target because target may be referenced in
        // source (e.g. x = x - 1)
        return me.evaluate(s) { source, me in
        if me.tracing {
          print(
            "\(t.site): info: assigning \(me[source])\(source) into \(target)")
        }
        return me.assign(target, from: source) { me in
        me.deleteAnyEphemeral(at: source, then: followup.code)
        }}}

    case let .initialization(i):
      // Storage must be allocated for the initializer value even if it's an
      // lvalue, so the vars bound to it have distinct values.  Because vars
      // will be bound to parts of the initializer and are mutable, it must
      // persist through the current scope.
      return allocate(i.initializer, mutable: true, persist: true) { rhs, me in
        me.evaluate(i.initializer, into: rhs) { rhs, me in
          me.match(i.bindings, toValueAt: rhs) { matched, me in
            matched ? followup : me.error(
              i.bindings, "Initialization pattern not matched by \(me[rhs])")
          }
        }
      }

    case let .if(
           condition: c, thenClause: trueClause, elseClause: falseClause, _):
      return evaluateAndConsume(c) { (condition: Bool, me) in
        return condition
          ? me.run(trueClause, then: followup)
          : falseClause.map { me.run($0, then: followup) } ?? followup
      }

    case let .return(e, _):
      return evaluate(e, into: frame.resultAddress) { _, me in
        me.frame.onReturn(&me)
      }

    case let .block(children, _):
      return inScope(then: followup) { me, rest in
        me.runBlock(children[...], then: rest)
      }

    case let .while(condition: c, body: body, _):
      // TODO: put a scope around the while body.
      let saved = (frame.onBreak, frame.onContinue)

      let onBreak = Task { me in
        (me.frame.onBreak, me.frame.onContinue) = saved
        return followup
      }
      let onContinue = Task { $0.while(c, run: body, then: onBreak) }

      (frame.onBreak, frame.onContinue) = (onBreak, onContinue)
      return onContinue(&self)

    case let .match(subject: s, clauses: clauses, _):
      return inScope(then: followup) { me, innerFollowup in
        me.allocate(s, persist: true) { subject, me in
        me.evaluate(s, into: subject) { subject, me in
        me.runMatch(s, at: subject, against: clauses[...], then: innerFollowup)
      }}}

    case .break:
      return frame.onBreak!

    case .continue:
      return frame.onContinue!
    }
  }

  mutating func inScope(
    then followup: Task, do body: (inout Self, Task)->Task
  ) -> Task {
    let mark=frame.persistentAllocations.count
    return body(
      &self,
      Task { me in
        sanityCheck(me.frame.ephemeralAllocations.isEmpty,
               "leaked \(me.frame.ephemeralAllocations)")
        return me.cleanUpPersistentAllocations(above: mark, then: followup)
      })
  }

  /// Runs `s` and follows up with `followup`.
  ///
  /// A convenience wrapper for `run(_:then:)` that supports cleaner syntax.
  mutating func run(_ s: Statement, followup: @escaping Task.Code) -> Task {
    run(s, then: Task(followup))
  }

  /// Executes the statements of `content` in order, then `followup`.
  mutating func runBlock(_ content: ArraySlice<Statement>, then followup: Task)
    -> Task
  {
    return content.isEmpty ? followup
      : run(content.first!) { me in
          me.runBlock(content.dropFirst(), then: followup)
        }
  }

  mutating func runMatch(
    _ subject: Expression, at subjectLocation: Address,
    against clauses: ArraySlice<MatchClause>,
    then followup: Task) -> Task
  {
    guard let clause = clauses.first else {
      return error(subject, "no pattern matches \(self[subjectLocation])")
    }

    let onMatch = Task { me in
      me.inScope(then: followup) { me, innerFollowup in
        me.run(clause.action, then: innerFollowup)
      }
    }
    guard let p = clause.pattern else { return onMatch }

    return match(p, toValueAt: subjectLocation) { matched, me in
      matched ? onMatch : me.runMatch(
        subject, at: subjectLocation, against: clauses.dropFirst(),
        then: followup)
    }
  }

  mutating func `while`(
    _ c: Expression, run body: Statement, then followup: Task) -> Task
  {
    return evaluateAndConsume(c) { (runBody: Bool, me) in
      return runBody
        ? me.run(
          body, then: Task { me in me.while(c, run: body, then: followup)})
        : followup
    }
  }
}

/// Values and memory.
extension Interpreter {
  /// Allocates an address for the result of evaluating `e`, passing it on to
  /// `followup` along with `self`.
  mutating func allocate(
    _ e: Expression, mutable: Bool = false, persist: Bool = false,
    then followup: @escaping FollowupWith<Address>
  ) -> Task {
    let a = memory.allocate(mutable: mutable)
    if tracing {
      print(
        "\(e.site): info: allocated \(a)"
          + " (\(persist ? "persistent" : "ephemeral"))")
    }
    if persist {
      frame.persistentAllocations.push(a)
    }
    else {
      frame.ephemeralAllocations[a] = e
    }
    return Task { me in followup(a, &me) }
  }

  /// Allocates an address for the result of evaluating `e`, passing it on to
  /// `followup` along with `self`.
  mutating func allocate(
    _ e: Expression, unlessNonNil destination: Address?,
    then followup: @escaping FollowupWith<Address>
  ) -> Task {
    if let a = destination { return Task { me in followup(a, &me) } }
    return allocate(e, then: followup)
  }

  /// Destroys and reclaims memory of the `n` locally-allocated values at the
  /// top of the allocation stack.
  mutating func cleanUpPersistentAllocations(
    above n: Int, then followup: Task
  ) -> Task {
    frame.persistentAllocations.count == n ? followup
      : deleteAnyEphemeral(at: frame.persistentAllocations.pop()!) { me in
        me.cleanUpPersistentAllocations(above: n, then: followup)
        }
  }

  /// If `a` was allocated to an ephemeral temporary, deinitializes and destroys
  /// it.
  mutating func deleteAnyEphemeral(
    at a: Address, then followup: @escaping Task.Code) -> Task {
    if let _ = frame.ephemeralAllocations.removeValue(forKey: a) {
      if tracing {
        print("  info: deleting \(a)")
      }
      memory.delete(a)
    }
    return Task(followup)
  }

  /// Deinitializes and destroys any addresses in `locations` that were
  /// allocated to an ephemeral temporary.
  mutating func deleteAnyEphemerals<C: Collection>(
    at locations: C, then followup: @escaping Task.Code
  ) -> Task
    where C.Element == Address
  {
    guard let a0 = locations.first else { return Task(followup) }
    return deleteAnyEphemeral(at: a0) { me in
      me.deleteAnyEphemerals(at: locations.dropFirst(), then: followup)
    }
  }

  /// Copies the value at `source` into the `target` address and continues with
  /// `followup`.
  mutating func copy(
    from source: Address, to target: Address,
    then followup: @escaping FollowupWith<Address>
  ) -> Task {
    if tracing {
      print("  info: copying \(self[source]) into \(target)")
    }
    return initialize(target, to: self[source], then: followup)
  }

  /// Copies the value at `source` into the `target` address and continues with
  /// `followup`.
  mutating func assign(
    _ target: Address, from source: Address,
    then followup: @escaping Task.Code
  ) -> Task {
    memory.assign(from: source, into: target)
    return Task(followup)
  }

  mutating func deinitialize(
    valueAt target: Address, then followup: @escaping Task.Code) -> Task
  {
    if tracing {
      print("  info: deinitializing \(target)")
    }
    memory.deinitialize(target)
    return Task(followup)
  }

  mutating func initialize(
    _ target: Address, to v: Value,
    then followup: @escaping FollowupWith<Address>) -> Task
  {
    if tracing {
      print("  info: initializing \(target) = \(v)")
    }
    memory.initialize(target, to: v)
    return Task { me in followup(target, &me) }
  }
}

typealias FollowupWith<T> = (T, inout Interpreter)->Task

/// Returns a followup that drops its argument and invokes f.
fileprivate func dropResult<T>(_ f: Task) -> FollowupWith<T>
{ { _, me in f(&me) } }

extension Interpreter {
  mutating func evaluateAndConsume<T>(
    _ e: Expression, in followup: @escaping FollowupWith<T>) -> Task {
    evaluate(e) { p, me in
      let v = me[p] as! T
      return me.deleteAnyEphemeral(at: p) { me in followup(v, &me) }
    }
  }

  /// Evaluates `e` (into `destination`, if supplied) and passes
  /// the address of the result on to `followup_`.
  ///
  /// - Parameter asCallee: `true` if `e` is in callee position in a function
  /// call expression.
  mutating func evaluate(
    _ e: Expression,
    asCallee: Bool = false,
    into destination: Address? = nil,
    then followup_: @escaping FollowupWith<Address>
  ) -> Task {
    if tracing {
      print("\(e.site): info: evaluating")
    }
    let followup = !tracing ? followup_
      : { a, me in
        print("\(e.site): info: result = \(me[a])")
        return followup_(a, &me)
      }

    // Handle all possible lvalue expressions
    switch e {
    case let .name(n):
      return evaluate(n, into: destination, then: followup)

    case let .memberAccess(m):
      return evaluate(m, asCallee: asCallee, into: destination, then: followup)

    case let .index(target: t, offset: i, _):
      UNIMPLEMENTED(t, i)

    case .integerLiteral, .booleanLiteral, .tupleLiteral,
         .unaryOperator, .binaryOperator, .functionCall, .intType, .boolType,
         .typeType, .functionType:
      return allocate(e, unlessNonNil: destination) { result, me in
        switch e {
        case let .integerLiteral(r, _):
          return me.initialize(result, to: r, then: followup)
        case let .booleanLiteral(r, _):
          return me.initialize(result, to: r, then: followup)
        case let .tupleLiteral(t):
          return me.evaluateTuple(t.elements[...], into: result, then: followup)
        case let .unaryOperator(x):
          return me.evaluate(x, into: result, then: followup)
        case let .binaryOperator(x):
          return me.evaluate(x, into: result, then: followup)
        case let .functionCall(x):
          return me.evaluate(x, into: result, then: followup)
        case .intType, .boolType, .typeType:
          UNIMPLEMENTED()
        case let .functionType(f):
          UNIMPLEMENTED(f)
        case .name, .memberAccess, .index:
          UNREACHABLE()
        }
      }
    }
  }

  /// Evaluates `name` (into `destination`, if supplied) and passes the address
  /// of the result on to `followup`.
  mutating func evaluate(
    _ name: Identifier, into destination: Address? = nil,
    then followup: @escaping FollowupWith<Address>
  ) -> Task {
    let d = program.definition[name]

    switch d {
    case let t as TypeDeclaration:
      return allocate(.name(name), unlessNonNil: destination) { output, me in
        me.initialize(output, to: t.declaredType, then: followup)
      }

    case let b as SimpleBinding:
      let source = (frame.locals[b] ?? globals[b])!
      return destination != nil
        ? copy(from: source, to: destination!, then: followup)
        : Task { me in followup(source, &me) }

    case let f as FunctionDefinition:
      UNIMPLEMENTED(f)

    case let a as Alternative:
      UNIMPLEMENTED(a)

    case let m as StructMember:
      UNIMPLEMENTED(m)

    default:
      UNIMPLEMENTED(d as Any)
    }
  }

  /// Evaluates `e` into `output` and passes the address of the result on to
  /// `followup`.
  mutating func evaluate(
    _ e: UnaryOperatorExpression, into output: Address,
    then followup: @escaping FollowupWith<Address>
  ) -> Task {
    evaluate(e.operand) { operand, me in
      let result: Value
      switch e.operation.text {
      case "-": result = -(me[operand] as! Int)
      case "not": result = !(me[operand] as! Bool)
      default: UNREACHABLE()
      }
      return me.deleteAnyEphemeral(at: operand) { me in
        me.initialize(output, to: result, then: followup)
      }
    }
  }

  /// Evaluates `e` into `output` and passes the address of the result on to
  /// `followup`.
  mutating func evaluate(
    _ e: BinaryOperatorExpression, into output: Address,
    then followup: @escaping FollowupWith<Address>
  ) -> Task {
    evaluate(e.lhs) { lhs, me in
      if e.operation.text == "and" && (me[lhs] as! Bool == false) {
        return me.copy(from: lhs, to: output, then: followup)
      }
      else if e.operation.text == "or" && (me[lhs] as! Bool == true) {
        return me.copy(from: lhs, to: output, then: followup)
      }

      return me.evaluate(e.rhs) { rhs, me in
        let result: Value
        switch e.operation.text {
        case "==": result = areEqual(me[lhs], me[rhs])
        case "-": result = (me[lhs] as! Int) - (me[rhs] as! Int)
        case "+": result = (me[lhs] as! Int) + (me[rhs] as! Int)
        case "and", "or": result = me[rhs] as! Bool
        default: UNIMPLEMENTED(e)
        }
        return me.deleteAnyEphemerals(at: [lhs, rhs]) { me in
          me.initialize(output, to: result, then: followup)
        }
      }
    }
  }

  /// Evaluates `e` into `output` and passes the address of the result on to
  /// `followup`.
  mutating func evaluate(
    _ e: FunctionCall<Expression>, into output: Address,
    then followup: @escaping FollowupWith<Address>
  ) -> Task {
    evaluate(e.callee, asCallee: true) { callee, me in
      me.evaluate(.tupleLiteral(e.arguments)) { arguments, me in
        let result: Value
        switch me[callee].type {
        case .function:
          UNIMPLEMENTED(e)

        case .type:
          switch Type(me[callee])! {
          case let .alternative(discriminator, parent: resultType):
            // FIXME: there will be an extra copy of the payload; the result
            // should adopt the payload in memory.
            result = ChoiceValue(
              type_: resultType,
              discriminator: discriminator,
              payload: me[arguments] as! Tuple<Value>)

          case .struct:
            UNIMPLEMENTED()
          case .int, .bool, .type, .function, .tuple, .choice, .error:
            UNREACHABLE()
          }

        case .int, .bool, .tuple, .choice, .error, .alternative, .struct:
          UNREACHABLE()
        }
        return me.deleteAnyEphemerals(at: [callee, arguments]) { me in
          me.initialize(output, to: result, then: followup)
        }
      }
    }
  }

  /// Evaluates `e` (into `output`, if supplied) and passes the address of
  /// the result on to `followup`.
  mutating func evaluate(
    _ e: MemberAccessExpression, asCallee: Bool, into output: Address?,
    then followup: @escaping FollowupWith<Address>
  ) -> Task {
    evaluate(e.base) { base, me in
      switch me[base].type {
      case .struct:
        UNIMPLEMENTED()

      case .tuple:
        let source = me.memory.substructure(at: base)[e.member]!
        return output != nil
          ? me.copy(from: source, to: output!, then: followup)
          : Task { me in followup(source, &me) }

      case .type:
        // Handle access to a type member, like a static member in C++.
        switch Type(me[base])! {
        case let .choice(parentID):
          return me.allocate(.memberAccess(e), unlessNonNil: output)
          { output, me in

            let id: ASTIdentity<Alternative>
              = parentID.structure[e.member]!.identity
            let result: Value = asCallee
              ? Type.alternative(id, parent: parentID)
              : ChoiceValue(
                type_: parentID, discriminator: id, payload: .init())

            return me.deleteAnyEphemeral(at: base) { me in
              me.initialize(output, to: result, then: followup)
            }
          }
        default: UNREACHABLE()
        }
        fallthrough
      default:
        UNREACHABLE("\(e)")
      }
    }
  }


  /// Evaluates `e` into `output` and passes the address of the result on to
  /// `followup`.
  mutating func evaluateTuple(
    _ e: ArraySlice<TupleLiteral.Element>,
    into output: Address,
    parts: [FieldID: Address] = [:],
    positionalCount: Int = 0,
    then followup: @escaping FollowupWith<Address>
  ) -> Task {
    // FIXME: too many copies
    if e.isEmpty {
      return initialize(output, to: Tuple(parts).mapFields { self[$0] })
      { output, me in
        me.deleteAnyEphemerals(at: parts.values) { me in
          followup(output, &me)
        }
      }
    }
    else {
      let e0 = e.first!
      return evaluate(e0.payload) { payload, me in
        let key: FieldID = e0.label.map { .label($0) }
          ?? .position(positionalCount)
        var p = parts
        p[key] = payload
        return me.evaluateTuple(
          e.dropFirst(), into: output, parts: p,
          positionalCount: positionalCount + (e0.label == nil ? 1 : 0),
          then: followup)
      }
    }
  }
}

func areEqual(_ l: Value, _ r: Value) -> Bool {
  switch (l as Any, r as Any) {

  case let (lh as AnyHashable, rh as AnyHashable):
    return lh == rh

  case let (lt as TupleValue, rt as TupleValue):
    return lt.count == rt.count && lt.elements.allSatisfy { k, v0 in
      rt.elements[k].map { v1 in areEqual(v0, v1) } ?? false
    }

  case let (lc as ChoiceValue, rc as ChoiceValue):
    return lc.discriminator == rc.discriminator
      && areEqual(lc.payload, rc.payload)

  case let (lt as TupleType, rt as TupleType):
    return lt == rt

  default:
    // All things that aren't equatable are considered equal if their types
    // match and unequal otherwise, to preserve reflexivity.
    return type(of: l) == type(of: r)
  }
}

extension Interpreter {
  /// Matches `p` to the value at `source`, binding variables in `p` to
  /// the corresponding parts of the value, and calling `followup` with an
  /// indication of whether the match was successful.
  mutating func match(
    _ p: Pattern, toValueAt source: Address,
    then followup: @escaping FollowupWith<Bool>
  ) -> Task {
    if tracing {
      print("\(p.site): info: matching against value \(self[source])")
    }
    switch p {
    case let .atom(t):
      return evaluate(t) { target, me in
        let matched = areEqual(me[target], me[source])
        return me.deleteAnyEphemeral(at: target) { me in
          followup(matched, &me)
        }
      }

    case let .variable(b):
      if tracing {
        print("\(b.name.site): info: binding \(self[source])\(source)")
      }
      frame.locals[b] = source
      return Task { me in followup(true, &me) }

    case let .tuple(x):
      return match(x, toValueAt: source, then: followup)

    case let .functionCall(x):
      return match(x, toValueAt: source, then: followup)

    case let .functionType(x): UNIMPLEMENTED(x)
    }
  }

  mutating func match(
    _ p: FunctionCall<Pattern>, toValueAt source: Address,
    then followup: @escaping FollowupWith<Bool>
  ) -> Task {
    return evaluate(p.callee, asCallee: true) { callee, me in
      switch me[source].type {
      case .struct:
        UNIMPLEMENTED()
      case .choice:
        let c = me[source] as! ChoiceValue
        let calleeMatched = Type(me[callee]) == c.alternativeType

        return me.deleteAnyEphemeral(at: callee) { me in
          if !calleeMatched { return followup(false, &me) }

          let payload = me.memory.substructure(at: source)[2]!
          return me.match(p.arguments, toValueAt: payload, then: followup)
        }
      case .int, .bool, .type, .function, .tuple, .error, .alternative:
        UNREACHABLE()
      }
    }
  }

  mutating func match(
    _ p: TuplePattern, toValueAt source: Address,
    then followup: @escaping FollowupWith<Bool>
  ) -> Task {
    if self[source] is TupleValue {
      let sourceStructure = self.memory.substructure(at: source)
      if p.count == sourceStructure.count {
        return matchElements(
          p.fields().elements[...], toValuesAt: sourceStructure, then: followup)
      }
    }
    return Task { me in followup(false, &me) }
  }

  mutating func matchElements(
    _ p: Tuple<Pattern>.Elements.SubSequence, toValuesAt source: Tuple<Address>,
    then followup: @escaping FollowupWith<Bool>
  ) -> Task {
    guard let (k0, p0) = p.first
    else { return Task { me in followup(true, &me) } }
    return match(p0, toValueAt: source.elements[k0]!) { matched, me in
      matched
        ? me.matchElements(p.dropFirst(), toValuesAt: source, then: followup)
        : Task { me in followup(false, &me) }
    }
  }
}

// TODO: move this
/// Just like the built-in assert except that it prints the full path to the
/// file.
///
/// Better for IDEs.
func sanityCheck(
  _ condition: @autoclosure () -> Bool,
  _ message: @autoclosure () -> String = String(),
  filePath: StaticString = #filePath, line: UInt = #line
) {
  Swift.assert(condition(), message(), file: (filePath), line: line)
}

fileprivate extension TupleSyntax {
  func fields() -> Tuple<Payload> {
    var l = ErrorLog()
    let r = fields(reportingDuplicatesIn: &l)
    assert(l.isEmpty)
    return r
  }
}
// TODO: UNREACHABLE variadic signature
// TODO: enums for unary and binary operators.
// TODO: drive matching from source value type.
// TODO: break assign down into subtasks.
