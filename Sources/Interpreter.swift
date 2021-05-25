// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A notional “self-returning function” type.
///
/// Swift doesn't allow that function type to be declared directly, but we can
/// indirect through this struct.
fileprivate struct Onward {
  /// A type representing the underlying implementation.
  typealias Code = (inout Interpreter)->Onward

  /// Creates an instance with the semantics of `implementation`.
  init(_ code: @escaping Code) {
    self.code = code
  }

  /// Executes `self` in `context`.
  func callAsFunction(_ context: inout Interpreter) -> Onward {
    code(&context)
  }

  /// The underlying implementation.
  let code: Code
}

/// A continuation function that takes an input.
fileprivate typealias With<T> = (T, inout Interpreter)->Onward

/// All the data that needs to be saved and restored across function call
/// boundaries.
fileprivate struct CallFrame {
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
  var onReturn: Onward

  /// The code to execute when the current loop exits.
  var onBreak: Onward? = nil

  /// Code that returns to the top of the current loop, if any.
  var onContinue: Onward? = nil
}

/// The engine that executes the program.
struct Interpreter {
  /// The program being executed.
  fileprivate let program: ExecutableProgram

  /// The frame for the current function.
  fileprivate var frame: CallFrame

  /// Mapping from global bindings to address.
  fileprivate var globals: ASTDictionary<SimpleBinding, Address> = .init()

  /// Storage for all addressable values.
  fileprivate var memory = Memory()

  /// The next execution step.
  fileprivate var nextStep: Onward

  /// True iff the program is still running.
  fileprivate var running: Bool = true

  /// A record of any errors encountered.
  fileprivate var errors: ErrorLog = []

  /// True iff we are printing an evaluation trace to stdout
  public var tracing: Bool = false

  /// Creates an instance that runs `p`.
  ///
  /// - Requires: `p.main != nil`
  init(_ p: ExecutableProgram) {
    self.program = p

    frame = CallFrame(
      resultAddress: memory.allocate(),
      onReturn: Onward { me in
        me.cleanUpPersistentAllocations(
          above: 0, then: Onward { me in me.terminate() })
      })

    // First step runs the body of `main`
    nextStep = Onward { [main = program.main!] me in
      me.run(main.body!, then: me.frame.onReturn)
    }
  }

  /// Runs the program to completion and returns the exit code, if any.
  mutating func run() -> Int? {
    while step() {}
    return memory.value(at: frame.resultAddress) as? Int
  }
}

fileprivate extension Interpreter {

  /// Advances execution by one unit of work, returning `true` iff the program
  /// is still running and `false` otherwise.
  mutating func step() -> Bool {
    if running {
      nextStep = nextStep(&self)
    }
    return running
  }

  /// Exits the running program.
  mutating func terminate() -> Onward {
    running = false
    return Onward { _ in fatalError("Terminated program can't continue.") }
  }

  /// Adds an error at the site of `offender` to the error log and marks the
  /// program as terminated.
  ///
  /// Returns a non-executable task for convenience.
  @discardableResult
  mutating func error<Node: AST>(
    _ offender: Node, _ message: String , notes: [CarbonError.Note] = []
  ) -> Onward {
    errors.append(CarbonError(message, at: offender.site, notes: notes))
    return terminate()
  }

  /// Accesses the value in `memory` at `a`, or halts the interpreted program
  /// with an error if `a` is not an initialized address, returning Type.error.
  subscript(a: Address) -> Value {
    memory[a]
  }
}

fileprivate extension Interpreter {
  /// Executes `s`, and then, absent interesting control flow,
  /// `proceed`.
  ///
  /// An example of interesting control flow is a return statement, which
  /// ignores any `proceed` and exits the current function instead.
  ///
  /// In fact this function only executes one “unit of work” and packages the
  /// rest of executing `s` (if any), and whatever follows that, into the
  /// returned `Onward`.
  mutating func run(_ s: Statement, then proceed: Onward) -> Onward {
    if tracing {
      print("\(s.site): info: running statement")
    }
    sanityCheck(
      frame.ephemeralAllocations.isEmpty,
      "leaked \(frame.ephemeralAllocations)")

    switch s {
    case let .expressionStatement(e, _):
      return evaluate(e) { resultAddress, me in
        me.deleteAnyEphemeral(at: resultAddress, then: proceed.code)
      }

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
        me.deleteAnyEphemeral(at: source, then: proceed.code)
        }}}

    case let .initialization(i):
      // Storage must be allocated for the initializer value even if it's an
      // lvalue, so the vars bound to it have distinct values.  Because vars
      // will be bound to parts of the initializer and are mutable, it must
      // persist through the current scope.
      return allocate(i.initializer, mutable: true, persist: true) { rhsArea, me in
        me.evaluate(i.initializer, into: rhsArea) { rhs, me in
          me.match(i.bindings, toValueAt: rhs) { matched, me in
            matched ? proceed : me.error(
              i.bindings, "Initialization pattern not matched by \(me[rhs])")
          }
        }
      }

    case let .if(
           condition: c, thenClause: trueClause, elseClause: falseClause, _):
      return evaluateAndConsume(c) { (condition: Bool, me) in
        return condition
          ? me.run(trueClause, then: proceed)
          : falseClause.map { me.run($0, then: proceed) } ?? proceed
      }

    case let .return(e, _):
      return evaluate(e, into: frame.resultAddress) { _, me in
        me.frame.onReturn(&me)
      }

    case let .block(children, _):
      return inScope(then: proceed) { me, rest in
        me.runBlock(children[...], then: rest)
      }

    case let .while(condition: c, body: body, _):
      let saved = (frame.onBreak, frame.onContinue)
      let mark=frame.persistentAllocations.count

      let onBreak = Onward { me in
        (me.frame.onBreak, me.frame.onContinue) = saved
        return me.cleanUpPersistentAllocations(above: mark, then: proceed)
      }

      let onContinue = Onward { me in
        return me.cleanUpPersistentAllocations(
          above: mark, then: Onward { $0.while(c, run: body, then: onBreak) })
      }

      (frame.onBreak, frame.onContinue) = (onBreak, onContinue)
      return onContinue(&self)

    case let .match(subject: s, clauses: clauses, _):
      return inScope(then: proceed) { me, innerFollowup in
        me.allocate(s, persist: true) { subjectArea, me in
        me.evaluate(s, into: subjectArea) { subject, me in
        me.runMatch(s, at: subject, against: clauses[...], then: innerFollowup)
      }}}

    case .break:
      return frame.onBreak!

    case .continue:
      return frame.onContinue!
    }
  }

  mutating func inScope(
    then proceed: Onward, do body: (inout Self, Onward)->Onward
  ) -> Onward {
    let mark=frame.persistentAllocations.count
    return body(
      &self,
      Onward { me in
        sanityCheck(me.frame.ephemeralAllocations.isEmpty,
               "leaked \(me.frame.ephemeralAllocations)")
        return me.cleanUpPersistentAllocations(above: mark, then: proceed)
      })
  }

  /// Runs `s` and follows up with `proceed`.
  ///
  /// A convenience wrapper for `run(_:then:)` that supports cleaner syntax.
  mutating func run(_ s: Statement, proceed: @escaping Onward.Code) -> Onward {
    run(s, then: Onward(proceed))
  }

  /// Executes the statements of `content` in order, then `proceed`.
  mutating func runBlock(_ content: ArraySlice<Statement>, then proceed: Onward)
    -> Onward
  {
    return content.isEmpty ? proceed
      : run(content.first!) { me in
          me.runBlock(content.dropFirst(), then: proceed)
        }
  }

  mutating func runMatch(
    _ subject: Expression, at subjectLocation: Address,
    against clauses: ArraySlice<MatchClause>,
    then proceed: Onward) -> Onward
  {
    guard let clause = clauses.first else {
      return error(subject, "no pattern matches \(self[subjectLocation])")
    }

    let onMatch = Onward { me in
      me.inScope(then: proceed) { me, innerFollowup in
        me.run(clause.action, then: innerFollowup)
      }
    }
    guard let p = clause.pattern else { return onMatch }

    return match(p, toValueAt: subjectLocation) { matched, me in
      matched ? onMatch : me.runMatch(
        subject, at: subjectLocation, against: clauses.dropFirst(),
        then: proceed)
    }
  }

  mutating func `while`(
    _ c: Expression, run body: Statement, then proceed: Onward) -> Onward
  {
    return evaluateAndConsume(c) { (runBody: Bool, me) in
      return runBody
        ? me.run(
          body, then: Onward { me in me.while(c, run: body, then: proceed)})
        : proceed
    }
  }
}

/// Values and memory.
fileprivate extension Interpreter {
  /// Allocates an address for the result of evaluating `e`, passing it on to
  /// `proceed` along with `self`.
  mutating func allocate(
    _ e: Expression, mutable: Bool = false, persist: Bool = false,
    then proceed: @escaping With<Address>
  ) -> Onward {
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
    return Onward { me in proceed(a, &me) }
  }

  /// Allocates an address for the result of evaluating `e`, passing it on to
  /// `proceed` along with `self`.
  mutating func allocate(
    _ e: Expression, unlessNonNil destination: Address?,
    then proceed: @escaping With<Address>
  ) -> Onward {
    if let a = destination { return Onward { me in proceed(a, &me) } }
    return allocate(e, then: proceed)
  }

  /// Destroys and reclaims memory of the `n` locally-allocated values at the
  /// top of the allocation stack.
  mutating func cleanUpPersistentAllocations(
    above n: Int, then proceed: Onward
  ) -> Onward {
    frame.persistentAllocations.count == n ? proceed
      : deleteLocalValue_doNotCallDirectly(
        at: frame.persistentAllocations.pop()!
      ) { me in me.cleanUpPersistentAllocations(above: n, then: proceed) }
  }

  mutating func deleteLocalValue_doNotCallDirectly(
    at a: Address, then proceed: @escaping Onward.Code
  ) -> Onward {
    if tracing { print("  info: deleting \(a)") }
    memory.delete(a)
    return Onward(proceed)
  }

  /// If `a` was allocated to an ephemeral temporary, deinitializes and destroys
  /// it.
  mutating func deleteAnyEphemeral(
    at a: Address, then proceed: @escaping Onward.Code) -> Onward {
    if let _ = frame.ephemeralAllocations.removeValue(forKey: a) {
      return deleteLocalValue_doNotCallDirectly(at: a, then: proceed)
    }
    return Onward(proceed)
  }

  /// Deinitializes and destroys any addresses in `locations` that were
  /// allocated to an ephemeral temporary.
  mutating func deleteAnyEphemerals<C: Collection>(
    at locations: C, then proceed: @escaping Onward.Code
  ) -> Onward
    where C.Element == Address
  {
    guard let a0 = locations.first else { return Onward(proceed) }
    return deleteAnyEphemeral(at: a0) { me in
      me.deleteAnyEphemerals(at: locations.dropFirst(), then: proceed)
    }
  }

  /// Copies the value at `source` into the `target` address and continues with
  /// `proceed`.
  mutating func copy(
    from source: Address, to target: Address,
    then proceed: @escaping With<Address>
  ) -> Onward {
    if tracing {
      print("  info: copying \(self[source]) into \(target)")
    }
    return initialize(target, to: self[source], then: proceed)
  }

  /// Copies the value at `source` into the `target` address and continues with
  /// `proceed`.
  mutating func assign(
    _ target: Address, from source: Address,
    then proceed: @escaping Onward.Code
  ) -> Onward {
    if tracing {
      print("  info: assigning \(self[source])\(source) into \(target)")
    }
    memory.assign(from: source, into: target)
    return Onward(proceed)
  }

  mutating func deinitialize(
    valueAt target: Address, then proceed: @escaping Onward.Code) -> Onward
  {
    if tracing {
      print("  info: deinitializing \(target)")
    }
    memory.deinitialize(target)
    return Onward(proceed)
  }

  mutating func initialize(
    _ target: Address, to v: Value,
    then proceed: @escaping With<Address>) -> Onward
  {
    if tracing {
      print("  info: initializing \(target) = \(v)")
    }
    memory.initialize(target, to: v)
    return Onward { me in proceed(target, &me) }
  }
}

fileprivate extension Interpreter {
  mutating func evaluateAndConsume<T>(
    _ e: Expression, in proceed: @escaping With<T>) -> Onward {
    evaluate(e) { p, me in
      let v = me[p] as! T
      return me.deleteAnyEphemeral(at: p) { me in proceed(v, &me) }
    }
  }

  /// Evaluates `e` (into `destination`, if supplied) and passes
  /// the address of the result on to `proceed_`.
  ///
  /// - Parameter asCallee: `true` if `e` is in callee position in a function
  /// call expression.
  mutating func evaluate(
    _ e: Expression,
    asCallee: Bool = false,
    into destination: Address? = nil,
    then proceed_: @escaping With<Address>
  ) -> Onward {
    if tracing {
      print(
        "\(e.site): info: evaluating "
          + (asCallee ? "as callee " : "")
          + (destination != nil ? "into \(destination!)" : ""))
    }
    let proceed = !tracing ? proceed_
      : { a, me in
        print("\(e.site): info: result = \(me[a])")
        return proceed_(a, &me)
      }

    // Handle all possible lvalue expressions
    switch e {
    case let .name(n):
      return evaluate(n, into: destination, then: proceed)

    case let .memberAccess(m):
      return evaluate(m, asCallee: asCallee, into: destination, then: proceed)

    case let .index(target: t, offset: i, _):
      return evaluateIndex(target: t, offset: i, into: destination, then: proceed)

    case .integerLiteral, .booleanLiteral, .tupleLiteral,
         .unaryOperator, .binaryOperator, .functionCall, .intType, .boolType,
         .typeType, .functionType:
      return allocate(e, unlessNonNil: destination) { result, me in
        switch e {
        case let .integerLiteral(r, _):
          return me.initialize(result, to: r, then: proceed)
        case let .booleanLiteral(r, _):
          return me.initialize(result, to: r, then: proceed)
        case let .tupleLiteral(t):
          return me.evaluateTuple(t.elements[...], into: result, then: proceed)
        case let .unaryOperator(x):
          return me.evaluate(x, into: result, then: proceed)
        case let .binaryOperator(x):
          return me.evaluate(x, into: result, then: proceed)
        case let .functionCall(x):
          return me.evaluate(x, into: result, then: proceed)
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
  /// of the result on to `proceed`.
  mutating func evaluate(
    _ name: Identifier, into destination: Address? = nil,
    then proceed: @escaping With<Address>
  ) -> Onward {
    let d = program.definition[name]

    switch d {
    case let t as TypeDeclaration:
      return allocate(.name(name), unlessNonNil: destination) { output, me in
        me.initialize(output, to: t.declaredType, then: proceed)
      }

    case let b as SimpleBinding:
      let source = (frame.locals[b] ?? globals[b])!
      return destination != nil
        ? copy(from: source, to: destination!, then: proceed)
        : Onward { me in proceed(source, &me) }

    case let f as FunctionDefinition:
      // Bogus parameterTypes and returnType until I figure out how to get those.  -Jeremy
      let function = FunctionValue(type: .function(parameterTypes: Tuple(), returnType: .int), code: f)
      /* This commented version causes a leak. -Jeremy
      return allocate(.name(name), unlessNonNil: destination) { output, me in
        me.initialize(output, to: function, then: proceed) }
       */
      if destination == nil {
        let address = memory.allocate(mutable: false)
        return initialize(address, to: function, then: proceed)
      } else {
        return initialize(destination!, to: function, then: proceed)
      }

    case let a as Alternative:
      UNIMPLEMENTED(a)

    case let m as StructMember:
      UNIMPLEMENTED(m)

    default:
      UNIMPLEMENTED(d as Any)
    }
  }

  /// Evaluates `e` into `output` and passes the address of the result on to
  /// `proceed`.
  mutating func evaluate(
    _ e: UnaryOperatorExpression, into output: Address,
    then proceed: @escaping With<Address>
  ) -> Onward {
    evaluate(e.operand) { operand, me in
      let result: Value
      switch e.operation.text {
      case "-": result = -(me[operand] as! Int)
      case "not": result = !(me[operand] as! Bool)
      default: UNREACHABLE()
      }
      return me.deleteAnyEphemeral(at: operand) { me in
        me.initialize(output, to: result, then: proceed)
      }
    }
  }

  /// Evaluates `e` into `output` and passes the address of the result on to
  /// `proceed`.
  mutating func evaluate(
    _ e: BinaryOperatorExpression, into output: Address,
    then proceed: @escaping With<Address>
  ) -> Onward {
    evaluate(e.lhs) { lhs, me in
      if e.operation.text == "and" && (me[lhs] as! Bool == false) {
        return me.copy(from: lhs, to: output, then: proceed)
      }
      else if e.operation.text == "or" && (me[lhs] as! Bool == true) {
        return me.copy(from: lhs, to: output, then: proceed)
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
          me.initialize(output, to: result, then: proceed)
        }
      }
    }
  }

  /// Evaluates `e` into `output` and passes the address of the result on to
  /// `proceed`.
  mutating func evaluate(
    _ e: FunctionCall<Expression>, into output: Address,
    then proceed: @escaping With<Address>
  ) -> Onward {
    evaluate(e.callee, asCallee: true) { callee, me in
      me.evaluate(.tupleLiteral(e.arguments)) { arguments, me in
        // TODO: instead of using the callee value's type to dispatch, use the
        // static type of the e.callee expression.
        switch me[callee].type {
        case .function:
          let function = me[callee] as! FunctionValue
          let old_frame = me.frame
          let new_frame =
            CallFrame(locals: ASTDictionary<SimpleBinding, Address>(),
                      resultAddress: output,
                      onReturn: Onward { me in
                        me.cleanUpPersistentAllocations(
                          above: 0,
                          then: Onward{ me in
                            me.frame = old_frame
                            return me.deleteAnyEphemeral(at: arguments) { me in
                              proceed(output, &me) } }) })
          me.frame = new_frame
          return me.match(function.code.parameters, toValueAt: arguments) { matched, me in
            if matched {
              return me.run(function.code.body!,
                            then: Onward { me in
                              // Return an empty tuple when the function falls off the end.
                              me.initialize(me.frame.resultAddress, to: Tuple() ,
                                            then: { _, me in me.frame.onReturn })
                            })
            } else {
              return me.error(e, "failed to match parameters and arguments in function call")
            }
          }

        case .type:
          switch Type(me[callee])! {
          case let .alternative(discriminator, parent: resultType):
            // FIXME: there will be an extra copy of the payload; the result
            // should adopt the payload in memory.
            let result = ChoiceValue(
              type_: resultType,
              discriminator: discriminator,
              payload: me[arguments] as! Tuple<Value>)
            return me.deleteAnyEphemerals(at: [callee, arguments]) { me in
              me.initialize(output, to: result, then: proceed)
            }

          case .struct:
            UNIMPLEMENTED()
          case .int, .bool, .type, .function, .tuple, .choice, .error:
            UNREACHABLE()
          }

        case .int, .bool, .tuple, .choice, .error, .alternative, .struct:
          UNREACHABLE()
        }
      }
    }
  }

  /// Evaluates `e` (into `output`, if supplied) and passes the address of
  /// the result on to `proceed`.
  mutating func evaluate(
    _ e: MemberAccessExpression, asCallee: Bool, into output: Address?,
    then proceed: @escaping With<Address>
  ) -> Onward {
    evaluate(e.base) { base, me in
      switch me[base].type {
      case .struct:
        UNIMPLEMENTED()

      case .tuple:
        let source = me.memory.substructure(at: base)[e.member]!
        return output != nil
          ? me.copy(from: source, to: output!, then: proceed)
          : Onward { me in proceed(source, &me) }

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
              me.initialize(output, to: result, then: proceed)
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

  mutating func evaluateIndex(
    target t: Expression, offset i: Expression, into output: Address?,
    then proceed: @escaping With<Address>
  ) -> Onward {
    evaluate(t, into: output) { targetAddress, me in
      me.evaluate(i) { indexAddress, me in
        let index = me[indexAddress] as! Int
        guard let resultAddress
                = me.memory.substructure(at: targetAddress)[index]
        else {
          return me.error(
            i, "crazy bad index \(index) for tuple \(me[targetAddress])")
        }

        return me.deleteAnyEphemeral(at: indexAddress) { me in
          output == nil ? proceed(resultAddress, &me)
            : me.copy(from: resultAddress, to: output!) { _, me in
              me.deleteAnyEphemeral(at: targetAddress) { me in
                proceed(output!, &me)
              }
            }
        }
      }
    }
  }

  /// Evaluates `e` into `output` and passes the address of the result on to
  /// `proceed`.
  mutating func evaluateTuple(
    _ e: ArraySlice<TupleLiteral.Element>,
    into output: Address,
    parts: [FieldID: Address] = [:],
    positionalCount: Int = 0,
    then proceed: @escaping With<Address>
  ) -> Onward {
    // FIXME: too many copies
    if e.isEmpty {
      return initialize(output, to: Tuple(parts).mapFields { self[$0] })
      { output, me in
        me.deleteAnyEphemerals(at: parts.values) { me in
          proceed(output, &me)
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
          then: proceed)
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

fileprivate extension Interpreter {
  /// Matches `p` to the value at `source`, binding variables in `p` to
  /// the corresponding parts of the value, and calling `proceed` with an
  /// indication of whether the match was successful.
  mutating func match(
    _ p: Pattern, toValueAt source: Address,
    then proceed: @escaping With<Bool>
  ) -> Onward {
    if tracing {
      print("\(p.site): info: matching against value \(self[source])")
    }
    switch p {
    case let .atom(t):
      return evaluate(t) { target, me in
        let matched = areEqual(me[target], me[source])
        return me.deleteAnyEphemeral(at: target) { me in
          proceed(matched, &me)
        }
      }

    case let .variable(b):
      if tracing {
        print("\(b.name.site): info: binding \(self[source])\(source)")
      }
      frame.locals[b] = source
      return Onward { me in proceed(true, &me) }

    case let .tuple(x):
      return match(x, toValueAt: source, then: proceed)

    case let .functionCall(x):
      return match(x, toValueAt: source, then: proceed)

    case let .functionType(x): UNIMPLEMENTED(x)
    }
  }

  mutating func match(
    _ p: FunctionCall<Pattern>, toValueAt source: Address,
    then proceed: @escaping With<Bool>
  ) -> Onward {
    return evaluate(p.callee, asCallee: true) { callee, me in
      switch me[source].type {
      case .struct:
        UNIMPLEMENTED()
      case .choice:
        let c = me[source] as! ChoiceValue
        let calleeMatched = Type(me[callee]) == c.alternativeType

        return me.deleteAnyEphemeral(at: callee) { me in
          if !calleeMatched { return proceed(false, &me) }

          let payload = me.memory.substructure(at: source)[2]!
          return me.match(p.arguments, toValueAt: payload, then: proceed)
        }
      case .int, .bool, .type, .function, .tuple, .error, .alternative:
        UNREACHABLE()
      }
    }
  }

  mutating func match(
    _ p: TuplePattern, toValueAt source: Address,
    then proceed: @escaping With<Bool>
  ) -> Onward {
    if self[source] is TupleValue {
      let sourceStructure = self.memory.substructure(at: source)
      if p.count == sourceStructure.count {
        return matchElements(
          p.fields().elements[...], toValuesAt: sourceStructure, then: proceed)
      }
    }
    return Onward { me in proceed(false, &me) }
  }

  mutating func matchElements(
    _ p: Tuple<Pattern>.Elements.SubSequence, toValuesAt source: Tuple<Address>,
    then proceed: @escaping With<Bool>
  ) -> Onward {
    guard let (k0, p0) = p.first
    else { return Onward { me in proceed(true, &me) } }
    return match(p0, toValueAt: source.elements[k0]!) { matched, me in
      matched
        ? me.matchElements(p.dropFirst(), toValuesAt: source, then: proceed)
        : Onward { me in proceed(false, &me) }
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
// TODO: output vs. destination?
