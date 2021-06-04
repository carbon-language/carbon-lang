// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A notional “self-returning function” type, to be used as a continuation.
///
/// Swift doesn't allow a recursive function type to be declared directly, but
/// we can indirect through `Onward`.
fileprivate typealias Next = (inout Interpreter)->Onward

/// A notional “self-returning function” type, to be returned from continuations.
///
/// Swift doesn't allow a recursive function type to be declared directly, but
/// we can indirect through this struct.
fileprivate struct Onward {

  /// Creates an instance with the semantics of `implementation`.
  init(_ code: @escaping Next) {
    self.code = code
  }

  /// Executes `self` in `context`.
  func callAsFunction(_ context: inout Interpreter) -> Onward {
    code(&context)
  }

  /// The underlying implementation.
  let code: Next
}

/// A continuation function that takes an input.
fileprivate typealias Consumer<T> = (T, inout Interpreter)->Onward

/// An operator for constructing a continuation result from a `Consumer<T>`
/// function and an input value.
infix operator => : DefaultPrecedence

/// Creates a continuation result notionally corresponding to `followup(x)`.
///
/// You can think of `a => f` or as a way of binding `a` to the argument of `f`
/// and coercing the result to `Onward`.
fileprivate func => <T>(x: T, followup: @escaping Consumer<T>) -> Onward {
  Onward { me in followup(x, &me) }
}

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

  /// Mapping from expression to its static type.
  fileprivate var staticType: ASTDictionary<Expression, Type> {
    program.staticType
  }

  /// Creates an instance that runs `p`.
  ///
  /// - Requires: `p.main != nil`
  init(_ p: ExecutableProgram) {
    self.program = p

    frame = CallFrame(
      resultAddress: memory.allocate(boundTo: .int),
      onReturn: Onward { me in
        me.cleanUpPersistentAllocations(above: 0) { me in me.terminate() }
      })

    // First step runs the body of `main`
    nextStep = Onward { [main = program.main!] me in
      me.run(main.body!, then: me.frame.onReturn.code)
    }
  }

  /// Runs the program to completion and returns the exit code, if any.
  mutating func run() -> Int? {
    while step() {}
    return memory[frame.resultAddress] as? Int
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
  mutating func run(_ s: Statement, then proceed: @escaping Next) -> Onward {
    if tracing {
      print("\(s.site): info: running statement")
    }
    sanityCheck(
      frame.ephemeralAllocations.isEmpty,
      "leaked \(frame.ephemeralAllocations)")

    switch s {
    case let .expressionStatement(e, _):
      return evaluate(e) { resultAddress, me in
        me.deleteAnyEphemeral(at: resultAddress, then: proceed)
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
              "\(t.site): info: assigning"
                + " \(me[source])\(source) into \(target)")
          }
          return me.assign(target, from: source) { me in
            me.deleteAnyEphemeral(at: source, then: proceed)
          }
        }
      }

    case let .initialization(i):
      // Storage must be allocated for the initializer value even if it's an
      // lvalue, so the vars bound to it have distinct values.  Because vars
      // will be bound to parts of the initializer and are mutable, it must
      // persist through the current scope.
      return allocate(i.initializer, mutable: true, persist: true) { rhsArea, me in
        me.evaluate(i.initializer, into: rhsArea) { rhs, me in
          me.match(
            i.bindings, toValueOfType: me.staticType[i.initializer]!, at: rhs)
          { matched, me in
            matched ? Onward(proceed) : me.error(
              i.bindings, "Initialization pattern not matched by \(me[rhs])")
          }
        }
      }

    case let .if(c, s0, else: s1, _):
      return evaluateAndConsume(c) { (condition: Bool, me) in
        if condition {
          return me.run(s0, then: proceed)
        }
        else {
          if let s1 = s1 { return me.run(s1, then: proceed) }
          else { return Onward(proceed) }
        }
      }

    case let .return(e, _):
      return evaluate(e, into: frame.resultAddress) { _, me in
        me.frame.onReturn(&me)
      }

    case let .block(children, _):
      return inScope(
        do: { me, proceed1 in me.runBlock(children[...], then: proceed1) },
        then: proceed)

    case let .while(condition, body, _):
      let savedLoopContext = (frame.onBreak, frame.onContinue)
      let mark=frame.persistentAllocations.count

      let onBreak = Onward { me in
        (me.frame.onBreak, me.frame.onContinue) = savedLoopContext
        return me.cleanUpPersistentAllocations(above: mark, then: proceed)
      }

      let onContinue = Onward { me in
        return me.cleanUpPersistentAllocations(above: mark) {
          $0.runWhile(condition, body, then: onBreak.code)
        }
      }

      (frame.onBreak, frame.onContinue) = (onBreak, onContinue)
      return onContinue(&self)

    case let .match(subject: e, clauses: clauses, _):
      return inScope(do: { me, proceed1 in
        me.allocate(e, persist: true) { subjectArea, me in
          me.evaluate(e, into: subjectArea) { subject, me in
            me.runMatch(e, at: subject, against: clauses[...], then: proceed1)
          }}}, then: proceed)

    case .break:
      return frame.onBreak!

    case .continue:
      return frame.onContinue!
    }
  }

  mutating func inScope(
    do body: (inout Self, @escaping Next)->Onward,
    then proceed: @escaping Next) -> Onward
  {
    let mark=frame.persistentAllocations.count
    return body(&self) { me in
      sanityCheck(
        me.frame.ephemeralAllocations.isEmpty,
        "leaked \(me.frame.ephemeralAllocations)")

      return me.cleanUpPersistentAllocations(above: mark, then: proceed)
    }
  }

  /// Executes the statements of `content` in order, then `proceed`.
  mutating func runBlock(
    _ content: ArraySlice<Statement>, then proceed: @escaping Next) -> Onward
  {
    content.isEmpty ? Onward(proceed) : run(content.first!) { me in
      me.runBlock(content.dropFirst(), then: proceed)
    }
  }

  mutating func runMatch(
    _ e: Expression, at subject: Address,
    against clauses: ArraySlice<MatchClause>,
    then proceed: @escaping Next) -> Onward
  {
    guard let clause = clauses.first else {
      return error(e, "no pattern matches \(self[subject])")
    }

    let onMatch = Onward { me in
      me.inScope(
        do: { me, proceed in me.run(clause.action, then: proceed) },
        then: proceed)
    }
    guard let p = clause.pattern else { return onMatch }

    return match(p, toValueOfType: staticType[e]!, at: subject) { matched, me in
      if matched { return onMatch }
      return me.runMatch(
        e, at: subject, against: clauses.dropFirst(), then: proceed)
    }
  }

  mutating func runWhile(
    _ c: Expression, _ body: Statement, then proceed: @escaping Next
  ) -> Onward {
    return evaluateAndConsume(c) { (runBody: Bool, me) in
      return runBody
        ? me.run(body) { me in me.runWhile(c, body, then: proceed)}
        : Onward(proceed)
    }
  }
}

/// Values and memory.
fileprivate extension Interpreter {
  /// Allocates an address earmarked for the eventual result of evaluating `e`,
  /// passing it on to `proceed` along with `self`.
  mutating func allocate(
    _ e: Expression, mutable: Bool = false, persist: Bool = false,
    then proceed: @escaping Consumer<Address>) -> Onward
  {
    let t = staticType[e]!
    let a = memory.allocate(boundTo: t, mutable: mutable)
    if tracing {
      print(
        "\(e.site): info: allocated \(a) bound to \(t)"
          + " (\(persist ? "persistent" : "ephemeral"))")
    }
    if persist {
      frame.persistentAllocations.push(a)
    }
    else {
      frame.ephemeralAllocations[a] = e
    }
    return a => proceed
  }

  /// Allocates an address for the result of evaluating `e`, passing it on to
  /// `proceed` along with `self`.
  mutating func allocate(
    _ e: Expression, unlessNonNil destination: Address?,
    then proceed: @escaping Consumer<Address>) -> Onward
  {
    destination.map { $0 => proceed } ?? allocate(e, then: proceed)
  }

  /// Destroys and reclaims memory of locally-allocated values at the top of the
  /// allocation stack until the stack's count is `n`.
  mutating func cleanUpPersistentAllocations(
    above n: Int, then proceed: @escaping Next) -> Onward
  {
    frame.persistentAllocations.count == n ? Onward(proceed)
      : deleteLocalValue_doNotCallDirectly(
        at: frame.persistentAllocations.pop()!
      ) { me in me.cleanUpPersistentAllocations(above: n, then: proceed) }
  }

  mutating func deleteLocalValue_doNotCallDirectly(
    at a: Address, then proceed: @escaping Next) -> Onward
  {
    if tracing { print("  info: deleting \(a)") }
    memory.deinitialize(a)
    memory.deallocate(a)
    return Onward(proceed)
  }

  /// If `a` was allocated to an ephemeral temporary, deinitializes and destroys
  /// it.
  mutating func deleteAnyEphemeral(
    at a: Address, then proceed: @escaping Next) -> Onward
  {
    if let _ = frame.ephemeralAllocations.removeValue(forKey: a) {
      return deleteLocalValue_doNotCallDirectly(at: a, then: proceed)
    }
    return Onward(proceed)
  }

  /// Deinitializes and destroys any addresses in `locations` that were
  /// allocated to an ephemeral temporary.
  mutating func deleteAnyEphemerals<C: Collection>(
    at locations: C, then proceed: @escaping Next) -> Onward
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
    then proceed: @escaping Consumer<Address>) -> Onward
  {
    if tracing {
      print("  info: copying \(self[source]) into \(target)")
    }
    return initialize(target, to: self[source], then: proceed)
  }

  /// Copies the value at `source` into the `target` address and continues with
  /// `proceed`.
  mutating func assign(
    _ target: Address, from source: Address,
    then proceed: @escaping Next) -> Onward
  {
    if tracing {
      print("  info: assigning \(self[source])\(source) into \(target)")
    }
    memory.assign(from: source, into: target)
    return Onward(proceed)
  }

  mutating func deinitialize(
    valueAt target: Address, then proceed: @escaping Next) -> Onward
  {
    if tracing {
      print("  info: deinitializing \(target)")
    }
    memory.deinitialize(target)
    return Onward(proceed)
  }

  mutating func initialize(
    _ target: Address, to v: Value,
    then proceed: @escaping Consumer<Address>) -> Onward
  {
    if tracing {
      print("  info: initializing \(target) = \(v)")
    }
    memory.initialize(target, to: v)
    return target => proceed
  }
}

fileprivate extension Interpreter {
  mutating func evaluateAndConsume<T>(
    _ e: Expression, in proceed: @escaping Consumer<T>) -> Onward {
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
    _ e: Expression, asCallee: Bool = false, into destination: Address? = nil,
    then proceed_: @escaping Consumer<Address>) -> Onward
  {
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
    then proceed: @escaping Consumer<Address>) -> Onward
  {
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
        : source => proceed

    case let f as FunctionDefinition:
      let result = FunctionValue(
        dynamic_type: program.typeOfNameDeclaredBy[f.identity]!.final!, code: f)

      return allocate(.name(name), unlessNonNil: destination) { output, me in
        me.initialize(output, to: result, then: proceed)
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
    then proceed: @escaping Consumer<Address>) -> Onward
  {
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
    then proceed: @escaping Consumer<Address>) -> Onward
  {
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
    then proceed: @escaping Consumer<Address>) -> Onward
  {
    evaluate(e.callee, asCallee: true) { calleeAddress, me in
      me.evaluate(.tupleLiteral(e.arguments)) { arguments, me in
        switch me.staticType[e.callee]! {

        case .function:
          let callee = me[calleeAddress] as! FunctionValue

          return me.deleteAnyEphemeral(at: calleeAddress) { me in
            let old_frame = me.frame

            me.frame = CallFrame(
              resultAddress: output,
              onReturn: Onward { me in
                me.cleanUpPersistentAllocations(above: 0) { me in
                  me.frame = old_frame
                  return me.deleteAnyEphemeral(at: arguments) { me in
                    proceed(output, &me)
                  }
                }
              })

            let argumentsType = me.staticType[.tupleLiteral(e.arguments)]!
            return me.match(
              callee.code.parameters,
              toValueOfType: argumentsType.tuple!, at: arguments
            ) { matched, me in
              if matched {
                return me.run(callee.code.body!) { me in
                  // Return an empty tuple when the function falls off the end.
                  me.initialize(me.frame.resultAddress, to: Tuple()) {
                    _, me in me.frame.onReturn
                  }
                }
              }
              return me.error(
                e.arguments,
                "arguments don't match literal values in parameter list",
                notes: [("parameter list", callee.code.parameters.site)])
            }
          }

        case let .alternative(discriminator):
          // FIXME: there will be an extra copy of the payload; the result
          // should adopt the payload in memory.
          let result = ChoiceValue(
            type: me.program.enclosingChoice[discriminator.structure]!.identity,
            discriminator: discriminator,
            payload: me[arguments] as! Tuple<Value>)
          return
            me.deleteAnyEphemerals(at: [calleeAddress, arguments]) { me in
              me.initialize(output, to: result, then: proceed)
            }

        case .type:
          guard case .struct(let id) = me[calleeAddress] as! Type else {
            UNREACHABLE()
          }
          let result = StructValue(
            type: id,
            payload: me[arguments] as! Tuple<Value>)
          return 
            me.deleteAnyEphemerals(at: [calleeAddress, arguments]) { me in
              me.initialize(output, to: result, then: proceed)
            }
          case .int, .bool, .choice, .struct, .tuple, .error:
            UNREACHABLE()
        }
      }
    }
  }

  /// Evaluates `e` (into `output`, if supplied) and passes the address of
  /// the result on to `proceed`.
  mutating func evaluate(
    _ e: MemberAccessExpression, asCallee: Bool, into output: Address?,
    then proceed: @escaping Consumer<Address>) -> Onward
  {
    evaluate(e.base) { base, me in
      switch me.staticType[e.base] {
      case .struct:
        let source = base.^e.member

        return output != nil
          ? me.copy(from: source, to: output!, then: proceed)
          : source => proceed
        
      case .tuple:
        let source = base.^e.member

        return output != nil
          ? me.copy(from: source, to: output!, then: proceed)
          : source => proceed

      case .type:
        // Handle access to a type member, like a static member in C++.
        switch Type(me[base])! {
        case let .choice(parentID):
          return me.allocate(.memberAccess(e), unlessNonNil: output)
          { output, me in

            let id: ASTIdentity<Alternative>
              = parentID.structure[e.member]!.identity
            let result: Value = asCallee
              ? AlternativeValue(id)
              : ChoiceValue(type: parentID, discriminator: id, payload: .init())

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
    then proceed: @escaping Consumer<Address>) -> Onward
  {
    evaluate(t, into: output) { targetAddress, me in
      me.evaluate(i) { indexAddress, me in
        let index = me[indexAddress] as! Int
        let resultAddress = targetAddress.^index

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
    then proceed: @escaping Consumer<Address>) -> Onward
  {
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
    _ p: Pattern,
    toValueOfType sourceType: Type, at source: Address,
    then proceed: @escaping Consumer<Bool>) -> Onward
  {
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
      return true => proceed

    case let .tuple(x):
      return match(
        x, toValueOfType: sourceType.tuple!, at: source, then: proceed)

    case let .functionCall(x):
      return match(x, toValueOfType: sourceType, at: source, then: proceed)

    case let .functionType(x): UNIMPLEMENTED(x)
    }
  }

  mutating func match(
    _ p: FunctionCall<Pattern>,
    toValueOfType subjectType: Type, at subject: Address,
    then proceed: @escaping Consumer<Bool>) -> Onward
  {
    switch subjectType {
    case .struct:
      UNIMPLEMENTED()

    case .choice:
      let subjectAlternative = (self[subject] as! ChoiceValue).discriminator

      if staticType[p.callee] != .alternative(subjectAlternative) {
        return false => proceed
      }

      return match(
        p.arguments, toValueOfType: program.payloadType[subjectAlternative]!,
        at: subject, then: proceed)

    case .int, .bool, .type, .function, .tuple, .error, .alternative:
      UNREACHABLE()
    }
  }

  mutating func match(
    _ p: TuplePattern,
    toValueOfType subjectTypes: Tuple<Type>, at subject: Address,
    then proceed: @escaping Consumer<Bool>) -> Onward
  {
    let p1 = p.fields()
    if !subjectTypes.isCongruent(to: p1) { return false => proceed }

    return matchElements(
      p1.elements[...],
      toValuesOfType: subjectTypes, at: subject, then: proceed)
  }

  mutating func matchElements(
    _ p: Tuple<Pattern>.Elements.SubSequence,
    toValuesOfType subjectTypes: Tuple<Type>, at subject: Address,
    then proceed: @escaping Consumer<Bool>) -> Onward
  {
    guard let (k0, p0) = p.first else { return true => proceed }
    return match(
      p0, toValueOfType: subjectTypes[k0]!, at: subject.^k0
    ) { matched, me in
      if !matched { return false => proceed }
      return me.matchElements(
        p.dropFirst(),
        toValuesOfType: subjectTypes, at: subject, then: proceed)
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
    sanityCheck(l.isEmpty)
    return r
  }
}
// TODO: UNREACHABLE variadic signature
// TODO: enums for unary and binary operators.
// TODO: drive matching from source value type.
// TODO: break assign down into subtasks.
// TODO: output vs. destination?
