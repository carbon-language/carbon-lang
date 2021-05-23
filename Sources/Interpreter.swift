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
  init(_ implementation: @escaping Code) {
    self.implementation = implementation
  }

  /// Executes `self` in `context`.
  func callAsFunction(_ context: inout Interpreter) -> Task {
    implementation(&context)
  }

  /// The underlying implementation.
  private let implementation: Code
}

/// All the data that needs to be saved and restored across function call
/// boundaries.
struct CallFrame {
  /// A record that a particular value is alive
  enum Allocation {
    /// A bound name; find its address in `locals`.
    case variable(SimpleBinding)

    /// A temporary value; the `Expression` field is just for diagnostic
    /// purposes.
    case temporary(Expression, Address)
  }

  /// The locations that have been allocated and need to be cleaned up before
  /// the call exits.
  ///
  /// When scopes are exited, the allocations for that scope are popped and
  /// cleaned up.
  var allocations: Stack<Allocation> = .init()

  /// A mapping from local bindings to addresses.
  ///
  /// Entries are cleared when the allocations are cleaned up.
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
    return memory[frame.resultAddress] as! (Int?)
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
    if tracing {
      print("\(s.site): info: running statement")
    }
    switch s {
    case let .expressionStatement(e, _):
      UNIMPLEMENTED(e)

    case let .assignment(target: t, source: s, _):
      return evaluate(t) { target, me in
        // Can't evaluate source into target because target may be referenced in
        // source (e.g. x = x - 1)
        me.evaluate(s) { source, me in
          me.deinitialize(valueAt: target) { me in
            me.copy(from: source, to: target, then: dropResult(followup))
          }
        }
      }

    case let .initialization(i):
      // This can only be cleaned up completely on scope exit because variables
      // are being bound to (parts of) the value.  Whether it's possible to tear
      // down unused *parts* of the value earlier is unknown.
      let rhs = memory.allocate(mutable: true)
      return evaluate(i.initializer, into: rhs) { rhs, me in
        return me.match(i.bindings, toValueAt: rhs) { matched, me in
          matched ? followup : me.error(
            i.bindings, "Initialization pattern not matched by \(me[rhs])")
        }
      }

    case let .if(condition: c, thenClause: t, elseClause: f, _):
      return evaluate(c) { condition, me in
        let clause = me[condition] as! Bool
        // TODO: deallocate temporaries.
        return clause
          ? me.run(t, then: followup)
          : f.map { me.run($0, then: followup) }
            ?? followup
      }

    case let .return(e, _):
      return evaluate(
        e, into: frame.resultAddress, then: dropResult(frame.onReturn))

    case let .block(children, _):
      return runBlockContent(
        children[...],
        then: Task { [mark=frame.allocations.count] me in
          me.deleteAllocations(
            me.frame.allocations.count - mark, then: followup)
        })

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

    case let .match(subject: subject, clauses: clauses, _):
      UNIMPLEMENTED(subject, clauses)

    case .break:
      return frame.onBreak!

    case .continue:
      return frame.onContinue!
    }
  }

  /// Runs `s` and follows up with `followup`.
  ///
  /// A convenience wrapper for `run(_:then:)` that supports cleaner syntax.
  mutating func run(_ s: Statement, followup: @escaping Task.Code) -> Task {
    run(s, then: Task(followup))
  }

  /// Executes the statements of `content` in order, then `followup`.
  mutating func runBlockContent(
    _ content: ArraySlice<Statement>, then followup: Task
  ) -> Task {
    return content.isEmpty ? followup
      : run(content.first!) { me in
          me.runBlockContent(content.dropFirst(), then: followup)
        }
  }

  mutating func `while`(
    _ c: Expression, run body: Statement, then followup: Task) -> Task
  {
    return evaluate(c) { condition, me in
      let runBody = me[condition] as! Bool
      // TODO: deallocate temporaries.
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
    _ e: Expression, then followup: @escaping FollowupWith<Address>
  ) -> Task {
    let a = memory.allocate(mutable: false)
    if tracing {
      print("\(e.site): info: allocated \(a)")
    }
    frame.allocations.push(.temporary(e, a))
    return Task { me in followup(a, &me) }
  }

  /// Destroys and reclaims memory of the `n` locally-allocated values at the
  /// top of the allocation stack.
  mutating func deleteAllocations(_ n: Int, then followup: Task) -> Task {
    // Note memory allocations/deallocations are not currently considered units
    // of work so we can do them all at once here.
    for _ in 0..<n {
      switch frame.allocations.pop()! {
      case let .variable(v):
        if tracing {
          print("\(v.site): info: deleting \(frame.locals[v]!)")
        }
        memory.delete(frame.locals[v]!)
        frame.locals[v] = nil
      case let .temporary(e, address):
        if tracing {
          print("\(e.site): info: deleting \(address)")
        }
        memory.delete(address)
      }
    }
    return followup
  }

  /// Copies the value at `source` into the `target` address and continues with
  /// `followup`.
  mutating func copy(
    from source: Address, to target: Address,
    then followup: @escaping FollowupWith<Address>
  ) -> Task {
    return initialize(target, to: self[source], then: followup)
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

  mutating func temporary(
    _ e: Expression,
    at alreadyAllocated: Address?,
    then followup: @escaping FollowupWith<Address>) -> Task
  {
    alreadyAllocated.map { a in Task { me in followup(a, &me) } }
      ?? allocate(e, then: followup)
  }
}

typealias FollowupWith<T> = (T, inout Interpreter)->Task

/// Returns a followup that drops its argument and invokes f.
fileprivate func dropResult<T>(_ f: Task) -> FollowupWith<T>
{ { _, me in f(&me) } }

extension Interpreter {
  /// Evaluates `e` (into `destination`, if supplied) and passes
  /// the address of the result on to `followup_`.
  mutating func evaluate(
    _ e: Expression, into destination: Address? = nil,
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

    switch e {
    case let .name(n):
      return evaluate(n, into: destination, then: followup)

    case let .memberAccess(m):
      UNIMPLEMENTED(m)
    case let .index(target: t, offset: i, _):
      UNIMPLEMENTED(t, i)

    case let .integerLiteral(r, _):
      return temporary(e, at: destination) { result, me in
        me.initialize(result, to: r, then: followup)
      }

    case let .booleanLiteral(r, _):
      return temporary(e, at: destination) { result, me in
        me.initialize(result, to: r, then: followup)
      }

    case let .tupleLiteral(t):
      UNIMPLEMENTED(t)

    case let .unaryOperator(x):
      return evaluate(x, into: destination, then: followup)
    case let .binaryOperator(x):
      return evaluate(x, into: destination, then: followup)
    case let .functionCall(x):
      UNIMPLEMENTED(x)
    case .intType, .boolType, .typeType:
      UNIMPLEMENTED()
    case let .functionType(f):
      UNIMPLEMENTED(f)
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
    case let b as SimpleBinding:
      let source = (frame.locals[b] ?? globals[b])!
      return destination != nil
        ? copy(from: source, to: destination!, then: followup)
        : Task { me in followup(source, &me) }
    default:
      UNIMPLEMENTED(d as Any)
    }
  }

  /// Evaluates `e` (into `destination`, if supplied) and passes the address of
  /// the result on to `followup`.
  mutating func evaluate(
    _ e: UnaryOperatorExpression, into destination: Address? = nil,
    then followup: @escaping FollowupWith<Address>
  ) -> Task {
    evaluate(e.operand) { operand, me in
      let output: Value
      switch e.operation.text {
      case "-": output = -(me[operand] as! Int)
      case "not": output = !(me[operand] as! Bool)
      default: UNREACHABLE()
      }

      return me.temporary(.unaryOperator(e), at: destination) { result, me in
        me.initialize(result, to: output, then: followup)
      }
    }
  }

  /// Evaluates `e` (into `destination`, if supplied) and passes the address of
  /// the result on to `followup`.
  mutating func evaluate(
    _ e: BinaryOperatorExpression, into destination: Address? = nil,
    then followup: @escaping FollowupWith<Address>
  ) -> Task {
    temporary(.binaryOperator(e), at: destination) { result, me in
      me.evaluate(e.lhs) { lhs, me in
        if e.operation.text == "and" && (me[lhs] as! Bool == false) {
          return me.copy(from: lhs, to: result, then: followup)
        }
        else if e.operation.text == "or" && (me[lhs] as! Bool == true) {
          return me.copy(from: lhs, to: result, then: followup)
        }
        return me.evaluate(e.rhs) { rhs, me in
          switch e.operation.text {
          case "==":
            return me.initialize(
              result, to: areEqual(me[lhs], me[rhs]), then: followup)
          case "-":
            return me.initialize(
              result, to: (me[lhs] as! Int) - (me[rhs] as! Int), then: followup)
          default:
            UNIMPLEMENTED(e)
          }
        }
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
    followup: @escaping FollowupWith<Bool>
  ) -> Task {
    switch p {
    case let .atom(t):
      return evaluate(t) { target, me in
        let matched = areEqual(me[target], me[source])
        return Task { me in followup(matched, &me) }
      }

    case let .variable(b):
      frame.locals[b] = source
      return Task { me in followup(true, &me) }

    case let .tuple(x): UNIMPLEMENTED(x)
    case let .functionCall(x): UNIMPLEMENTED(x)
    case let .functionType(x): UNIMPLEMENTED(x)
    }
  }
}
