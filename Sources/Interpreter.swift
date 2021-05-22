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

  /// Notionally exits the running program.
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
}

extension Interpreter {
  /// Notionally executes `s`, and then, absent interesting control flow,
  /// `followup`.
  ///
  /// An example of interesting control flow is a return statement, which
  /// ignores any `followup` and exits the current function instead.
  ///
  /// In fact this function only executes one “unit of work” and packages the
  /// rest of executing `s` (if any), and whatever follows that, into the
  /// returned `Task`.
  mutating func run(_ s: Statement, then followup: Task) -> Task {
    switch s {
    case let .expressionStatement(e, _):
      UNIMPLEMENTED(e)

    case let .assignment(target: t, source: s, _):
      UNIMPLEMENTED(t, s)

    case let .initialization(i):
      // This can only be cleaned up completely on scope exit because variables
      // are being bound to (parts of) the value.  Whether it's possible to tear
      // down unused *parts* of the value earlier is unknown.
      let rhsAddress = memory.allocate(mutable: true)
      return evaluate(i.initializer, into: rhsAddress) { me in
        let rhs = me.memory[rhsAddress]
        return me.match(i.bindings, to: rhs) { matched, me in
          matched ? followup : me.error(
            i.bindings, "Initialization pattern not matched by \(rhs)")
        }
      }

    case let .if(condition: c, thenClause: trueClause, elseClause: maybeFalse, _):
      UNIMPLEMENTED(c, trueClause, maybeFalse as Any)

    case let .return(e, _):
      return evaluate(
        e, into: frame.resultAddress, then: frame.onReturn)

    case let .block(children, _):
      return runBlockContent(
        children[...],
        then: Task { [mark=frame.allocations.count] me in
          me.delete(me.frame.allocations.count - mark, then: followup)
        })

    case let .while(condition: c, body: body, _):
      UNIMPLEMENTED(c, body)

    case let .match(subject: subject, clauses: clauses, _):
      UNIMPLEMENTED(subject, clauses)

    case .break:
      UNIMPLEMENTED()

    case .continue:
      UNIMPLEMENTED()
    }
  }

  /// Notionally runs `s` and follows up with `followup`.
  ///
  /// A convenience wrapper for `run(_:then:)` that supports cleaner syntax.
  mutating func run(_ s: Statement, followup: @escaping Task.Code) -> Task {
    run(s, then: Task(followup))
  }

  /// Notionally executes the statements of `content` in order, then `followup`.
  mutating func runBlockContent(
    _ content: ArraySlice<Statement>, then followup: Task
  ) -> Task {
    return content.isEmpty ? followup
      : run(content.first!) { me in
          me.runBlockContent(content.dropFirst(), then: followup)
        }
  }

  /// Destroys and reclaims memory of the `n` locally-allocated values at the
  /// top of the allocation stack.
  mutating func delete(_ n: Int, then followup: Task) -> Task {
    // Note memory allocations/deallocations are not currently considered units
    // of work so we can do them all at once here.
    for _ in 0..<n {
      switch frame.allocations.pop()! {
      case let .variable(v):
        memory.delete(frame.locals[v]!)
        frame.locals[v] = nil
      case let .temporary(_, address):
        memory.delete(address)
      }
    }
    return followup
  }
}

extension Interpreter {
  /// Notationally evaluates `e`, placing the result in `destination`, and
  /// continues with `followup`.
  mutating func evaluate(
    _ e: Expression, into destination: Address, then followup: Task
  ) -> Task {
    switch e {
    case let .name(v):
      UNIMPLEMENTED(v)
    case let .memberAccess(m):
      UNIMPLEMENTED(m)
    case let .index(target: t, offset: i, _):
      UNIMPLEMENTED(t, i)
    case let .integerLiteral(r, _):
      memory.initialize(destination, to: r)
    case let .booleanLiteral(r, _):
      memory.initialize(destination, to: r)
    case let .tupleLiteral(t):
      UNIMPLEMENTED(t)

    case let .unaryOperator(x):
      return allocate(x.operand) { opAddress, me in
        me.evaluate(x.operand, into: opAddress) { me in
          switch x.operation.text {
          case "-":
            me.memory.initialize(
              destination, to: -(me.memory[opAddress] as! Int))
          case "not":
            me.memory.initialize(
              destination, to: !(me.memory[opAddress] as! Bool))
          default: UNREACHABLE()
          }
          // Reclaim opAddress
          return me.delete(1, then: followup)
        }
      }
    case let .binaryOperator(x):
      UNIMPLEMENTED(x)
    case let .functionCall(x):
      UNIMPLEMENTED(x)
    case .intType, .boolType, .typeType:
      UNIMPLEMENTED()
    case let .functionType(f):
      UNIMPLEMENTED(f)
    }
    return followup
  }

  /// Notationally evaluates `e`, placing the result in `destination`, and
  /// continues with `followup`.
  ///
  /// A convenience wrapper for `run(_:then:)` that supports cleaner syntax.
  mutating func evaluate(
    _ e: Expression, into destination: Address, followup: @escaping Task.Code
  ) -> Task {
    evaluate(e, into: destination, then: Task(followup))
  }

  /// Allocates an address for the result of evaluating `e`, passing it on to
  /// `followup` along with `self`.
  mutating func allocate(
    _ e: Expression, followup: @escaping (Address, inout Self)->Task
  ) -> Task {
    let a = memory.allocate(mutable: false)
    frame.allocations.push(.temporary(e, a))
    return Task { me in followup(a, &me) }
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
  /// Notionally matches `p` to `x`, binding variables in `p` to the
  /// corresponding parts of `x`, calling `followup` with an indication of
  /// whether the match was successful.
  mutating func match(
    _ p: Pattern, to x: Value,
    followup: @escaping (_ matched: Bool, inout Self)->Task
  ) -> Task {
    switch p {
    case let .atom(e):
      return allocate(e) { eAddress, me in
        me.evaluate(e, into: eAddress) { me in
          let matched = areEqual(me.memory[eAddress], x)
          return me.delete(1, then: Task { me in followup(matched, &me) })
        }
      }

    case let .variable(x): UNIMPLEMENTED(x)
    case let .tuple(x): UNIMPLEMENTED(x)
    case let .functionCall(x): UNIMPLEMENTED(x)
    case let .functionType(x): UNIMPLEMENTED(x)
    }
  }
}
