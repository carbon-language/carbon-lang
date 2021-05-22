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

  /// Creates an instance that runs `program`
  init(_ program: ExecutableProgram) {
    self.program = program

    frame = CallFrame(
      resultAddress: memory.allocate(),
      onReturn: Task { me in
        // When main returns, the program is finished.
        me.running = false
        return Task { _ in fatalError("Terminated program can't continue.") }
      })

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
      UNIMPLEMENTED(i)

    case let .if(condition: c, thenClause: trueClause, elseClause: maybeFalse, _):
      UNIMPLEMENTED(c, trueClause, maybeFalse as Any)

    case let .return(e, _):
      return evaluate(
        e, into: frame.resultAddress, then: frame.onReturn)

    case let .block(children, _):
      let bottomOfBlock=frame.allocations.count
      return runBlockContent(
        children[...],
        then: Task { $0.cleanAllocations(above: bottomOfBlock, then: followup)})

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

  /// A convenience wrapper for `run(_:then:)` that supports cleaner syntax.
  mutating func run(_ s: Statement, followup: @escaping Task.Code) -> Task {
    run(s, then: Task(followup))
  }

  /// Notionally executes the statements of `content` in order… then `followup`.
  mutating func runBlockContent(
    _ content: ArraySlice<Statement>, then followup: Task
  ) -> Task {
    return content.isEmpty ? followup
      : run(content.first!) { me in
          me.runBlockContent(content.dropFirst(), then: followup)
        }
  }

  /// Destroys and reclaims memory of zero or more locally-allocated values at
  /// the top of the allocation stack.
  mutating func cleanAllocations(above mark: Int, then followup: Task) -> Task {
    // Note memory allocations/deallocations are not currently considered units
    // of work so we can do them all at once here.
    while frame.allocations.count > mark {
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
      UNIMPLEMENTED(x)
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
}
