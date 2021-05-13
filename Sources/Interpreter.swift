// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// The engine that executes the program
struct Interpreter {
  /// Creates an instance for executing `program`.
  init(_ program: ExecutableProgram) {
    self.program = program
  }

  /// The program being executed.
  var //let
    program: ExecutableProgram


  struct Scope {
    enum Kind {
      // TODO: we'll eventually want at least 3 kinds of
      // local scope as well: loop (the scope of a `break`),
      // loop body (the scope of a `continue`), and block.
      case temporary, function
    }

    let kind: Kind

    /// The index in `todo` of the `Action` this scope is associated with. That's
    /// normally the `Action` that created this scope, but the association can
    /// be transferred using `Followup.delegate`.
    let actionIndex: Int

    /// The storage owned by this scope
    var owned: [Address] = []

    /// The value that `returnValueStorage` should be restored
    /// to when this scope is unwound. Should be non-nil only for function
    /// scopes.
    let callerReturnValueStorage: Address?
  }

  /// The stack of scopes that execution has entered, and not yet left.
  private var scopes: Stack<Scope> = .init()
  
  private(set) var returnValueStorage: Address? = nil

  /// Begins a new scope of the specified kind, associated with the currently-running `Action`.
  /// The interpreter will automatically end the scope when the associated `Action` is
  /// done, but it can also be ended explicitly by calling `endScope`. Conversely, the
  /// interpreter will discard the associated `Action` if the scope is unwound.
  /// To begin a function scope, use `beginFunctionScope` instead of this method.
  mutating func beginScope(kind: Scope.Kind) {
    assert(kind != .function, "Use beginFunctionScope instead")
    let actionIndex = todo.count

    // TODO: It isn't necessarily a bug for a single action to be
    // associated with multiple nested scopes, and in fact it may
    // be necessary for e.g. loop actions to create separate scopes
    // to distinguish `break` and `continue`. When and if those
    // situations arise, we will need a more careful rule about which
    // actions are discarded during unwinding, and possibly some
    // mechanism for notifying an action when some but not all of its
    // scopes have been unwound. In the meantime, we forbid those
    // situations for simplicity.
    assert(scopes.isEmpty || actionIndex != scopes.top.actionIndex)

    scopes.push(Scope(kind: kind, actionIndex: actionIndex,
                      callerReturnValueStorage: nil))
  }

  /// Begins a scope for a new function call, whose return value will be stored in
  /// `returnValueStorage`. Otherwise equivalent to `beginScope`.
  mutating func beginFunctionScope(returnValueStorage: Address) {
    scopes.push(Scope(kind: .function, actionIndex: todo.count,
                      callerReturnValueStorage: self.returnValueStorage))
    self.returnValueStorage = returnValueStorage
  }

  /// Explicitly ends the current innermost scope, which must have been
  /// started by the action currently being processed.
  mutating func endScope() {
    assert(scopes.top.actionIndex == todo.count,
           "Can't end scope started by another Action")
    endScopeUnchecked()
  }

  private mutating func endScopeUnchecked() {
    let scope = scopes.pop()!
    for a in scope.owned {
      memory.deinitialize(a)
      memory.deallocate(a)
    }
    if scope.kind == .function {
      returnValueStorage = scope.callerReturnValueStorage
    }
  }

  typealias ExitCode = Int

  /// Mapping from global declaration to addresses.
  // private(set)
  var globals: ASTDictionary<TopLevelDeclaration, Address> = .init()

  var memory = Memory()

  private var exitCodeStorage: Address? = nil

  /// The stack of pending actions.
  private var todo = Stack<Action>()
}

extension Interpreter {
  mutating func start() {
    // EvaluateCall expects to be executed in a temporary scope,
    // which owns the results of evaluating the callee and argument
    // expressions, so we provide one here. Every scope is associated
    // with an action, so we put a no-op action on the todo stack.

    beginScope(kind: .temporary)
    todo.push(NoopAction())

    exitCodeStorage = memory.allocate(boundTo: .int, from: .empty)
    todo.push(EvaluateCall(
      call: program.entryPoint!, returnValueStorage: exitCodeStorage!))
  }

  enum Status {
    case running
    case exited(_ exitCode: ExitCode)
  }

  /// Progress one step forward in the execution sequence, returning an exit
  /// code if the program terminated.
  mutating func step() -> Status {
    guard var current = todo.pop() else {
      let exitCode = memory[exitCodeStorage!] as! IntValue
      memory.assertCleanupDone(except: [exitCodeStorage!])
      return .exited(exitCode)
    }
    switch current.run(on: &self) {
    case .done:
      while case .some(let scope) = scopes.queryTop,
            scope.actionIndex == todo.count {
        endScopeUnchecked()
      }
    case .spawn(let child):
      todo.push(current)
      todo.push(child)
    case .delegate(to: let successor):
      todo.push(successor)
    case .unwindToFunctionCall:
      // End all scopes within the current function scope
      while scopes.top.kind != .function {
        endScopeUnchecked()
      }

      // End the current function scope
      let outermostUnwoundAction = scopes.top.actionIndex
      endScopeUnchecked()

      // Discard actions associated with the ended scopes
      while todo.count - 1 >= outermostUnwoundAction {
        _ = todo.pop()
      }
    }
    return .running
  }

  /// Allocates storage for a temporary that will hold the value of `e`
  mutating func allocateTemporary(
    `for` e: Expression, boundTo t: Type, mutable: Bool = false
  ) -> Address{
    let a = memory.allocate(
      boundTo: t, from: e.site.region, mutable: false)
    assert(scopes.top.kind == .temporary)
    scopes.top.owned.append(a)
    return a
  }

  /// Accesses the value stored for the declaration of the given name.
  subscript(_ name: Identifier) -> Value {
    return memory[address(of: name)]
  }

  /// Accesses the address of the declaration for the given name.
  func address(of name: Identifier) -> Address {
    let d = program.definition[name] ?? fatal("\(name) not defined")
    _ = d
    UNIMPLEMENTED()
    //return locals[AnyDeclaration(d)] ?? globals[d] ?? fatal("\(d) has no value")
  }
}

extension Interpreter {
  mutating func pushTodo_testingOnly(_ a: Action) { todo.push(a) }
}
