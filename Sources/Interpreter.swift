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

  // FIXME comment all the things
  struct Scope {
    // FIXME pull out to top level?
    enum Kind {
      case temporary, local, function
    }

    let kind: Kind

    /// The index in `todo` of the action that created this scope
    let actionIndex: Int

    /// The storage owned by this scope
    var owned: [Address] = []

    /// The value that `returnValueStorage` should be restored
    /// to when this scope is unwound. Should be non-nil only for function
    /// scopes.
    let callerReturnValueStorage: Address?
  }
  
  private var scopes: Stack<Scope> = .init()
  
  private(set) var returnValueStorage: Address? = nil

  /// Begins a new scope of the specified kind (which should not be `.function`).
  /// The scope will automatically end and be cleaned up when control is returned to `todo.top`,
  /// but it can also be ended explicitly by calling `endScope`.
  mutating func beginScope(kind: Scope.Kind) {
    assert(kind != .function, "Use beginFunctionScope instead")
    scopes.push(Scope(kind: kind, actionIndex: todo.count,
                      callerReturnValueStorage: nil))
  }

  /// Begins a scope for a new function call, whose return value will be stored in
  /// `returnValueStorage`.
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
    // FIXME: integrate scope handling into action?
    todo.push(NoopAction())
    beginScope(kind: .temporary)
    // FIXME: should this be part of the scope?
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
            // FIXME why isn't this actionIndex == todo.count?
            scope.actionIndex >= todo.count {
        endScopeUnchecked()
      }
    case .spawn(let child):
      todo.push(current)
      todo.push(child)
    case .chain(to: let successor):
      // FIXME: explain why we don't end scopes:
      // chaining can be delegation, but is it always?
      todo.push(successor)
    case .unwindToFunctionCall:
      while scopes.top.kind != .function {
        endScopeUnchecked()
      }
      endScopeUnchecked()

      while todo.count > scopes.top.actionIndex {
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
    UNIMPLEMENTED
    //return locals[AnyDeclaration(d)] ?? globals[d] ?? fatal("\(d) has no value")
  }
}

extension Interpreter {
  mutating func pushTodo_testingOnly(_ a: Action) { todo.push(a) }
}
