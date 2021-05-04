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

  // TODO(geoffromer): Replace these with explicit modeling of scopes
  // as part of the todo stack.
  /// A mapping from local name declarations to addresses.
  var locals: ASTDictionary<SimpleBinding, Address> = .init()
  /// A mapping from local expressions to addresses
  var temporaries: ASTDictionary<Expression, Address> = .init()
  
  /// The address that should be filled in by any `return` statements.
  var returnValueStorage: Address = -1

  /// A type that captures everything that needs to be restored after a callee
  /// returns.
  typealias FunctionContext = (
    locals: ASTDictionary<SimpleBinding, Address>,
    temporaries: ASTDictionary<Expression, Address>,
    returnValueStorage: Address)

  /// The function execution context.
  var functionContext: FunctionContext {
    get { (locals, temporaries, returnValueStorage) }
    set { (locals, temporaries, returnValueStorage) = newValue }
  }

  typealias ExitCode = Int

  /// Mapping from global declaration to addresses.
  // private(set)
  var globals: ASTDictionary<TopLevelDeclaration, Address> = .init()

  var memory = Memory()

  private var exitCodeStorage: Address = -1

  /// The stack of pending actions.
  private var todo = Stack<Action>()
}

extension Interpreter {
  mutating func start() {
    exitCodeStorage = memory.allocate(boundTo: .int, from: .empty)

    todo.push(EvaluateCall(
      call: program.entryPoint!,
      callerContext: functionContext, returnValueStorage: exitCodeStorage))
  }

  /// Progress one step forward in the execution sequence, returning an exit
  /// code if the program terminated.
  mutating func step() -> ExitCode? {
    guard var current = todo.pop() else {
      return (memory[exitCodeStorage] as! IntValue)
    }
    switch current.run(on: &self) {
    case .done: break
    case .spawn(let child):
      todo.push(current)
      todo.push(child)
    case .chain(to: let successor):
      todo.push(successor)
    case .unwind(let isSuccessor):
      while (!isSuccessor(todo.top)) { _ = todo.pop() }
    }
    return nil
  }

  /// Allocates storage for a temporary that will hold the value of `e`
  mutating func allocateTemporary(
    `for` e: Expression, boundTo t: Type, mutable: Bool = false
  ) -> Address{
    precondition(temporaries[e] == nil, "Temporary already initialized.")
    let a = memory.allocate(
      boundTo: t, from: e.site.region, mutable: false)
    temporaries[e] = a
    return a
  }

  /// Destroys any rvalue computed for `e` and removes `e` from `locals`.
  mutating func cleanUp(_ e: Expression) {
    defer { temporaries[e] = nil }
    if case .name(_) = e { return } // not an rvalue.

    let a = temporaries[e]!
    memory.deinitialize(a)
    memory.deallocate(a)
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
