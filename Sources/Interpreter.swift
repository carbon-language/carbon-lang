// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// The engine that executes the program
struct Interpreter {
  typealias ExitCode = Int

  /// The function execution context.
  typealias FunctionContext = (
    /// The first address allocated to the currently-executing function's frame.
    frameBase: Address,
    
    /// The place to store the currently-executing function's return value.
    resultStorage: Address,

    /// Where results should be stored when evaluating arguments to a function
    /// call.
    calleeFrameBase: Address
  )

  init(_ program: ExecutableProgram) {
    self.program = program
  }

  var //let
    program: ExecutableProgram
  var memory = Memory()
  private(set) var termination: ExitCode? = nil

  var functionContext = FunctionContext(
    frameBase: -1, resultStorage: -1, calleeFrameBase: -1)
  
  /// The stack of pending actions.
  private var todo = Stack<Action>()
}

extension Interpreter {
  /// Progress one step forward in the execution sequence, returning an exit
  /// code if the program terminated.
  mutating func step() {
    guard var current = todo.pop() else {
      termination = 0
      return
    }
    switch current.run(on: &self) {
    case .done: return
    case .spawn(let child):
      todo.push(current)
      todo.push(child)
    case .chain(to: let successor):
      todo.push(successor)
    }
  }

  /// Accesses the value of an already-evaluated expression that is valid in the
  /// currently-executing function context.
  subscript(_ e: Expression) -> Value {
    memory[program.expressionAddress[e].resolved(in: self)]
  }

  /// Initializes the storage for `e` to `v`.
  ///
  /// - Requires: e is an expression valid in the currently-executing function
  ///   context that has been allocated, is bound to `v.type`, and is not
  ///   initialized.
  mutating func initialize(_ e: Expression, to v: Value) {
    memory.initialize(program.expressionAddress[e].resolved(in: self), to: v)
  }

  /// Returns the storage for `e` to an uninitialized state.
  ///
  /// - Requires: e is an expression valid in the currently-executing function
  ///   context that has been evaluated and not deinitialized
  mutating func deinitialize(_ e: Expression) {
    memory.deinitialize(program.expressionAddress[e].resolved(in: self))
  }

  /// Accesses the value stored for the declaration of the given name.
  ///
  /// Every identifier gets mapped to a declaration, and every declaration has a
  /// corresponding value stored in memory.
  subscript(_ id: Identifier) -> Value {
    memory[
      program.declarationAddress[program.declaration[id]].resolved(in: self)]
  }
}

struct FunctionValue: Value {
  let type: Type
  let code: FunctionDefinition
}

extension Interpreter {
  mutating func pushTodo_testingOnly(_ a: Action) { todo.push(a) }
}
