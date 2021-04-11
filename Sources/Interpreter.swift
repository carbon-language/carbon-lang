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
  let program: ExecutableProgram
  var memory = Memory()
  var earlyExit: ExitCode?
  var functionContext = FunctionContext(
    frameBase: -1, resultStorage: -1, calleeFrameBase: -1)
  
  /// The stack of pending actions.
  private var todo = Stack<Action>()
}

extension Interpreter {
  /// Progress one step forward in the execution sequence, returning an exit
  /// code if the program terminated.
  mutating func step() -> ExitCode? {
    // pop the top action and run it on `self`
    if let e = earlyExit { return e }
    guard var a = todo.pop() else { return 0 }
    if let b = a.run(on: &self) {
      // If a result was returned, keep running `a` but run `b` first.
      todo.push(a)
      todo.push(b)
    }
    return nil
  }

  /// Returns the value an already-evaluated expression valid in the
  /// currently-executing function context.
  func value(_ e: Expression) -> Value {
    memory[program.expressionAddress[e].resolved(in: self)]
  }

  /// Returns the storage for `e` to an uninitialized state.
  ///
  /// - Requires: e is an expression valid in the currently-executing function
  ///   context that has been evaluated and not deinitialized
  mutating func deinitialize(_ e: Expression) {
    memory.deinitialize(program.expressionAddress[e].resolved(in: self))
  }
}

struct FunctionValue: Value {
  let type: Type
  let code: FunctionDefinition
}

