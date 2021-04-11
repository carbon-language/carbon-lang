// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

protocol Action {
  /// Updates the interpreter state, optionally returning an action to be
  /// executed as a subpart of this action.
  ///
  /// If the result is non-nil, `self` will be run again after the resulting
  /// action is completed.
  mutating func run(on i: inout Interpreter) -> Action?
}

/// The address of a (global or local) symbol, or local temporary, with respect
/// to the currently-executing function context.
struct RelativeAddress: Hashable {
  init(_ base: Context, _ offset: Int) {
    self.base = base
    self.offset = offset
  }

  /// Symbolic base of this address.
  enum Context {
    case global // global symbol.
    case local  // A parameter, local variable, or temporary.
    case callee // An argument to the callee.
    // expect to add "this-relative" for lambdas and method access.
  }
  
  let base: Context
  let offset: Int
  
  /// Returns the absolute address corresponding to `self` in `i`.
  func resolved(in i: Interpreter) -> Address {
    switch base {
    case .global: return offset
    case .local: return offset + i.functionContext.frameBase
    case .callee: return offset + i.functionContext.calleeFrameBase
    }
  }
}

/// The engine that executes the program
struct Interpreter {
  typealias ExitCode = Int

  /// The function execution context.
  typealias FunctionContext = (
    /// The address in memory of the first address allocated to the
    /// currently-executing function's frame.
    frameBase: Address,
    
    /// The place to store the currently-executing function's return value.
    resultStorage: Address,

    /// Where results should be stored when evaluating arguments to a function
    /// call.
    calleeFrameBase: Address
  )
  
  let program: ExecutableProgram
  var memory: Memory
  var earlyExit: ExitCode?
  var functionContext: FunctionContext
  
  /// The stack of pending actions.
  private var todo: Stack<Action>
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

struct Evaluate: Action {
  let source: Expression
  
  init(_ source: Expression) {
    self.source = source
  }
  mutating func run(on state: inout Interpreter) -> Action? {
    fatalError("implement me.")
  }
}

struct EvaluateTupleLiteral: Action {
  let source: TupleLiteral
  var nextElement: Int = 0
  
  init(_ source: TupleLiteral) {
    self.source = source
  }
  
  mutating func run(on state: inout Interpreter) -> Action? {
    if nextElement == source.body.count { return nil }
    defer { nextElement += 1 }
    return Evaluate(source.body[nextElement].value)
  }
}

struct Execute: Action {
  let source: Statement
  
  init(_ source: Statement) {
    self.source = source
  }
  mutating func run(on state: inout Interpreter) -> Action? {
    fatalError("implement me.")
  }
}

/*
struct NoOp: Action {
  func run(on state: inout Interpreter) -> Action? { nil }
}
 */

