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

  /// The kind of lookup we are doing.
  enum Context {
    case global // The address of a global symbol.
    case local  // The address of a parameter, local variable, or temporary.
    // expect to add "this-relative" for lambdas and method access.
  }
  
  let base: Context
  let offset: Int
  
  /// Returns the absolute address corresponding to `self` in `i`.
  func resolved(in i: Interpreter) -> Address {
    (base == .global ? 0 : i.frameBase) + offset
  }
}

/// The engine that executes the program
struct Interpreter {
  typealias ExitCode = Int

  let program: ExecutableProgram
  var memory: Memory
  var earlyExit: ExitCode?

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

  /// The address in memory of the first address allocated to the
  /// currently-executing function's frame.
  var frameBase: Address

  /// The stack of pending actions.
  private var todo: Stack<Action>
}

struct CallFunction: Action {
  let callee: Expression
  let arguments: TupleLiteral

  /// Updates the interpreter state, optionally returning an action to be
  /// pushed onto its todo stack.
  func run(on i: inout Interpreter) -> Action? {
    return nil
  }
}
