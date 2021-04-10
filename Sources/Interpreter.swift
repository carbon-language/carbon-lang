// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

protocol Action {
  /// Updates the interpreter state and optionally returning an action to be
  /// pushed onto its todo stack.
  func run(on i: inout Interpreter) -> Action?
}

struct RelativeAddress: Hashable {
  /// The lexical function nesting level of the function whose frame spans
  /// the indicated address.
  ///
  /// 0 indicates that the address is a global variable, 1 indicates a top-level
  /// function, 2 indicates a function or lambda nested in a top-level function,
  /// etc.
  var lexicalFrame: Int
  /// The offset of the indicated address from the lexical frame's base address.
  var offsetInFrame: Int

  /// Returns the absolute address corresponding to `self` in `i`.
  func resolved(in i: Interpreter) -> Address {
    return lexicalFrameBase[lexicalFrame] + offsetInFrame
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
    // pop the top action, run it on `self`, and push any result that returns.
    if let e = earlyExit { return e }
    guard let a = todo.pop() else { return 0 }
    if let b = a.run(on: &self) {
      todo.push(b)
    }
    return nil
  }

  /// The first address allocated to each lexical frame.
  ///
  /// This is not a call stack; it has one entry for the current lexical scope,
  /// and one for each enclosing The frame address of the currently-executing
  /// function is always `lexicalFrameAddress.last!`.  The frame where globals
  /// live is always `lexicalFrameAddress[0]`; expect that value to be zero.
  /// Intermediate lexical frames are represented by the chain of function
  /// scopes inside which the current function scope is nested (for lambdas and
  /// other nested functions).
  var lexicalFrameBase: Stack<Address>

  /// The stack of pending actions.
  private var todo: Stack<Action>
}

struct CallFunction: Action {
  let callee: FunctionDefinition

  /// Updates the interpreter state and optionally returning an action to be
  /// pushed onto its todo stack.
  func run(on i: inout Interpreter) -> Action? {
    if i.program.nestingLevel[callee] == i.
  }
}
