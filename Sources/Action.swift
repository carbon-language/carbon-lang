// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

enum Followup {
  case done                        // All finished.
  case spawn(_ child: Action)      // Still working, start child.
  case chain(_ successor: Action)  // All finished, start successor.
}

protocol Action {
  /// Updates the interpreter state, optionally returning an action to be
  /// executed as a subpart of this action.
  ///
  /// If the result is non-nil, `self` will be run again after the resulting
  /// action is completed.
  mutating func run(on i: inout Interpreter) -> Followup
}

struct Evaluate: Action {
  let source: Expression
  
  init(_ source: Expression) {
    self.source = source
  }

  mutating func run(on state: inout Interpreter) -> Followup {
    switch source.body {
    case .variable(let id):
      state.initialize(source, to: state[id])
      return .done
    default: fatalError("implement me.\n\(source)")
    }
  }
}

struct EvaluateTupleLiteral: Action {
  let source: TupleLiteral
  var nextElement: Int = 0
  
  init(_ source: TupleLiteral) {
    self.source = source
  }
  
  mutating func run(on state: inout Interpreter) -> Followup {
    if nextElement == source.body.count { return .done }
    defer { nextElement += 1 }
    return .spawn(Evaluate(source.body[nextElement].value))
  }
}

struct Execute: Action {
  let source: Statement
  
  init(_ source: Statement) {
    self.source = source
  }

  mutating func run(on state: inout Interpreter) -> Followup {
    switch source.body {
    case .block(let b):
      return .chain(ExecuteBlock(remaining: b[...]))
    default:
      fatalError("implement me.\n\(source)")
    }
  }
}

struct ExecuteBlock: Action {
  var remaining: ArraySlice<Statement>
  mutating func run(on state: inout Interpreter) -> Followup {
    guard let s = remaining.popFirst() else { return .done }
    return .spawn(Execute(s))
  }
}
