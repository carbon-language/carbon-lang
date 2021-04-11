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

