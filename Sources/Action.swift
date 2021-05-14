// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

enum Followup {
  /// All finished. Any scopes associated with the current Action will
  /// be cleaned up.
  case done

  /// Still working, start child.
  case spawn(_ child: Action)

  /// Stop running this action, and have `successor` take its place.
  /// Any scopes associated with the current Action will be transferred
  /// to `successor`.
  case delegate(_ successor: Action)

  /// All finished with the current function call. All scopes down to
  /// and including the uppermost function scope will be cleaned up,
  /// and execution will resume with the action associated with the
  /// uppermost remaining scope.
  // TODO: Should this be parameterized by Scope.Kind, rather than
  // function-specific?
  case unwindToFunctionCall
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
  let target: Address

  /// Creates an instance that evaluates `source` and initializes `target` with the result
  init(_ source: Expression, into target: Address) {
    self.source = source
    self.target = target
  }

  mutating func run(on state: inout Interpreter) -> Followup {
    switch source {
    case .name(let id):
      switch state.program.definition[id] {
      case let f as FunctionDefinition:
        state.memory.initialize(target, to: FunctionValue(type: .error, code: f))
        return .done
      case is SimpleBinding:
        state.memory.initialize(target, to: state.memory[state.address(of: id)])
        // N.B. of all expressions, this one doesn't need to be destroyed.
        return .done
      case nil:
        fatalError("No definition for '\(id)'")
      default: UNIMPLEMENTED()
      }
    case .memberAccess(_): UNIMPLEMENTED()
    case .index(target: _, offset: _, _): UNIMPLEMENTED()
    case .integerLiteral(let value, _):
      state.memory.initialize(target, to: value)
      return .done
    case .booleanLiteral(_, _): UNIMPLEMENTED()
    case .tupleLiteral(_): UNIMPLEMENTED()
    case .unaryOperator(_): UNIMPLEMENTED()
    case .binaryOperator(_): UNIMPLEMENTED()
    case .functionCall(_): UNIMPLEMENTED()
    case .intType(_): UNIMPLEMENTED()
    case .boolType(_): UNIMPLEMENTED()
    case .typeType(_): UNIMPLEMENTED()
    case .functionType(_): UNIMPLEMENTED()
    }
  }
}

struct EvaluateTupleLiteral: Action {
  let source: TupleLiteral
  var elements: [Address] = []
  
  init(_ source: TupleLiteral) {
    self.source = source
  }
  
  mutating func run(on state: inout Interpreter) -> Followup {
    let i = elements.count
    if i == source.count { return .done }
    let e = source[i].payload
    let a = state.allocateTemporary(for: e, boundTo: .error, mutable: false)
    elements.append(a)
    return .spawn(Evaluate(e, into: a))
  }
}

struct NoopAction: Action {
  mutating func run(on state: inout Interpreter) -> Followup {
    return .done
  }
}

struct Execute: Action {
  let source: Statement
  
  init(_ source: Statement) {
    self.source = source
  }

  mutating func run(on state: inout Interpreter) -> Followup {
    switch source {
    case .expressionStatement(_, _): UNIMPLEMENTED()
    case .assignment(target: _, source: _, _): UNIMPLEMENTED()
    case .initialization(_): UNIMPLEMENTED()
    case .if(condition: _, thenClause: _, elseClause: _, _): UNIMPLEMENTED()
    case .return(let operand, _):
      return .delegate(ExecuteReturn(operand))
    case .block(let b, _):
      return .delegate(ExecuteBlock(remaining: b[...]))
    case .while(condition: _, body: _, _): UNIMPLEMENTED()
    case .match(subject: _, clauses: _, _): UNIMPLEMENTED()
    case .break(_): UNIMPLEMENTED()
    case .continue(_): UNIMPLEMENTED()
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

struct ExecuteReturn: Action {
  init(_ operand: Expression) {
    self.operand = operand
  }

  /// The operand of the `return` statement
  let operand: Expression
  
  /// Notional coroutine state.
  private enum Step: Int {
    case start, evaluateOperand, transferControl
  }
  
  /// The current activity; `nil` means we haven't been started yet.
  private var step: Step = .start

  mutating func run(on state: inout Interpreter) -> Followup {
    switch step {
    case .start:
      step = .evaluateOperand
      return .spawn(Evaluate(operand, into: state.returnValueStorage!))
    case .evaluateOperand:
      step = .transferControl
      return .unwindToFunctionCall
    case .transferControl:
      fatalError("Unreachable")
    }
  }
}
