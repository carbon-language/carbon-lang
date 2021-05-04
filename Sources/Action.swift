// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

enum Followup {
  case done                        // All finished.
  case spawn(_ child: Action)      // Still working, start child.
  case chain(_ successor: Action)  // All finished, start successor.
  // All finished with this action *and* all actions between it and
  // the nearest lower action on the stack that matches `isSuccessor`
  case unwind(_ isSuccessor: (Action)->Bool)
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
        fatalError("No definition for '\(id.text)'")
      default: UNIMPLEMENTED
      }
    case .getField(_, _, _): UNIMPLEMENTED
    case .index(target: _, offset: _, _): UNIMPLEMENTED
    case .integerLiteral(let value, _):
      state.memory.initialize(target, to: value)
      return .done
    case .booleanLiteral(_, _): UNIMPLEMENTED
    case .tupleLiteral(_): UNIMPLEMENTED
    case .unaryOperator(operation: _, operand: _, _): UNIMPLEMENTED
    case .binaryOperator(operation: _, lhs: _, rhs: _, _): UNIMPLEMENTED
    case .functionCall(_): UNIMPLEMENTED
    case .intType(_): UNIMPLEMENTED
    case .boolType(_): UNIMPLEMENTED
    case .typeType(_): UNIMPLEMENTED
    case .functionType(_): UNIMPLEMENTED
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

struct CleanUp: Action {
  init(_ target: Expression) {
    self.target = target
  }
  let target: Expression

  mutating func run(on engine: inout Interpreter) -> Followup {
    engine.cleanUp(target)
    return .done
  }
}

struct CleanUpTupleLiteral: Action {
  let target: TupleLiteral
  var nextElement: Int = 0

  init(_ target: TupleLiteral) {
    self.target = target
  }

  mutating func run(on state: inout Interpreter) -> Followup {
    if nextElement == target.count { return .done }
    defer { nextElement += 1 }
    return .spawn(CleanUp(target[nextElement].payload))
  }
}

struct Execute: Action {
  let source: Statement
  
  init(_ source: Statement) {
    self.source = source
  }

  mutating func run(on state: inout Interpreter) -> Followup {
    switch source {
    case .expressionStatement(_, _): UNIMPLEMENTED
    case .assignment(target: _, source: _, _): UNIMPLEMENTED
    case .initialization(_): UNIMPLEMENTED
    case .if(condition: _, thenClause: _, elseClause: _, _): UNIMPLEMENTED
    case .return(let operand, _):
      return .chain(ExecuteReturn(operand))
    case .block(let b, _):
      return .chain(ExecuteBlock(remaining: b[...]))
    case .while(condition: _, body: _, _): UNIMPLEMENTED
    case .match(subject: _, clauses: _, _): UNIMPLEMENTED
    case .break(_): UNIMPLEMENTED
    case .continue(_): UNIMPLEMENTED
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
      return .spawn(Evaluate(operand, into: state.returnValueStorage))
    case .evaluateOperand:
      step = .transferControl
      return .unwind({ $0 is EvaluateCall })
    case .transferControl:
      fatalError("Unreachable")
    }
  }
}
