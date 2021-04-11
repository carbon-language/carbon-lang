// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct Call: Action {
  let callee: Expression
  let arguments: TupleLiteral
  let callerContext: Interpreter.FunctionContext
  let resultStorage: Address

  enum Step: Int {
    case start, evaluateCallee, evaluateArguments, invoke
  }
  var step: Step = .start

  // information stashed across steps.
  var calleeCode: FunctionDefinition!
  var frameSize: Int!
  
  private var arity: Int { arguments.body.count }
  
  /// Updates the interpreter state, optionally returning an action to be
  /// pushed onto its todo stack.
  ///
  /// If the result is non-nil, `self` will be run again after the resulting
  /// action is completed.
  mutating func run(on state: inout Interpreter) -> Action? {
    defer {
      if let nextStep = Step(rawValue: step.rawValue + 1) { step = nextStep }
    }

    // The step we're coming from
    switch step {
    case .start:
      return Evaluate(callee)
      
    case .evaluateCallee:
      calleeCode = (state.value(callee) as! FunctionValue).code
      
      // Prepare the callee's frame
      let frame = state.program.frameLayout[calleeCode]
      frameSize = frame.count
      
      state.functionContext.calleeFrameBase = state.memory.nextAddress
      for (type, mutable, site) in frame {
        _ = state.memory.allocate(boundTo: type, from: site, mutable: mutable)
      }
      
      return EvaluateTupleLiteral(arguments)
      
    case .evaluateArguments:
      // Prepare the context for the callee
      state.functionContext.resultStorage = resultStorage
      state.functionContext.frameBase = state.functionContext.calleeFrameBase
      
      return Execute(calleeCode.body.body!)

    case .invoke:
      let calleeFrameBase = state.functionContext.frameBase
      
      // Restore the caller's context.
      state.functionContext = callerContext
      
      // Deinitialize parameters.
      for i in 0..<arity {
        state.memory.deinitialize(state.functionContext.frameBase + i)
      }
      // Deallocate the callee's frame.
      for a in calleeFrameBase..<(calleeFrameBase + frameSize) {
        state.memory.deallocate(a)
      }
      // Deinitialize subexpression temporaries
      state.deinitialize(callee)
      for a in arguments.body {
        state.deinitialize(a.value)
      }
      return nil
    }
  }
}
