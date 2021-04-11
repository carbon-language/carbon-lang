// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Evaluates a function call.
struct Call: Action {
  /// Which function to call.
  let callee: Expression
  /// Argument expressions.
  let arguments: TupleLiteral
  /// Interpreter context to be restored after call completes.
  let callerContext: Interpreter.FunctionContext
  /// Where the result of the call shall be stored.
  let resultStorage: Address

  /// Notional coroutine state.
  ///
  /// A suspended Call action (on the todo list) is either `.nascent`, or it's
  /// doing what the step name indicates (via sub-actions).
  private enum Step: Int {
    case nascent, evaluatingCallee, evaluatingArguments, invoking
  }
  private var step: Step = .nascent

  // information stashed across steps.
  private var calleeCode: FunctionDefinition!
  private var frameSize: Int!
  
  private var arity: Int { arguments.body.count }
  
  /// Updates the interpreter state and optionally spawns a sub-action.
  mutating func run(on state: inout Interpreter) -> Action? {
    defer { // advance to next step automatically upon exit
      if let nextStep = Step(rawValue: step.rawValue + 1) { step = nextStep }
    }

    switch step {
    case .nascent:
      return Evaluate(callee)
      
    case .evaluatingCallee:
      calleeCode = (state.value(callee) as! FunctionValue).code
      
      // Prepare the callee's frame
      let frame = state.program.frameLayout[calleeCode]
      frameSize = frame.count
      
      state.functionContext.calleeFrameBase = state.memory.nextAddress
      for (type, mutable, site) in frame {
        _ = state.memory.allocate(boundTo: type, from: site, mutable: mutable)
      }
      
      return EvaluateTupleLiteral(arguments)
      
    case .evaluatingArguments:
      // Prepare the context for the callee
      state.functionContext.resultStorage = resultStorage
      state.functionContext.frameBase = state.functionContext.calleeFrameBase
      
      return Execute(calleeCode.body.body!)

    case .invoking:
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
