// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// Function call evaluator.
struct EvaluateCall: Action {
  /// Function call expression being evaluated
  let call: FunctionCall<Expression>
  /// Interpreter context to be restored after call completes.
  let callerContext: Interpreter.FunctionContext
  /// Where the result of the call shall be stored.
  let returnValueStorage: Address
  /// Where the callee value is stored
  var calleeStorage: Address = -1

  init(
    call: FunctionCall<Expression>,
    callerContext: Interpreter.FunctionContext,
    returnValueStorage: Address)
  {
    self.call = call
    self.callerContext = callerContext
    self.returnValueStorage = returnValueStorage
  }

  /// Notional coroutine state.
  ///
  /// A suspended Call action on the todo list is either nascent (with a `nil`
  /// step), or it's doing what the step name indicates (via sub-actions).
  private enum Step: Int {
    case evaluateCallee, evaluateArguments,
         runBody,
         cleanUpArguments, cleanUpCallee
  }

  /// The current activity; `nil` means we haven't been started yet.
  private var step: Step? = nil

  // information stashed across steps.
  /// The callee.
  private var calleeCode: FunctionDefinition!

  /// Updates the interpreter state and optionally spawns a sub-action.
  mutating func run(on state: inout Interpreter) -> Followup {
    let nextStep = Step(rawValue: step.map { $0.rawValue + 1 } ?? 0)!
    // Auto-advance on exit.
    defer { step = nextStep }

    switch nextStep {
    case .evaluateCallee:
      calleeStorage = state.allocateTemporary(
        for: call.callee, boundTo: .error, mutable: false)
      return .spawn(Evaluate(call.callee, into: calleeStorage))
      
    case .evaluateArguments:
      calleeCode = (state.memory[calleeStorage] as! FunctionValue).code
      
      return .spawn(EvaluateTupleLiteral(call.arguments))
      
    case .runBody:
      // Prepare the context for the callee
      state.returnValueStorage = returnValueStorage
      if (calleeCode.parameters.elements.count > 0) { UNIMPLEMENTED() }
      return .spawn(Execute(calleeCode.body!))

    case .cleanUpArguments:
      state.functionContext = callerContext
      return .spawn(CleanUpTupleLiteral(call.arguments))

    case .cleanUpCallee:
      return .chain(CleanUp(call.callee))
    }
  }
}
