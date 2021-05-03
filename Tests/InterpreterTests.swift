// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

/*
final class TestEvaluateCall: XCTestCase {
  func testMinimal() {
    guard let ast = CheckNoThrow(try "fn main() -> Int {}".parsedAsCarbon()),
          let exe = CheckNoThrow(try ExecutableProgram(_parsedProgram: ast))
    else { return }

    /// Return the site covering the given columns of line 1 in a mythical
    /// file called "<TOP>".
    func topColumns(_ r: ClosedRange<Int>) -> ASTSite {
      ASTSite(
        devaluing: SourceRegion(
          fileName: "<TOP>",
          .init(line: 1, column: r.lowerBound)
            ..< .init(line: 1, column: r.upperBound + 1)))
    }

    // Much of the following code will eventually be centralized in Interpreter,
    // but for the time being we're laboriously setting up a runnable state.

    // Prepare AST fragments for a call to main.
    let mainID = Identifier(text: "main", site: topColumns(1...4))
    let mainExpression = Expression.name(mainID)
    let arguments = TupleLiteral([], topColumns(5...6))
    let mainType = Type.function(parameterTypes: [], returnType: .int)

    var engine = Interpreter(exe)

    // Store the value of the main function.  All declarations have an address
    // in memory, inlcluding engine.main.
    let mainDeclarationAddress
      = engine.memory.allocate(boundTo: mainType, from: exe.main.site.region)
    engine.memory.initialize(
      mainDeclarationAddress, to: FunctionValue(type: mainType, code: exe.main))

    // Poke in some values that should really be computed by semantic analysis.
    // These pokes are the reason for various temporary `var //let` declarations
    // you may see in Interpreter and ExecutableProgram.
    let mainDeclaration = ast[0]
    engine.program.declaration[mainID] = mainDeclaration
    engine.globals[mainDeclaration] = mainDeclarationAddress

    // Allocate an address for the return value.
    let resultAddress = engine.memory.allocate(boundTo: .int, from: .empty)

    let call = EvaluateCall(
      callee: mainExpression, arguments: arguments,
      callerContext: engine.functionContext, returnValueStorage: resultAddress)

    engine.pushTodo_testingOnly(call)
    while engine.termination == nil {
      engine.step()
    }
  }
}

final class InterpreterTests: XCTestCase {
  
}
*/
