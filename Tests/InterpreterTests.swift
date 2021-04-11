// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import XCTest
@testable import CarbonInterpreter

final class TestEvaluateCall: XCTestCase {
  func testMinimal() {
    let ast = try! "fn main() -> Int {}".parsedAsCarbon(fromFile: "main.6c")
    
    func topColumns(_ r: ClosedRange<Int>) -> SourceRegion {
      SourceRegion(
        fileName: "<TOP>",
        .init(line: 1, column: r.lowerBound)
          ..< .init(line: 1, column: r.upperBound + 1))
    }

    let mainID = Identifier("main", topColumns(1...4))
    let arguments = TupleLiteral([], topColumns(5...6))

    /*
    var engine = Interpreter()
    
    let mainType = Type.function(parameterTypes: [], returnType: .void)
    
    let mainAddress = engine.memory.allocate(boundTo: mainType, from: .empty)
    guard case .function(let code) = ast[0].body else {
      fatalError("unexpected AST: \(ast)")
    }
    
    engine.memory.initialize(
      mainAddress, to: FunctionValue(type: mainType, code: code))
    
    let resultAddress = engine.memory.allocate(boundTo: .void, from: .empty)

    var call = EvaluateCall(
      callee: Expression(.variable(mainID), mainID.site),
      arguments: arguments,
      callerContext: engine.functionContext,
      resultStorage: resultAddress)
      
     */
  }
  
}

final class InterpreterTests: XCTestCase {
  
}
