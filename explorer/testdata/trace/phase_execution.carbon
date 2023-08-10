// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExplorerTest api;

interface TestInterface {}

fn Main() -> i32 {
  return 0;
}

// Place checks after code so that line numbers are stable, reducing merge
// conflicts.
// ARGS: --trace_file=- --trace_phase=execution %s
// AUTOUPDATE

// CHECK:STDOUT: * * * * * * * * * *  starting execution  * * * * * * * * * *
// CHECK:STDOUT: ------------------------------------------------------------
// CHECK:STDOUT: - - - - -  initializing globals  - - - - -
// CHECK:STDOUT: ------------------------------------------
// CHECK:STDOUT: >[] stack-push: DeclarationAction pos: 0 `interface TestInterface` (phase_execution.carbon:7)
// CHECK:STDOUT: ->> step DeclarationAction pos: 0 `interface TestInterface` (phase_execution.carbon:7) --->
// CHECK:STDOUT: *** declaration at (phase_execution.carbon:7)
// CHECK:STDOUT: ```
// CHECK:STDOUT: interface TestInterface {
// CHECK:STDOUT: }
// CHECK:STDOUT: ```
// CHECK:STDOUT: <[] stack-pop:  DeclarationAction pos: 0 `interface TestInterface` (phase_execution.carbon:7)
// CHECK:STDOUT: >[] stack-push: DeclarationAction pos: 0 `fn Main` (phase_execution.carbon:11)
// CHECK:STDOUT: ->> step DeclarationAction pos: 0 `fn Main` (phase_execution.carbon:11) --->
// CHECK:STDOUT: *** declaration at (phase_execution.carbon:11)
// CHECK:STDOUT: ```
// CHECK:STDOUT: fn Main ()-> i32 {
// CHECK:STDOUT: {
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: }
// CHECK:STDOUT: ```
// CHECK:STDOUT: <[] stack-pop:  DeclarationAction pos: 0 `fn Main` (phase_execution.carbon:11)
// CHECK:STDOUT: - - - - -  calling main function  - - - - -
// CHECK:STDOUT: -------------------------------------------
// CHECK:STDOUT: >[] stack-push: ValueExpressionAction pos: 0 `Main()` (<Main()>:0)
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 0 `Main()` (<Main()>:0) --->
// CHECK:STDOUT: >[] stack-push: ExpressionAction pos: 0 `Main()` (<Main()>:0)
// CHECK:STDOUT: ->> step ExpressionAction pos: 0 `Main()` (<Main()>:0) --->
// CHECK:STDOUT: >[] stack-push: ValueExpressionAction pos: 0 `Main` (<Main()>:0)
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 0 `Main` (<Main()>:0) --->
// CHECK:STDOUT: >[] stack-push: ExpressionAction pos: 0 `Main` (<Main()>:0)
// CHECK:STDOUT: ->> step ExpressionAction pos: 0 `Main` (<Main()>:0) --->
// CHECK:STDOUT: <[] stack-pop:  ExpressionAction pos: 0 `Main` (<Main()>:0)
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 1 `Main` results: [`fun<Main>`]  (<Main()>:0) --->
// CHECK:STDOUT: <[] stack-pop:  ValueExpressionAction pos: 1 `Main` results: [`fun<Main>`]  (<Main()>:0)
// CHECK:STDOUT: ->> step ExpressionAction pos: 1 `Main()` results: [`fun<Main>`]  scope: [] (<Main()>:0) --->
// CHECK:STDOUT: -() calling function: fun<Main>
// CHECK:STDOUT: match pattern ()
// CHECK:STDOUT: from value expression with value ()
// CHECK:STDOUT: >[] stack-push: ScopeAction pos: 0  scope: [] (None)
// CHECK:STDOUT: >[] stack-push: StatementAction pos: 0 `{return 0;}` (phase_execution.carbon:11)
// CHECK:STDOUT: ->> step StatementAction pos: 0 `{return 0;}` (phase_execution.carbon:11) --->
// CHECK:STDOUT: *** statement at (phase_execution.carbon:11)
// CHECK:STDOUT: ```
// CHECK:STDOUT: {
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ```
// CHECK:STDOUT: >[] stack-push: StatementAction pos: 0 `return 0;` (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step StatementAction pos: 0 `return 0;` (phase_execution.carbon:10) --->
// CHECK:STDOUT: *** statement at (phase_execution.carbon:10)
// CHECK:STDOUT: ```
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: ```
// CHECK:STDOUT: >[] stack-push: ValueExpressionAction pos: 0 `0` (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 0 `0` (phase_execution.carbon:10) --->
// CHECK:STDOUT: >[] stack-push: ExpressionAction pos: 0 `0` (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step ExpressionAction pos: 0 `0` (phase_execution.carbon:10) --->
// CHECK:STDOUT: <[] stack-pop:  ExpressionAction pos: 0 `0` (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 1 `0` results: [`0`]  (phase_execution.carbon:10) --->
// CHECK:STDOUT: <[] stack-pop:  ValueExpressionAction pos: 1 `0` results: [`0`]  (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step StatementAction pos: 1 `return 0;` results: [`0`]  (phase_execution.carbon:10) --->
// CHECK:STDOUT: *** statement at (phase_execution.carbon:10)
// CHECK:STDOUT: ```
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: ```
// CHECK:STDOUT: <[] stack-pop:  StatementAction pos: 1 `return 0;` results: [`0`]  (phase_execution.carbon:10)
// CHECK:STDOUT: <[] stack-pop:  StatementAction pos: 1 `{return 0;}` scope: [] (phase_execution.carbon:11)
// CHECK:STDOUT: <[] stack-pop:  ScopeAction pos: 0  scope: [] (None)
// CHECK:STDOUT: >[] stack-push: CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: >[] stack-push: CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: ->> step CleanUpAction pos: 0  scope: [] (stack cleanup:1) --->
// CHECK:STDOUT: <[] stack-pop:  CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: ->> step CleanUpAction pos: 0  scope: [] (stack cleanup:1) --->
// CHECK:STDOUT: <[] stack-pop:  CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: ->> step ExpressionAction pos: 2 `Main()` results: [`fun<Main>`, `0`]  scope: [] (<Main()>:0) --->
// CHECK:STDOUT: <[] stack-pop:  ExpressionAction pos: 2 `Main()` results: [`fun<Main>`, `0`]  scope: [] (<Main()>:0)
// CHECK:STDOUT: >[] stack-push: CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: ->> step CleanUpAction pos: 0  scope: [] (stack cleanup:1) --->
// CHECK:STDOUT: <[] stack-pop:  CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 1 `Main()` results: [`0`]  (<Main()>:0) --->
// CHECK:STDOUT: <[] stack-pop:  ValueExpressionAction pos: 1 `Main()` results: [`0`]  (<Main()>:0)
// CHECK:STDOUT: ==> interpreter result: 0
// CHECK:STDOUT: result: 0
