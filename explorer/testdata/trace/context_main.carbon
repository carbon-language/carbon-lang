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
// ARGS: --trace_file=- --trace_phase=all --trace_file_context=main %s
// AUTOUPDATE

// CHECK:STDOUT: * * * * * * * * * *  source program  * * * * * * * * * *
// CHECK:STDOUT: --------------------------------------------------------
// CHECK:STDOUT: interface TestInterface {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn Main () -> i32
// CHECK:STDOUT: {
// CHECK:STDOUT:   return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT:
// CHECK:STDOUT: * * * * * * * * * *  resolving names  * * * * * * * * * *
// CHECK:STDOUT: ---------------------------------------------------------
// CHECK:STDOUT: ==> declared `TestInterface` as `interface TestInterface` in `package` (context_main.carbon:7)
// CHECK:STDOUT: ==> declared `Main` as `fn Main` in `package` (context_main.carbon:11)
// CHECK:STDOUT: ->> resolving decl `interface TestInterface` (context_main.carbon:7)
// CHECK:STDOUT: ==> marked `TestInterface` declared but not usable in `package`
// CHECK:STDOUT: ==> marked `TestInterface` usable in `package`
// CHECK:STDOUT: ==> declared `Self` as `Self` in `interface TestInterface` (context_main.carbon:7)
// CHECK:STDOUT: <<- finished resolving decl `interface TestInterface` (context_main.carbon:7)
// CHECK:STDOUT: ->> resolving decl `fn Main` (context_main.carbon:11)
// CHECK:STDOUT: ==> marked `Main` declared but not usable in `package`
// CHECK:STDOUT: ==> marked `Main` usable in `package`
// CHECK:STDOUT: ->> resolving stmt `{ ... }` (context_main.carbon:11)
// CHECK:STDOUT: ->> resolving stmt `return ...;` (context_main.carbon:10)
// CHECK:STDOUT: <<- finished resolving stmt `return ...;` (context_main.carbon:10)
// CHECK:STDOUT: <<- finished resolving stmt `{ ... }` (context_main.carbon:11)
// CHECK:STDOUT: <<- finished resolving decl `fn Main` (context_main.carbon:11)
// CHECK:STDOUT: ==> resolved `Main` as `fn Main` in `package` (<Main()>:0)
// CHECK:STDOUT:
// CHECK:STDOUT:
// CHECK:STDOUT: * * * * * * * * * *  resolving control flow  * * * * * * * * * *
// CHECK:STDOUT: ----------------------------------------------------------------
// CHECK:STDOUT: ==> flow-resolved return statement `return 0;` in `fn Main` (context_main.carbon:10)
// CHECK:STDOUT:
// CHECK:STDOUT:
// CHECK:STDOUT: * * * * * * * * * *  type checking  * * * * * * * * * *
// CHECK:STDOUT: -------------------------------------------------------
// CHECK:STDOUT:
// CHECK:STDOUT: *** declaration at (context_main.carbon:7)
// CHECK:STDOUT: ```
// CHECK:STDOUT: interface TestInterface {
// CHECK:STDOUT: }
// CHECK:STDOUT: ```
// CHECK:STDOUT: ->> declaring `interface TestInterface` (context_main.carbon:7)
// CHECK:STDOUT: <<- finished declaring `interface TestInterface` (context_main.carbon:7)
// CHECK:STDOUT:
// CHECK:STDOUT: *** type checking declaration at (context_main.carbon:7)
// CHECK:STDOUT: ```
// CHECK:STDOUT: interface TestInterface {
// CHECK:STDOUT: }
// CHECK:STDOUT: ```
// CHECK:STDOUT: ->> checking InterfaceDeclaration `interface TestInterface` (context_main.carbon:7)
// CHECK:STDOUT: ->> checking `interface TestInterface` (context_main.carbon:7)
// CHECK:STDOUT: ==> impl declarations for `interface TestInterface` (context_main.carbon:7)
// CHECK:STDOUT:     []
// CHECK:STDOUT:     [`bool` as `interface EqWith(U = bool)`,
// CHECK:STDOUT:     `i32` as `interface EqWith(U = i32)`,
// CHECK:STDOUT:     `String` as `interface EqWith(U = String)`,
// CHECK:STDOUT:     `i32` as `interface CompareWith(U = i32)`,
// CHECK:STDOUT:     `String` as `interface CompareWith(U = String)`,
// CHECK:STDOUT:     `i32` as `interface LessWith(U = i32)`,
// CHECK:STDOUT:     `String` as `interface LessWith(U = String)`,
// CHECK:STDOUT:     `i32` as `interface LessEqWith(U = i32)`,
// CHECK:STDOUT:     `String` as `interface LessEqWith(U = String)`,
// CHECK:STDOUT:     `i32` as `interface GreaterWith(U = i32)`,
// CHECK:STDOUT:     `String` as `interface GreaterWith(U = String)`,
// CHECK:STDOUT:     `i32` as `interface GreaterEqWith(U = i32)`,
// CHECK:STDOUT:     `String` as `interface GreaterEqWith(U = String)`,
// CHECK:STDOUT:     `i32` as `interface Negate`,
// CHECK:STDOUT:     `i32` as `interface AddWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface SubWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface MulWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface DivWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface ModWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface BitComplement`,
// CHECK:STDOUT:     `i32` as `interface BitAndWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface BitOrWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface BitXorWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface LeftShiftWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface RightShiftWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface Inc`,
// CHECK:STDOUT:     `i32` as `interface Dec`,
// CHECK:STDOUT:     `U` as `interface __EqualConverter` [0],
// CHECK:STDOUT:     `(T1,)` as `interface As(T = (U1,))` [0, 0, 0; 1, 1, 0, 0, 0],
// CHECK:STDOUT:     `(T1, T2)` as `interface As(T = (U1, U2))` [0, 0, 0; 0, 0, 1; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1],
// CHECK:STDOUT:     `(T1, U1)` as `interface EqWith(U = (T2, U2))` [0, 0, 0; 0, 0, 1; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1],
// CHECK:STDOUT:     `(T1, T2, T3)` as `interface As(T = (U1, U2, U3))` [0, 0, 0; 0, 0, 1; 0, 0, 2; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1; 1, 1, 0, 0, 2],
// CHECK:STDOUT:     `T` as `interface ImplicitAs(T = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface As(T = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface ImplicitAs(T = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface As(T = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface AssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface AddAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface SubAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface MulAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface DivAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface ModAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface BitAndAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface BitOrAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface BitXorAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface LeftShiftAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface RightShiftAssignWith(U = U)` [0; 1, 1, 0]]
// CHECK:STDOUT: <<- finished checking `interface TestInterface` (context_main.carbon:7)
// CHECK:STDOUT:
// CHECK:STDOUT: *** declaration at (context_main.carbon:11)
// CHECK:STDOUT: ```
// CHECK:STDOUT: fn Main () -> i32
// CHECK:STDOUT: {
// CHECK:STDOUT:   return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT: ```
// CHECK:STDOUT: ->> declaring function `Main` (context_main.carbon:11)
// CHECK:STDOUT: ->> checking TuplePattern `()` (context_main.carbon:9)
// CHECK:STDOUT: ->> checking IntTypeLiteral `i32` (context_main.carbon:9)
// CHECK:STDOUT: >[] stack-push: ValueExpressionAction pos: 0 `i32` (context_main.carbon:9)
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 0 `i32` (context_main.carbon:9) --->
// CHECK:STDOUT: >[] stack-push: ExpressionAction pos: 0 `i32` (context_main.carbon:9)
// CHECK:STDOUT: ->> step ExpressionAction pos: 0 `i32` (context_main.carbon:9) --->
// CHECK:STDOUT: <[] stack-pop:  ExpressionAction pos: 0 `i32` (context_main.carbon:9)
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 1 `i32` results: [`i32`]  (context_main.carbon:9) --->
// CHECK:STDOUT: <[] stack-pop:  ValueExpressionAction pos: 1 `i32` results: [`i32`]  (context_main.carbon:9)
// CHECK:STDOUT: ->> finished declaring function `Main` of type `fn () -> i32` (context_main.carbon:11)
// CHECK:STDOUT:
// CHECK:STDOUT: *** type checking declaration at (context_main.carbon:11)
// CHECK:STDOUT: ```
// CHECK:STDOUT: fn Main () -> i32
// CHECK:STDOUT: {
// CHECK:STDOUT:   return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT: ```
// CHECK:STDOUT: ->> checking FunctionDeclaration `fn Main` (context_main.carbon:11)
// CHECK:STDOUT: ->> checking function `Main` (context_main.carbon:11)
// CHECK:STDOUT: ==> impl declarations for `fn Main` (context_main.carbon:11)
// CHECK:STDOUT:     []
// CHECK:STDOUT:     [`bool` as `interface EqWith(U = bool)`,
// CHECK:STDOUT:     `i32` as `interface EqWith(U = i32)`,
// CHECK:STDOUT:     `String` as `interface EqWith(U = String)`,
// CHECK:STDOUT:     `i32` as `interface CompareWith(U = i32)`,
// CHECK:STDOUT:     `String` as `interface CompareWith(U = String)`,
// CHECK:STDOUT:     `i32` as `interface LessWith(U = i32)`,
// CHECK:STDOUT:     `String` as `interface LessWith(U = String)`,
// CHECK:STDOUT:     `i32` as `interface LessEqWith(U = i32)`,
// CHECK:STDOUT:     `String` as `interface LessEqWith(U = String)`,
// CHECK:STDOUT:     `i32` as `interface GreaterWith(U = i32)`,
// CHECK:STDOUT:     `String` as `interface GreaterWith(U = String)`,
// CHECK:STDOUT:     `i32` as `interface GreaterEqWith(U = i32)`,
// CHECK:STDOUT:     `String` as `interface GreaterEqWith(U = String)`,
// CHECK:STDOUT:     `i32` as `interface Negate`,
// CHECK:STDOUT:     `i32` as `interface AddWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface SubWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface MulWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface DivWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface ModWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface BitComplement`,
// CHECK:STDOUT:     `i32` as `interface BitAndWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface BitOrWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface BitXorWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface LeftShiftWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface RightShiftWith(U = i32)`,
// CHECK:STDOUT:     `i32` as `interface Inc`,
// CHECK:STDOUT:     `i32` as `interface Dec`,
// CHECK:STDOUT:     `U` as `interface __EqualConverter` [0],
// CHECK:STDOUT:     `(T1,)` as `interface As(T = (U1,))` [0, 0, 0; 1, 1, 0, 0, 0],
// CHECK:STDOUT:     `(T1, T2)` as `interface As(T = (U1, U2))` [0, 0, 0; 0, 0, 1; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1],
// CHECK:STDOUT:     `(T1, U1)` as `interface EqWith(U = (T2, U2))` [0, 0, 0; 0, 0, 1; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1],
// CHECK:STDOUT:     `(T1, T2, T3)` as `interface As(T = (U1, U2, U3))` [0, 0, 0; 0, 0, 1; 0, 0, 2; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1; 1, 1, 0, 0, 2],
// CHECK:STDOUT:     `T` as `interface ImplicitAs(T = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface As(T = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface ImplicitAs(T = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface As(T = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface AssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface AddAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface SubAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface MulAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface DivAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface ModAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface BitAndAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface BitOrAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface BitXorAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface LeftShiftAssignWith(U = U)` [0; 1, 1, 0],
// CHECK:STDOUT:     `T` as `interface RightShiftAssignWith(U = U)` [0; 1, 1, 0]]
// CHECK:STDOUT:
// CHECK:STDOUT: *** type checking stmt at (context_main.carbon:11)
// CHECK:STDOUT: ```
// CHECK:STDOUT: {
// CHECK:STDOUT:   return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT: ```
// CHECK:STDOUT: ->> checking Block `{ ... }` (context_main.carbon:11)
// CHECK:STDOUT:
// CHECK:STDOUT: *** type checking stmt at (context_main.carbon:10)
// CHECK:STDOUT: ```
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: ```
// CHECK:STDOUT: ->> checking ReturnExpression `return ...;` (context_main.carbon:10)
// CHECK:STDOUT: ->> checking IntLiteral `0` (context_main.carbon:10)
// CHECK:STDOUT: ->> finished checking function `Main` (context_main.carbon:11)
// CHECK:STDOUT: ->> checking CallExpression `Main()` (<Main()>:0)
// CHECK:STDOUT: ->> checking IdentifierExpression `Main` (<Main()>:0)
// CHECK:STDOUT: ->> checking TupleLiteral `()` (<Main()>:0)
// CHECK:STDOUT: ->> checking call to function of type `fn () -> i32` with arguments of type `()` (<Main()>:0)
// CHECK:STDOUT: ->> performing argument deduction for bindings: []
// CHECK:STDOUT: ==> deduction succeeded with results: []
// CHECK:STDOUT:
// CHECK:STDOUT:
// CHECK:STDOUT: * * * * * * * * * *  resolving unformed variables  * * * * * * * * * *
// CHECK:STDOUT: ----------------------------------------------------------------------
// CHECK:STDOUT: ->> resolving-unformed in decl `interface TestInterface` (context_main.carbon:7)
// CHECK:STDOUT: ->> resolving-unformed in decl `fn Main` (context_main.carbon:11)
// CHECK:STDOUT: ->> resolving-unformed in stmt `{ ... }` (context_main.carbon:11)
// CHECK:STDOUT: ->> resolving-unformed in stmt `return ...;` (context_main.carbon:10)
// CHECK:STDOUT:
// CHECK:STDOUT:
// CHECK:STDOUT: * * * * * * * * * *  printing declarations  * * * * * * * * * *
// CHECK:STDOUT: ---------------------------------------------------------------
// CHECK:STDOUT: interface TestInterface {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn Main () -> i32
// CHECK:STDOUT: {
// CHECK:STDOUT:   return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT:
// CHECK:STDOUT: * * * * * * * * * *  starting execution  * * * * * * * * * *
// CHECK:STDOUT: ------------------------------------------------------------
// CHECK:STDOUT:
// CHECK:STDOUT: - - - - -  initializing globals  - - - - -
// CHECK:STDOUT: ------------------------------------------
// CHECK:STDOUT: >[] stack-push: DeclarationAction pos: 0 `interface TestInterface` (context_main.carbon:7)
// CHECK:STDOUT: ->> step DeclarationAction pos: 0 `interface TestInterface` (context_main.carbon:7) --->
// CHECK:STDOUT:
// CHECK:STDOUT: *** declaration at (context_main.carbon:7)
// CHECK:STDOUT: ```
// CHECK:STDOUT: interface TestInterface {
// CHECK:STDOUT: }
// CHECK:STDOUT: ```
// CHECK:STDOUT: <[] stack-pop:  DeclarationAction pos: 0 `interface TestInterface` (context_main.carbon:7)
// CHECK:STDOUT: >[] stack-push: DeclarationAction pos: 0 `fn Main` (context_main.carbon:11)
// CHECK:STDOUT: ->> step DeclarationAction pos: 0 `fn Main` (context_main.carbon:11) --->
// CHECK:STDOUT:
// CHECK:STDOUT: *** declaration at (context_main.carbon:11)
// CHECK:STDOUT: ```
// CHECK:STDOUT: fn Main () -> i32
// CHECK:STDOUT: {
// CHECK:STDOUT:   return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT: ```
// CHECK:STDOUT: <[] stack-pop:  DeclarationAction pos: 0 `fn Main` (context_main.carbon:11)
// CHECK:STDOUT:
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
// CHECK:STDOUT: === match pattern `()`
// CHECK:STDOUT:     from value expression with value `()`
// CHECK:STDOUT: >[] stack-push: ScopeAction pos: 0  scope: [] (None)
// CHECK:STDOUT: >[] stack-push: StatementAction pos: 0 `{ ... }` (context_main.carbon:11)
// CHECK:STDOUT: ->> step StatementAction pos: 0 `{ ... }` (context_main.carbon:11) --->
// CHECK:STDOUT:
// CHECK:STDOUT: *** statement at (context_main.carbon:11)
// CHECK:STDOUT: ```
// CHECK:STDOUT: {
// CHECK:STDOUT:   return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT: ```
// CHECK:STDOUT: >[] stack-push: StatementAction pos: 0 `return ...;` (context_main.carbon:10)
// CHECK:STDOUT: ->> step StatementAction pos: 0 `return ...;` (context_main.carbon:10) --->
// CHECK:STDOUT:
// CHECK:STDOUT: *** statement at (context_main.carbon:10)
// CHECK:STDOUT: ```
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: ```
// CHECK:STDOUT: >[] stack-push: ValueExpressionAction pos: 0 `0` (context_main.carbon:10)
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 0 `0` (context_main.carbon:10) --->
// CHECK:STDOUT: >[] stack-push: ExpressionAction pos: 0 `0` (context_main.carbon:10)
// CHECK:STDOUT: ->> step ExpressionAction pos: 0 `0` (context_main.carbon:10) --->
// CHECK:STDOUT: <[] stack-pop:  ExpressionAction pos: 0 `0` (context_main.carbon:10)
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 1 `0` results: [`0`]  (context_main.carbon:10) --->
// CHECK:STDOUT: <[] stack-pop:  ValueExpressionAction pos: 1 `0` results: [`0`]  (context_main.carbon:10)
// CHECK:STDOUT: ->> step StatementAction pos: 1 `return ...;` results: [`0`]  (context_main.carbon:10) --->
// CHECK:STDOUT:
// CHECK:STDOUT: *** statement at (context_main.carbon:10)
// CHECK:STDOUT: ```
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: ```
// CHECK:STDOUT: <[] stack-pop:  StatementAction pos: 1 `return ...;` results: [`0`]  (context_main.carbon:10)
// CHECK:STDOUT: <[] stack-pop:  StatementAction pos: 1 `{ ... }` scope: [] (context_main.carbon:11)
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
// CHECK:STDOUT:
// CHECK:STDOUT:
// CHECK:STDOUT: * * * * * * * * * *  printing timing  * * * * * * * * * *
// CHECK:STDOUT: ---------------------------------------------------------
// CHECK:STDOUT: Time elapsed in ExecProgram: {{\d+}}ms
// CHECK:STDOUT: Time elapsed in AnalyzeProgram: {{\d+}}ms
// CHECK:STDOUT: Time elapsed in AddPrelude: {{\d+}}ms
// CHECK:STDOUT: Time elapsed in Parse: {{\d+}}ms
// CHECK:STDOUT: result: 0
