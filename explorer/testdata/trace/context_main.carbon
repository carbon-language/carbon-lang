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
// CHECK:STDOUT: ********** source program **********
// CHECK:STDOUT: interface TestInterface {
// CHECK:STDOUT: }
// CHECK:STDOUT: fn Main ()-> i32 {
// CHECK:STDOUT: {
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: }
// CHECK:STDOUT: ********** resolving names **********
// CHECK:STDOUT: --- declared `TestInterface` as `interface TestInterface` in `package` (context_main.carbon:7)
// CHECK:STDOUT: --- declared `Main` as `fn Main` in `package` (context_main.carbon:11)
// CHECK:STDOUT: ** resolving decl `interface TestInterface` (context_main.carbon:7)
// CHECK:STDOUT: --- marked `TestInterface` declared but not usable in `package`
// CHECK:STDOUT: --- marked `TestInterface` usable in `package`
// CHECK:STDOUT: --- declared `Self` as `Self` in `interface TestInterface` (context_main.carbon:7)
// CHECK:STDOUT: ** finished resolving decl `interface TestInterface` (context_main.carbon:7)
// CHECK:STDOUT: ** resolving decl `fn Main` (context_main.carbon:11)
// CHECK:STDOUT: --- marked `Main` declared but not usable in `package`
// CHECK:STDOUT: --- marked `Main` usable in `package`
// CHECK:STDOUT: ** resolving stmt `{return 0;}` (context_main.carbon:11)
// CHECK:STDOUT: ** resolving stmt `return 0;` (context_main.carbon:10)
// CHECK:STDOUT: ** finished resolving stmt `return 0;` (context_main.carbon:10)
// CHECK:STDOUT: ** finished resolving stmt `{return 0;}` (context_main.carbon:11)
// CHECK:STDOUT: ** finished resolving decl `fn Main` (context_main.carbon:11)
// CHECK:STDOUT: --- resolved `Main` as `fn Main` in `package` (<Main()>:0)
// CHECK:STDOUT: ********** resolving control flow **********
// CHECK:STDOUT: --- flow-resolved return statement `return 0;` in `fn Main` (context_main.carbon:10)
// CHECK:STDOUT: ********** type checking **********
// CHECK:STDOUT: ** declaring interface TestInterface
// CHECK:STDOUT: ** finished declaring interface TestInterface
// CHECK:STDOUT: checking InterfaceDeclaration
// CHECK:STDOUT: ** checking interface TestInterface
// CHECK:STDOUT: impl declarations:{{ }}
// CHECK:STDOUT: impl declarations: bool as interface EqWith(U = bool), i32 as interface EqWith(U = i32), String as interface EqWith(U = String), i32 as interface CompareWith(U = i32), String as interface CompareWith(U = String), i32 as interface LessWith(U = i32), String as interface LessWith(U = String), i32 as interface LessEqWith(U = i32), String as interface LessEqWith(U = String), i32 as interface GreaterWith(U = i32), String as interface GreaterWith(U = String), i32 as interface GreaterEqWith(U = i32), String as interface GreaterEqWith(U = String), i32 as interface Negate, i32 as interface AddWith(U = i32), i32 as interface SubWith(U = i32), i32 as interface MulWith(U = i32), i32 as interface DivWith(U = i32), i32 as interface ModWith(U = i32), i32 as interface BitComplement, i32 as interface BitAndWith(U = i32), i32 as interface BitOrWith(U = i32), i32 as interface BitXorWith(U = i32), i32 as interface LeftShiftWith(U = i32), i32 as interface RightShiftWith(U = i32), i32 as interface Inc, i32 as interface Dec, U as interface __EqualConverter [0], (T1,) as interface As(T = (U1,)) [0, 0, 0; 1, 1, 0, 0, 0], (T1, T2) as interface As(T = (U1, U2)) [0, 0, 0; 0, 0, 1; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1], (T1, U1) as interface EqWith(U = (T2, U2)) [0, 0, 0; 0, 0, 1; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1], (T1, T2, T3) as interface As(T = (U1, U2, U3)) [0, 0, 0; 0, 0, 1; 0, 0, 2; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1; 1, 1, 0, 0, 2], T as interface ImplicitAs(T = U) [0; 1, 1, 0], T as interface As(T = U) [0; 1, 1, 0], T as interface ImplicitAs(T = U) [0; 1, 1, 0], T as interface As(T = U) [0; 1, 1, 0], T as interface AssignWith(U = U) [0; 1, 1, 0], T as interface AddAssignWith(U = U) [0; 1, 1, 0], T as interface SubAssignWith(U = U) [0; 1, 1, 0], T as interface MulAssignWith(U = U) [0; 1, 1, 0], T as interface DivAssignWith(U = U) [0; 1, 1, 0], T as interface ModAssignWith(U = U) [0; 1, 1, 0], T as interface BitAndAssignWith(U = U) [0; 1, 1, 0], T as interface BitOrAssignWith(U = U) [0; 1, 1, 0], T as interface BitXorAssignWith(U = U) [0; 1, 1, 0], T as interface LeftShiftAssignWith(U = U) [0; 1, 1, 0], T as interface RightShiftAssignWith(U = U) [0; 1, 1, 0]
// CHECK:STDOUT: ** finished checking interface TestInterface
// CHECK:STDOUT: ** declaring function Main
// CHECK:STDOUT: checking TuplePattern ()
// CHECK:STDOUT: checking IntTypeLiteral i32
// CHECK:STDOUT: (+) stack-push: i32 .0.
// CHECK:STDOUT: (+) stack-push: i32 .0.
// CHECK:STDOUT: --- step exp i32 .0. (context_main.carbon:9) --->
// CHECK:STDOUT: (-) stack-pop: i32 .0.
// CHECK:STDOUT: (-) stack-pop: i32 .1. {{\[\[}}i32]]
// CHECK:STDOUT: ** finished declaring function Main of type fn () -> i32
// CHECK:STDOUT: checking FunctionDeclaration
// CHECK:STDOUT: ** checking function Main
// CHECK:STDOUT: impl declarations:{{ }}
// CHECK:STDOUT: impl declarations: bool as interface EqWith(U = bool), i32 as interface EqWith(U = i32), String as interface EqWith(U = String), i32 as interface CompareWith(U = i32), String as interface CompareWith(U = String), i32 as interface LessWith(U = i32), String as interface LessWith(U = String), i32 as interface LessEqWith(U = i32), String as interface LessEqWith(U = String), i32 as interface GreaterWith(U = i32), String as interface GreaterWith(U = String), i32 as interface GreaterEqWith(U = i32), String as interface GreaterEqWith(U = String), i32 as interface Negate, i32 as interface AddWith(U = i32), i32 as interface SubWith(U = i32), i32 as interface MulWith(U = i32), i32 as interface DivWith(U = i32), i32 as interface ModWith(U = i32), i32 as interface BitComplement, i32 as interface BitAndWith(U = i32), i32 as interface BitOrWith(U = i32), i32 as interface BitXorWith(U = i32), i32 as interface LeftShiftWith(U = i32), i32 as interface RightShiftWith(U = i32), i32 as interface Inc, i32 as interface Dec, U as interface __EqualConverter [0], (T1,) as interface As(T = (U1,)) [0, 0, 0; 1, 1, 0, 0, 0], (T1, T2) as interface As(T = (U1, U2)) [0, 0, 0; 0, 0, 1; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1], (T1, U1) as interface EqWith(U = (T2, U2)) [0, 0, 0; 0, 0, 1; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1], (T1, T2, T3) as interface As(T = (U1, U2, U3)) [0, 0, 0; 0, 0, 1; 0, 0, 2; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1; 1, 1, 0, 0, 2], T as interface ImplicitAs(T = U) [0; 1, 1, 0], T as interface As(T = U) [0; 1, 1, 0], T as interface ImplicitAs(T = U) [0; 1, 1, 0], T as interface As(T = U) [0; 1, 1, 0], T as interface AssignWith(U = U) [0; 1, 1, 0], T as interface AddAssignWith(U = U) [0; 1, 1, 0], T as interface SubAssignWith(U = U) [0; 1, 1, 0], T as interface MulAssignWith(U = U) [0; 1, 1, 0], T as interface DivAssignWith(U = U) [0; 1, 1, 0], T as interface ModAssignWith(U = U) [0; 1, 1, 0], T as interface BitAndAssignWith(U = U) [0; 1, 1, 0], T as interface BitOrAssignWith(U = U) [0; 1, 1, 0], T as interface BitXorAssignWith(U = U) [0; 1, 1, 0], T as interface LeftShiftAssignWith(U = U) [0; 1, 1, 0], T as interface RightShiftAssignWith(U = U) [0; 1, 1, 0]
// CHECK:STDOUT: checking Block {
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: checking ReturnExpression return 0;
// CHECK:STDOUT: checking IntLiteral 0
// CHECK:STDOUT: ** finished checking function Main
// CHECK:STDOUT: checking CallExpression Main()
// CHECK:STDOUT: checking IdentifierExpression Main
// CHECK:STDOUT: checking TupleLiteral ()
// CHECK:STDOUT: checking call to function of type fn () -> i32
// CHECK:STDOUT: with arguments of type: ()
// CHECK:STDOUT: performing argument deduction for bindings:{{ }}
// CHECK:STDOUT: deduction succeeded with results: {}
// CHECK:STDOUT: ********** resolving unformed variables **********
// CHECK:STDOUT: *** resolving-unformed in decl `interface TestInterface` (context_main.carbon:7)
// CHECK:STDOUT: *** resolving-unformed in decl `fn Main` (context_main.carbon:11)
// CHECK:STDOUT: *** resolving-unformed in stmt `{return 0;}` (context_main.carbon:11)
// CHECK:STDOUT: *** resolving-unformed in stmt `return 0;` (context_main.carbon:10)
// CHECK:STDOUT: ********** printing declarations **********
// CHECK:STDOUT: interface TestInterface {
// CHECK:STDOUT: }
// CHECK:STDOUT: fn Main ()-> i32 {
// CHECK:STDOUT: {
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: }
// CHECK:STDOUT: ********** starting execution **********
// CHECK:STDOUT: ********** initializing globals **********
// CHECK:STDOUT: (+) stack-push: interface TestInterface {
// CHECK:STDOUT: }
// CHECK:STDOUT:  .0.
// CHECK:STDOUT: --- step decl interface TestInterface .0. (context_main.carbon:7) --->
// CHECK:STDOUT: (-) stack-pop: interface TestInterface {
// CHECK:STDOUT: }
// CHECK:STDOUT:  .0.
// CHECK:STDOUT: (+) stack-push: fn Main ()-> i32 {
// CHECK:STDOUT: {
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: }
// CHECK:STDOUT:  .0.
// CHECK:STDOUT: --- step decl fn Main .0. (context_main.carbon:11) --->
// CHECK:STDOUT: (-) stack-pop: fn Main ()-> i32 {
// CHECK:STDOUT: {
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: }
// CHECK:STDOUT:  .0.
// CHECK:STDOUT: ********** calling main function **********
// CHECK:STDOUT: (+) stack-push: Main() .0.
// CHECK:STDOUT: (+) stack-push: Main() .0.
// CHECK:STDOUT: --- step exp Main() .0. (<Main()>:0) --->
// CHECK:STDOUT: (+) stack-push: Main .0.
// CHECK:STDOUT: (+) stack-push: Main .0.
// CHECK:STDOUT: --- step exp Main .0. (<Main()>:0) --->
// CHECK:STDOUT: (-) stack-pop: Main .0.
// CHECK:STDOUT: (-) stack-pop: Main .1. {{\[\[}}fun<Main>]]
// CHECK:STDOUT: --- step exp Main() .1. (<Main()>:0) --->
// CHECK:STDOUT: calling function: fun<Main>
// CHECK:STDOUT: match pattern ()
// CHECK:STDOUT: from value expression with value ()
// CHECK:STDOUT: (+) stack-push: .0. {}
// CHECK:STDOUT: (+) stack-push: {return 0;} .0.
// CHECK:STDOUT: --- step stmt {return 0;} .0. (context_main.carbon:11) --->
// CHECK:STDOUT: (+) stack-push: return 0; .0.
// CHECK:STDOUT: --- step stmt return 0; .0. (context_main.carbon:10) --->
// CHECK:STDOUT: (+) stack-push: 0 .0.
// CHECK:STDOUT: (+) stack-push: 0 .0.
// CHECK:STDOUT: --- step exp 0 .0. (context_main.carbon:10) --->
// CHECK:STDOUT: (-) stack-pop: 0 .0.
// CHECK:STDOUT: (-) stack-pop: 0 .1. {{\[\[}}0]]
// CHECK:STDOUT: --- step stmt return 0; .1. (context_main.carbon:10) --->
// CHECK:STDOUT: (-) stack-pop: return 0; .1. {{\[\[}}0]]
// CHECK:STDOUT: (-) stack-pop: {return 0;} .1. {}
// CHECK:STDOUT: (-) stack-pop: .0. {}
// CHECK:STDOUT: (+) stack-push: clean up.0. {}
// CHECK:STDOUT: (+) stack-push: clean up.0. {}
// CHECK:STDOUT: (-) stack-pop: clean up.0. {}
// CHECK:STDOUT: (-) stack-pop: clean up.0. {}
// CHECK:STDOUT: --- step exp Main() .2. (<Main()>:0) --->
// CHECK:STDOUT: (-) stack-pop: Main() .2. {{\[\[}}fun<Main>, 0]] {}
// CHECK:STDOUT: (+) stack-push: clean up.0. {}
// CHECK:STDOUT: (-) stack-pop: clean up.0. {}
// CHECK:STDOUT: (-) stack-pop: Main() .1. {{\[\[}}0]]
// CHECK:STDOUT: interpreter result: 0
// CHECK:STDOUT: ********** printing timing **********
// CHECK:STDOUT: Time elapsed in ExecProgram: {{\d+}}ms
// CHECK:STDOUT: Time elapsed in AnalyzeProgram: {{\d+}}ms
// CHECK:STDOUT: Time elapsed in AddPrelude: {{\d+}}ms
// CHECK:STDOUT: Time elapsed in Parse: {{\d+}}ms
// CHECK:STDOUT: result: 0
