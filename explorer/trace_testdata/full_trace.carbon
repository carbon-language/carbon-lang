// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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


package ExplorerTest api;

// CHECK:STDOUT: --- add exposed name `interface TestInterface` (full_trace.carbon:[[@LINE+2]])
// CHECK:STDOUT: --- declared `TestInterface` in `interface TestInterface` (full_trace.carbon:[[@LINE+1]])
interface TestInterface {}

fn Main() -> i32 {
  return 0;
// CHECK:STDOUT: --- add exposed name `fn Main` (full_trace.carbon:[[@LINE+173]])
// CHECK:STDOUT: --- declared `Main` in `fn Main` (full_trace.carbon:[[@LINE+172]])
// CHECK:STDOUT: ** resolving decl `interface TestInterface` (full_trace.carbon:[[@LINE-6]])
// CHECK:STDOUT: --- marked `TestInterface` declared but not usable
// CHECK:STDOUT: --- marked `TestInterface` usable
// CHECK:STDOUT: --- declared `Self` in `Self` (full_trace.carbon:[[@LINE-9]])
// CHECK:STDOUT: ** finished resolving decl `interface TestInterface` (full_trace.carbon:[[@LINE-10]])
// CHECK:STDOUT: ** resolving decl `fn Main` (full_trace.carbon:[[@LINE+166]])
// CHECK:STDOUT: --- marked `Main` declared but not usable
// CHECK:STDOUT: ** resolving pattern `()` (full_trace.carbon:[[@LINE-11]])
// CHECK:STDOUT: ** finished resolving pattern `()` (full_trace.carbon:[[@LINE-12]])
// CHECK:STDOUT: ** resolving expr `i32` (full_trace.carbon:[[@LINE-13]])
// CHECK:STDOUT: ** finished resolving expr `i32` (full_trace.carbon:[[@LINE-14]])
// CHECK:STDOUT: --- marked `Main` usable
// CHECK:STDOUT: ** resolving stmt `{return 0;}` (full_trace.carbon:[[@LINE+159]])
// CHECK:STDOUT: ** resolving stmt `return 0;` (full_trace.carbon:[[@LINE-16]])
// CHECK:STDOUT: ** resolving expr `0` (full_trace.carbon:[[@LINE-17]])
// CHECK:STDOUT: ** finished resolving expr `0` (full_trace.carbon:[[@LINE-18]])
// CHECK:STDOUT: ** finished resolving stmt `return 0;` (full_trace.carbon:[[@LINE-19]])
// CHECK:STDOUT: ** finished resolving stmt `{return 0;}` (full_trace.carbon:[[@LINE+154]])
// CHECK:STDOUT: ** finished resolving decl `fn Main` (full_trace.carbon:[[@LINE+153]])
// CHECK:STDOUT: ** resolving expr `Main()` (<Main()>:0)
// CHECK:STDOUT: ** resolving expr `Main` (<Main()>:0)
// CHECK:STDOUT: ** resolving expr `()` (<Main()>:0)
// CHECK:STDOUT: ** finished resolving expr `()` (<Main()>:0)
// CHECK:STDOUT: ** finished resolving expr `Main()` (<Main()>:0)
// CHECK:STDOUT: ********** resolving control flow **********
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
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory:{{ }}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp i32 .0. (full_trace.carbon:[[@LINE-44]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory:{{ }}
// CHECK:STDOUT: }
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
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface TestInterface .0. (full_trace.carbon:[[@LINE-86]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl fn Main .0. (full_trace.carbon:[[@LINE+82]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: ********** calling main function **********
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp Main() .0. (<Main()>:0) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: Main .0. ## Main() .1.
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp Main .0. (<Main()>:0) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: Main() .1. {{[[][[]}}fun<Main>]]
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp Main() .1. (<Main()>:0) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: () .0. ## Main() .2. {{[[][[]}}fun<Main>]]
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp () .0. (<Main()>:0) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: Main() .2. {{[[][[]}}fun<Main>, ()]]
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp Main() .2. (<Main()>:0) --->
// CHECK:STDOUT: calling function: fun<Main>
// CHECK:STDOUT: match pattern ()
// CHECK:STDOUT: from value expression with value ()
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: {return 0;} .0. ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step stmt {return 0;} .0. (full_trace.carbon:[[@LINE+44]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: return 0; .0. ## {return 0;} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step stmt return 0; .0. (full_trace.carbon:[[@LINE-135]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: 0 .0. ## return 0; .1. ## {return 0;} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp 0 .0. (full_trace.carbon:[[@LINE-140]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: return 0; .1. {{[[][[]}}0]] ## {return 0;} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step stmt return 0; .1. (full_trace.carbon:[[@LINE-145]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.0. {} ## clean up.0. {} ## Main() .3. {{[[][[]}}fun<Main>, (), 0]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.0. {} ## Main() .3. {{[[][[]}}fun<Main>, (), 0]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: Main() .3. {{[[][[]}}fun<Main>, (), 0]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp Main() .3. (<Main()>:0) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.0. {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: interpreter result: 0
// CHECK:STDOUT: ********** printing timing **********
// CHECK:STDOUT: Time elapsed in ExecProgram: {{[0-9]+}}ms
// CHECK:STDOUT: Time elapsed in AnalyzeProgram: {{[0-9]+}}ms
// CHECK:STDOUT: Time elapsed in AddPrelude: {{[0-9]+}}ms
// CHECK:STDOUT: Time elapsed in Parse: {{[0-9]+}}ms
// CHECK:STDOUT: result: 0
}
