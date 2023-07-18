// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ********** source program **********
// CHECK:STDOUT: interface TestInterface {
// CHECK:STDOUT: }
// CHECK:STDOUT: namespace N;fn Foo (n: i32)-> i32 {
// CHECK:STDOUT: {
// CHECK:STDOUT: return (n + 1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: }
// CHECK:STDOUT: fn Main ()-> i32 {
// CHECK:STDOUT: {
// CHECK:STDOUT: return N.Foo(0);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: }
// CHECK:STDOUT: ********** resolving names **********


package ExplorerTest api;

// CHECK:STDOUT: --- declared `TestInterface` as `interface TestInterface` in `package` (full_trace.carbon:[[@LINE+1]])
interface TestInterface {}

// CHECK:STDOUT: --- declared `N` as `namespace N` in `package` (full_trace.carbon:[[@LINE+1]])
namespace N;

// CHECK:STDOUT: --- resolved `N` as `namespace N` in `package` (full_trace.carbon:[[@LINE+1]])
fn N.Foo(n: i32) -> i32 {
  return n + 1;
// CHECK:STDOUT: --- declared `Foo` as `fn N.Foo` in `namespace N` (full_trace.carbon:[[@LINE+1]])
}

fn Main() -> i32 {
  return N.Foo(0);
// CHECK:STDOUT: --- declared `Main` as `fn Main` in `package` (full_trace.carbon:[[@LINE+360]])
// CHECK:STDOUT: ** resolving decl `interface TestInterface` (full_trace.carbon:[[@LINE-14]])
// CHECK:STDOUT: --- marked `TestInterface` declared but not usable in `package`
// CHECK:STDOUT: --- marked `TestInterface` usable in `package`
// CHECK:STDOUT: --- declared `Self` as `Self` in `interface TestInterface` (full_trace.carbon:[[@LINE-17]])
// CHECK:STDOUT: ** finished resolving decl `interface TestInterface` (full_trace.carbon:[[@LINE-18]])
// CHECK:STDOUT: ** resolving decl `namespace N` (full_trace.carbon:[[@LINE-16]])
// CHECK:STDOUT: --- marked `N` usable in `package`
// CHECK:STDOUT: ** finished resolving decl `namespace N` (full_trace.carbon:[[@LINE-18]])
// CHECK:STDOUT: ** resolving decl `fn N.Foo` (full_trace.carbon:[[@LINE-13]])
// CHECK:STDOUT: --- resolved `N` as `namespace N` in `package` (full_trace.carbon:[[@LINE-17]])
// CHECK:STDOUT: --- marked `Foo` declared but not usable in `namespace N`
// CHECK:STDOUT: --- declared `n` as `n` in `fn N.Foo` (full_trace.carbon:[[@LINE-19]])
// CHECK:STDOUT: --- marked `Foo` usable in `namespace N`
// CHECK:STDOUT: ** resolving stmt `{return (n + 1);}` (full_trace.carbon:[[@LINE-18]])
// CHECK:STDOUT: ** resolving stmt `return (n + 1);` (full_trace.carbon:[[@LINE-21]])
// CHECK:STDOUT: --- resolved `n` as `n` in `fn N.Foo` (full_trace.carbon:[[@LINE-22]])
// CHECK:STDOUT: ** finished resolving stmt `return (n + 1);` (full_trace.carbon:[[@LINE-23]])
// CHECK:STDOUT: ** finished resolving stmt `{return (n + 1);}` (full_trace.carbon:[[@LINE-22]])
// CHECK:STDOUT: ** finished resolving decl `fn N.Foo` (full_trace.carbon:[[@LINE-23]])
// CHECK:STDOUT: ** resolving decl `fn Main` (full_trace.carbon:[[@LINE+340]])
// CHECK:STDOUT: --- marked `Main` declared but not usable in `package`
// CHECK:STDOUT: --- marked `Main` usable in `package`
// CHECK:STDOUT: ** resolving stmt `{return N.Foo(0);}` (full_trace.carbon:[[@LINE+337]])
// CHECK:STDOUT: ** resolving stmt `return N.Foo(0);` (full_trace.carbon:[[@LINE-25]])
// CHECK:STDOUT: --- resolved `N` as `namespace N` in `package` (full_trace.carbon:[[@LINE-26]])
// CHECK:STDOUT: --- resolved `Foo` as `fn N.Foo` in `namespace N` (full_trace.carbon:[[@LINE-27]])
// CHECK:STDOUT: ** finished resolving stmt `return N.Foo(0);` (full_trace.carbon:[[@LINE-28]])
// CHECK:STDOUT: ** finished resolving stmt `{return N.Foo(0);}` (full_trace.carbon:[[@LINE+332]])
// CHECK:STDOUT: ** finished resolving decl `fn Main` (full_trace.carbon:[[@LINE+331]])
// CHECK:STDOUT: --- resolved `Main` as `fn Main` in `package` (<Main()>:0)
// CHECK:STDOUT: ********** resolving control flow **********
// CHECK:STDOUT: ********** type checking **********
// CHECK:STDOUT: ** declaring interface TestInterface
// CHECK:STDOUT: ** finished declaring interface TestInterface
// CHECK:STDOUT: checking InterfaceDeclaration
// CHECK:STDOUT: ** checking interface TestInterface
// CHECK:STDOUT: impl declarations:{{ }}
// CHECK:STDOUT: impl declarations: bool as interface EqWith(U = bool), i32 as interface EqWith(U = i32), String as interface EqWith(U = String), i32 as interface CompareWith(U = i32), String as interface CompareWith(U = String), i32 as interface LessWith(U = i32), String as interface LessWith(U = String), i32 as interface LessEqWith(U = i32), String as interface LessEqWith(U = String), i32 as interface GreaterWith(U = i32), String as interface GreaterWith(U = String), i32 as interface GreaterEqWith(U = i32), String as interface GreaterEqWith(U = String), i32 as interface Negate, i32 as interface AddWith(U = i32), i32 as interface SubWith(U = i32), i32 as interface MulWith(U = i32), i32 as interface DivWith(U = i32), i32 as interface ModWith(U = i32), i32 as interface BitComplement, i32 as interface BitAndWith(U = i32), i32 as interface BitOrWith(U = i32), i32 as interface BitXorWith(U = i32), i32 as interface LeftShiftWith(U = i32), i32 as interface RightShiftWith(U = i32), i32 as interface Inc, i32 as interface Dec, U as interface __EqualConverter [0], (T1,) as interface As(T = (U1,)) [0, 0, 0; 1, 1, 0, 0, 0], (T1, T2) as interface As(T = (U1, U2)) [0, 0, 0; 0, 0, 1; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1], (T1, U1) as interface EqWith(U = (T2, U2)) [0, 0, 0; 0, 0, 1; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1], (T1, T2, T3) as interface As(T = (U1, U2, U3)) [0, 0, 0; 0, 0, 1; 0, 0, 2; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1; 1, 1, 0, 0, 2], T as interface ImplicitAs(T = U) [0; 1, 1, 0], T as interface As(T = U) [0; 1, 1, 0], T as interface ImplicitAs(T = U) [0; 1, 1, 0], T as interface As(T = U) [0; 1, 1, 0], T as interface AssignWith(U = U) [0; 1, 1, 0], T as interface AddAssignWith(U = U) [0; 1, 1, 0], T as interface SubAssignWith(U = U) [0; 1, 1, 0], T as interface MulAssignWith(U = U) [0; 1, 1, 0], T as interface DivAssignWith(U = U) [0; 1, 1, 0], T as interface ModAssignWith(U = U) [0; 1, 1, 0], T as interface BitAndAssignWith(U = U) [0; 1, 1, 0], T as interface BitOrAssignWith(U = U) [0; 1, 1, 0], T as interface BitXorAssignWith(U = U) [0; 1, 1, 0], T as interface LeftShiftAssignWith(U = U) [0; 1, 1, 0], T as interface RightShiftAssignWith(U = U) [0; 1, 1, 0]
// CHECK:STDOUT: ** finished checking interface TestInterface
// CHECK:STDOUT: checking NamespaceDeclaration
// CHECK:STDOUT: ** declaring function Foo
// CHECK:STDOUT: checking TuplePattern (n: i32)
// CHECK:STDOUT: checking BindingPattern n: i32
// CHECK:STDOUT: checking ExpressionPattern i32
// CHECK:STDOUT: checking IntTypeLiteral i32
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory:{{ }}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp i32 .0. (full_trace.carbon:[[@LINE-57]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory:{{ }}
// CHECK:STDOUT: }
// CHECK:STDOUT: finished checking tuple pattern field n: i32
// CHECK:STDOUT: checking IntTypeLiteral i32
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory:{{ }}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp i32 .0. (full_trace.carbon:[[@LINE-68]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory:{{ }}
// CHECK:STDOUT: }
// CHECK:STDOUT: ** finished declaring function Foo of type fn (i32,) -> i32
// CHECK:STDOUT: checking FunctionDeclaration
// CHECK:STDOUT: ** checking function Foo
// CHECK:STDOUT: impl declarations:{{ }}
// CHECK:STDOUT: impl declarations: bool as interface EqWith(U = bool), i32 as interface EqWith(U = i32), String as interface EqWith(U = String), i32 as interface CompareWith(U = i32), String as interface CompareWith(U = String), i32 as interface LessWith(U = i32), String as interface LessWith(U = String), i32 as interface LessEqWith(U = i32), String as interface LessEqWith(U = String), i32 as interface GreaterWith(U = i32), String as interface GreaterWith(U = String), i32 as interface GreaterEqWith(U = i32), String as interface GreaterEqWith(U = String), i32 as interface Negate, i32 as interface AddWith(U = i32), i32 as interface SubWith(U = i32), i32 as interface MulWith(U = i32), i32 as interface DivWith(U = i32), i32 as interface ModWith(U = i32), i32 as interface BitComplement, i32 as interface BitAndWith(U = i32), i32 as interface BitOrWith(U = i32), i32 as interface BitXorWith(U = i32), i32 as interface LeftShiftWith(U = i32), i32 as interface RightShiftWith(U = i32), i32 as interface Inc, i32 as interface Dec, U as interface __EqualConverter [0], (T1,) as interface As(T = (U1,)) [0, 0, 0; 1, 1, 0, 0, 0], (T1, T2) as interface As(T = (U1, U2)) [0, 0, 0; 0, 0, 1; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1], (T1, U1) as interface EqWith(U = (T2, U2)) [0, 0, 0; 0, 0, 1; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1], (T1, T2, T3) as interface As(T = (U1, U2, U3)) [0, 0, 0; 0, 0, 1; 0, 0, 2; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1; 1, 1, 0, 0, 2], T as interface ImplicitAs(T = U) [0; 1, 1, 0], T as interface As(T = U) [0; 1, 1, 0], T as interface ImplicitAs(T = U) [0; 1, 1, 0], T as interface As(T = U) [0; 1, 1, 0], T as interface AssignWith(U = U) [0; 1, 1, 0], T as interface AddAssignWith(U = U) [0; 1, 1, 0], T as interface SubAssignWith(U = U) [0; 1, 1, 0], T as interface MulAssignWith(U = U) [0; 1, 1, 0], T as interface DivAssignWith(U = U) [0; 1, 1, 0], T as interface ModAssignWith(U = U) [0; 1, 1, 0], T as interface BitAndAssignWith(U = U) [0; 1, 1, 0], T as interface BitOrAssignWith(U = U) [0; 1, 1, 0], T as interface BitXorAssignWith(U = U) [0; 1, 1, 0], T as interface LeftShiftAssignWith(U = U) [0; 1, 1, 0], T as interface RightShiftAssignWith(U = U) [0; 1, 1, 0]
// CHECK:STDOUT: checking Block {
// CHECK:STDOUT: return (n + 1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: checking ReturnExpression return (n + 1);
// CHECK:STDOUT: checking OperatorExpression (n + 1)
// CHECK:STDOUT: checking IdentifierExpression n
// CHECK:STDOUT: checking IntLiteral 1
// CHECK:STDOUT: ** finished checking function Foo
// CHECK:STDOUT: ** declaring function Main
// CHECK:STDOUT: checking TuplePattern ()
// CHECK:STDOUT: checking IntTypeLiteral i32
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory:{{ }}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: i32 .0. ## i32 .1.
// CHECK:STDOUT: memory:{{ }}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp i32 .0. (full_trace.carbon:[[@LINE-89]]) --->
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
// CHECK:STDOUT: return N.Foo(0);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: checking ReturnExpression return N.Foo(0);
// CHECK:STDOUT: checking CallExpression N.Foo(0)
// CHECK:STDOUT: checking SimpleMemberAccessExpression N.Foo
// CHECK:STDOUT: checking IdentifierExpression Foo
// CHECK:STDOUT: checking TupleLiteral (0)
// CHECK:STDOUT: checking IntLiteral 0
// CHECK:STDOUT: checking call to function of type fn (i32,) -> i32
// CHECK:STDOUT: with arguments of type: (i32,)
// CHECK:STDOUT: performing argument deduction for bindings:{{ }}
// CHECK:STDOUT: deducing i32 from i32
// CHECK:STDOUT: deduction succeeded with results: {}
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
// CHECK:STDOUT: namespace N;fn Foo (n: i32)-> i32 {
// CHECK:STDOUT: {
// CHECK:STDOUT: return (n + 1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: }
// CHECK:STDOUT: fn Main ()-> i32 {
// CHECK:STDOUT: {
// CHECK:STDOUT: return N.Foo(0);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: }
// CHECK:STDOUT: ********** starting execution **********
// CHECK:STDOUT: ********** initializing globals **********
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface TestInterface .0. (full_trace.carbon:[[@LINE-155]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl namespace N .0. (full_trace.carbon:[[@LINE-161]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl fn N.Foo .0. (full_trace.carbon:[[@LINE-164]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl fn Main .0. (full_trace.carbon:[[@LINE+191]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: ********** calling main function **********
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: Main() .0. ## Main() .1.
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp Main() .0. (<Main()>:0) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: Main .0. ## Main() .1. ## Main() .1.
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: Main .0. ## Main .1. ## Main() .1. ## Main() .1.
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp Main .0. (<Main()>:0) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: Main .1. {{[[][[]}}fun<Main>]] ## Main() .1. ## Main() .1.
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: Main() .1. {{[[][[]}}fun<Main>]] ## Main() .1.
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp Main() .1. (<Main()>:0) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: () .0. ## Main() .2. {{[[][[]}}fun<Main>]] ## Main() .1.
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp () .0. (<Main()>:0) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: Main() .2. {{[[][[]}}fun<Main>, ()]] ## Main() .1.
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp Main() .2. (<Main()>:0) --->
// CHECK:STDOUT: calling function: fun<Main>
// CHECK:STDOUT: match pattern ()
// CHECK:STDOUT: from value expression with value ()
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: {return N.Foo(0);} .0. ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step stmt {return N.Foo(0);} .0. (full_trace.carbon:[[@LINE+153]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: return N.Foo(0); .0. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step stmt return N.Foo(0); .0. (full_trace.carbon:[[@LINE-213]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: N.Foo(0) .0. ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp N.Foo(0) .0. (full_trace.carbon:[[@LINE-218]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: N.Foo .0. ## N.Foo(0) .1. ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp N.Foo .0. (full_trace.carbon:[[@LINE-223]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: Foo .0. ## N.Foo(0) .1. ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp Foo .0. (full_trace.carbon:[[@LINE-228]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: N.Foo(0) .1. {{[[][[]}}fun<N.Foo>]] ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp N.Foo(0) .1. (full_trace.carbon:[[@LINE-233]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: (0) .0. ## N.Foo(0) .2. {{[[][[]}}fun<N.Foo>]] ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp (0) .0. (full_trace.carbon:[[@LINE-238]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: 0 .0. ## (0) .1. ## N.Foo(0) .2. {{[[][[]}}fun<N.Foo>]] ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp 0 .0. (full_trace.carbon:[[@LINE-243]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: (0) .1. {{[[][[]}}0]] ## N.Foo(0) .2. {{[[][[]}}fun<N.Foo>]] ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp (0) .1. (full_trace.carbon:[[@LINE-248]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: N.Foo(0) .2. {{[[][[]}}fun<N.Foo>, (0,)]] ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp N.Foo(0) .2. (full_trace.carbon:[[@LINE-253]]) --->
// CHECK:STDOUT: calling function: fun<N.Foo>
// CHECK:STDOUT: match pattern (Placeholder<n>,)
// CHECK:STDOUT: from value expression with value (0,)
// CHECK:STDOUT: match pattern Placeholder<n>
// CHECK:STDOUT: from value expression with value 0
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: {return (n + 1);} .0. ## .0. {n: i32: lval<Allocation(1)>} ## N.Foo(0) .3. {{[[][[]}}fun<N.Foo>, (0,)]] {} ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: 0
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step stmt {return (n + 1);} .0. (full_trace.carbon:[[@LINE-266]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: return (n + 1); .0. ## {return (n + 1);} .1. {} ## .0. {n: i32: lval<Allocation(1)>} ## N.Foo(0) .3. {{[[][[]}}fun<N.Foo>, (0,)]] {} ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: 0
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step stmt return (n + 1); .0. (full_trace.carbon:[[@LINE-273]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: (n + 1) .0. ## return (n + 1); .1. ## {return (n + 1);} .1. {} ## .0. {n: i32: lval<Allocation(1)>} ## N.Foo(0) .3. {{[[][[]}}fun<N.Foo>, (0,)]] {} ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: 0
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp (n + 1) .0. (full_trace.carbon:[[@LINE-278]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: n .0. ## (n + 1) .1. ## return (n + 1); .1. ## {return (n + 1);} .1. {} ## .0. {n: i32: lval<Allocation(1)>} ## N.Foo(0) .3. {{[[][[]}}fun<N.Foo>, (0,)]] {} ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: 0
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp n .0. (full_trace.carbon:[[@LINE-283]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: (n + 1) .1. {{[[][[]}}0]] ## return (n + 1); .1. ## {return (n + 1);} .1. {} ## .0. {n: i32: lval<Allocation(1)>} ## N.Foo(0) .3. {{[[][[]}}fun<N.Foo>, (0,)]] {} ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: 0
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp (n + 1) .1. (full_trace.carbon:[[@LINE-288]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: 1 .0. ## (n + 1) .2. {{[[][[]}}0]] ## return (n + 1); .1. ## {return (n + 1);} .1. {} ## .0. {n: i32: lval<Allocation(1)>} ## N.Foo(0) .3. {{[[][[]}}fun<N.Foo>, (0,)]] {} ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: 0
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp 1 .0. (full_trace.carbon:[[@LINE-293]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: (n + 1) .2. {{[[][[]}}0, 1]] ## return (n + 1); .1. ## {return (n + 1);} .1. {} ## .0. {n: i32: lval<Allocation(1)>} ## N.Foo(0) .3. {{[[][[]}}fun<N.Foo>, (0,)]] {} ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: 0
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp (n + 1) .2. (full_trace.carbon:[[@LINE-298]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: return (n + 1); .1. {{[[][[]}}1]] ## {return (n + 1);} .1. {} ## .0. {n: i32: lval<Allocation(1)>} ## N.Foo(0) .3. {{[[][[]}}fun<N.Foo>, (0,)]] {} ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: 0
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step stmt return (n + 1); .1. (full_trace.carbon:[[@LINE-303]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.0. {} ## clean up.0. {n: i32: lval<Allocation(1)>} ## N.Foo(0) .3. {{[[][[]}}fun<N.Foo>, (0,), 1]] {} ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: 0
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.0. {n: i32: lval<Allocation(1)>} ## N.Foo(0) .3. {{[[][[]}}fun<N.Foo>, (0,), 1]] {} ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: 0
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: destroy.0. ## clean up.1. {n: i32: lval<Allocation(1)>} ## N.Foo(0) .3. {{[[][[]}}fun<N.Foo>, (0,), 1]] {} ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: 0
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.1. {n: i32: lval<Allocation(1)>} ## N.Foo(0) .3. {{[[][[]}}fun<N.Foo>, (0,), 1]] {} ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: 0
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.2. {n: i32: lval<Allocation(1)>} ## N.Foo(0) .3. {{[[][[]}}fun<N.Foo>, (0,), 1]] {} ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: !!0
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: N.Foo(0) .3. {{[[][[]}}fun<N.Foo>, (0,), 1]] {} ## return N.Foo(0); .1. ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: !!0
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp N.Foo(0) .3. (full_trace.carbon:[[@LINE-323]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.0. {} ## return N.Foo(0); .1. {{[[][[]}}1]] ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: !!0
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: return N.Foo(0); .1. {{[[][[]}}1]] ## {return N.Foo(0);} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: !!0
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step stmt return N.Foo(0); .1. (full_trace.carbon:[[@LINE-332]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.0. {} ## clean up.0. {} ## Main() .3. {{[[][[]}}fun<Main>, (), 1]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: !!0
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.0. {} ## Main() .3. {{[[][[]}}fun<Main>, (), 1]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: !!0
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: Main() .3. {{[[][[]}}fun<Main>, (), 1]] {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: !!0
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp Main() .3. (<Main()>:0) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.0. {}
// CHECK:STDOUT: memory: 0: Heap{}, 1: !!0
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:{{ }}
// CHECK:STDOUT: memory: 0: Heap{}, 1: !!0
// CHECK:STDOUT: }
// CHECK:STDOUT: interpreter result: 1
// CHECK:STDOUT: ********** printing timing **********
// CHECK:STDOUT: Time elapsed in ExecProgram: {{[0-9]+}}ms
// CHECK:STDOUT: Time elapsed in AnalyzeProgram: {{[0-9]+}}ms
// CHECK:STDOUT: Time elapsed in AddPrelude: {{[0-9]+}}ms
// CHECK:STDOUT: Time elapsed in Parse: {{[0-9]+}}ms
// CHECK:STDOUT: result: 1
}
