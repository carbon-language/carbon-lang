// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExplorerTest api;

interface TestInterface {}

namespace N;

fn N.Foo(n: i32) -> i32 {
  return n + 1;
}

fn Main() -> i32 {
  var x: i32 = N.Foo(0);
  return x;
}

// Place checks after code so that line numbers are stable, reducing merge
// conflicts.
// ARGS: --trace_file=- --trace_phase=all %s
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
// CHECK:STDOUT: var x: i32 = N.Foo(0);
// CHECK:STDOUT: return x;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: }
// CHECK:STDOUT: ********** resolving names **********
// CHECK:STDOUT: --- declared `TestInterface` as `interface TestInterface` in `package` (phase_all.carbon:7)
// CHECK:STDOUT: --- declared `N` as `namespace N` in `package` (phase_all.carbon:9)
// CHECK:STDOUT: --- resolved `N` as `namespace N` in `package` (phase_all.carbon:11)
// CHECK:STDOUT: --- declared `Foo` as `fn N.Foo` in `namespace N` (phase_all.carbon:13)
// CHECK:STDOUT: --- declared `Main` as `fn Main` in `package` (phase_all.carbon:18)
// CHECK:STDOUT: ** resolving decl `interface TestInterface` (phase_all.carbon:7)
// CHECK:STDOUT: --- marked `TestInterface` declared but not usable in `package`
// CHECK:STDOUT: --- marked `TestInterface` usable in `package`
// CHECK:STDOUT: --- declared `Self` as `Self` in `interface TestInterface` (phase_all.carbon:7)
// CHECK:STDOUT: ** finished resolving decl `interface TestInterface` (phase_all.carbon:7)
// CHECK:STDOUT: ** resolving decl `namespace N` (phase_all.carbon:9)
// CHECK:STDOUT: --- marked `N` usable in `package`
// CHECK:STDOUT: ** finished resolving decl `namespace N` (phase_all.carbon:9)
// CHECK:STDOUT: ** resolving decl `fn N.Foo` (phase_all.carbon:13)
// CHECK:STDOUT: --- resolved `N` as `namespace N` in `package` (phase_all.carbon:11)
// CHECK:STDOUT: --- marked `Foo` declared but not usable in `namespace N`
// CHECK:STDOUT: --- declared `n` as `n` in `fn N.Foo` (phase_all.carbon:11)
// CHECK:STDOUT: --- marked `Foo` usable in `namespace N`
// CHECK:STDOUT: ** resolving stmt `{return (n + 1);}` (phase_all.carbon:13)
// CHECK:STDOUT: ** resolving stmt `return (n + 1);` (phase_all.carbon:12)
// CHECK:STDOUT: --- resolved `n` as `n` in `fn N.Foo` (phase_all.carbon:12)
// CHECK:STDOUT: ** finished resolving stmt `return (n + 1);` (phase_all.carbon:12)
// CHECK:STDOUT: ** finished resolving stmt `{return (n + 1);}` (phase_all.carbon:13)
// CHECK:STDOUT: ** finished resolving decl `fn N.Foo` (phase_all.carbon:13)
// CHECK:STDOUT: ** resolving decl `fn Main` (phase_all.carbon:18)
// CHECK:STDOUT: --- marked `Main` declared but not usable in `package`
// CHECK:STDOUT: --- marked `Main` usable in `package`
// CHECK:STDOUT: ** resolving stmt `{var x: i32 = N.Foo(0);return x;}` (phase_all.carbon:18)
// CHECK:STDOUT: ** resolving stmt `var x: i32 = N.Foo(0);` (phase_all.carbon:16)
// CHECK:STDOUT: --- resolved `N` as `namespace N` in `package` (phase_all.carbon:16)
// CHECK:STDOUT: --- resolved `Foo` as `fn N.Foo` in `namespace N` (phase_all.carbon:16)
// CHECK:STDOUT: --- declared `x` as `x` in `{var x: i32 = N.Foo(0);return x;}` (phase_all.carbon:16)
// CHECK:STDOUT: ** finished resolving stmt `var x: i32 = N.Foo(0);` (phase_all.carbon:16)
// CHECK:STDOUT: ** resolving stmt `return x;` (phase_all.carbon:17)
// CHECK:STDOUT: --- resolved `x` as `x` in `{var x: i32 = N.Foo(0);return x;}` (phase_all.carbon:17)
// CHECK:STDOUT: ** finished resolving stmt `return x;` (phase_all.carbon:17)
// CHECK:STDOUT: ** finished resolving stmt `{var x: i32 = N.Foo(0);return x;}` (phase_all.carbon:18)
// CHECK:STDOUT: ** finished resolving decl `fn Main` (phase_all.carbon:18)
// CHECK:STDOUT: --- resolved `Main` as `fn Main` in `package` (<Main()>:0)
// CHECK:STDOUT: ********** resolving control flow **********
// CHECK:STDOUT: --- flow-resolved return statement `return (n + 1);` in `fn N.Foo` (phase_all.carbon:12)
// CHECK:STDOUT: --- flow-resolved return statement `return x;` in `fn Main` (phase_all.carbon:17)
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
// CHECK:STDOUT: (+) stack-push: ValueExpressionAction pos: 0 `i32` (phase_all.carbon:11)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 0 `i32` (phase_all.carbon:11) --->
// CHECK:STDOUT: (+) stack-push: ExpressionAction pos: 0 `i32` (phase_all.carbon:11)
// CHECK:STDOUT: --- step ExpressionAction pos: 0 `i32` (phase_all.carbon:11) --->
// CHECK:STDOUT: (-) stack-pop:  ExpressionAction pos: 0 `i32` (phase_all.carbon:11)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 1 `i32` results: [`i32`]  (phase_all.carbon:11) --->
// CHECK:STDOUT: (-) stack-pop:  ValueExpressionAction pos: 1 `i32` results: [`i32`]  (phase_all.carbon:11)
// CHECK:STDOUT: finished checking tuple pattern field n: i32
// CHECK:STDOUT: checking IntTypeLiteral i32
// CHECK:STDOUT: (+) stack-push: ValueExpressionAction pos: 0 `i32` (phase_all.carbon:11)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 0 `i32` (phase_all.carbon:11) --->
// CHECK:STDOUT: (+) stack-push: ExpressionAction pos: 0 `i32` (phase_all.carbon:11)
// CHECK:STDOUT: --- step ExpressionAction pos: 0 `i32` (phase_all.carbon:11) --->
// CHECK:STDOUT: (-) stack-pop:  ExpressionAction pos: 0 `i32` (phase_all.carbon:11)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 1 `i32` results: [`i32`]  (phase_all.carbon:11) --->
// CHECK:STDOUT: (-) stack-pop:  ValueExpressionAction pos: 1 `i32` results: [`i32`]  (phase_all.carbon:11)
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
// CHECK:STDOUT: (+) stack-push: ValueExpressionAction pos: 0 `i32` (phase_all.carbon:15)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 0 `i32` (phase_all.carbon:15) --->
// CHECK:STDOUT: (+) stack-push: ExpressionAction pos: 0 `i32` (phase_all.carbon:15)
// CHECK:STDOUT: --- step ExpressionAction pos: 0 `i32` (phase_all.carbon:15) --->
// CHECK:STDOUT: (-) stack-pop:  ExpressionAction pos: 0 `i32` (phase_all.carbon:15)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 1 `i32` results: [`i32`]  (phase_all.carbon:15) --->
// CHECK:STDOUT: (-) stack-pop:  ValueExpressionAction pos: 1 `i32` results: [`i32`]  (phase_all.carbon:15)
// CHECK:STDOUT: ** finished declaring function Main of type fn () -> i32
// CHECK:STDOUT: checking FunctionDeclaration
// CHECK:STDOUT: ** checking function Main
// CHECK:STDOUT: impl declarations:{{ }}
// CHECK:STDOUT: impl declarations: bool as interface EqWith(U = bool), i32 as interface EqWith(U = i32), String as interface EqWith(U = String), i32 as interface CompareWith(U = i32), String as interface CompareWith(U = String), i32 as interface LessWith(U = i32), String as interface LessWith(U = String), i32 as interface LessEqWith(U = i32), String as interface LessEqWith(U = String), i32 as interface GreaterWith(U = i32), String as interface GreaterWith(U = String), i32 as interface GreaterEqWith(U = i32), String as interface GreaterEqWith(U = String), i32 as interface Negate, i32 as interface AddWith(U = i32), i32 as interface SubWith(U = i32), i32 as interface MulWith(U = i32), i32 as interface DivWith(U = i32), i32 as interface ModWith(U = i32), i32 as interface BitComplement, i32 as interface BitAndWith(U = i32), i32 as interface BitOrWith(U = i32), i32 as interface BitXorWith(U = i32), i32 as interface LeftShiftWith(U = i32), i32 as interface RightShiftWith(U = i32), i32 as interface Inc, i32 as interface Dec, U as interface __EqualConverter [0], (T1,) as interface As(T = (U1,)) [0, 0, 0; 1, 1, 0, 0, 0], (T1, T2) as interface As(T = (U1, U2)) [0, 0, 0; 0, 0, 1; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1], (T1, U1) as interface EqWith(U = (T2, U2)) [0, 0, 0; 0, 0, 1; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1], (T1, T2, T3) as interface As(T = (U1, U2, U3)) [0, 0, 0; 0, 0, 1; 0, 0, 2; 1, 1, 0, 0, 0; 1, 1, 0, 0, 1; 1, 1, 0, 0, 2], T as interface ImplicitAs(T = U) [0; 1, 1, 0], T as interface As(T = U) [0; 1, 1, 0], T as interface ImplicitAs(T = U) [0; 1, 1, 0], T as interface As(T = U) [0; 1, 1, 0], T as interface AssignWith(U = U) [0; 1, 1, 0], T as interface AddAssignWith(U = U) [0; 1, 1, 0], T as interface SubAssignWith(U = U) [0; 1, 1, 0], T as interface MulAssignWith(U = U) [0; 1, 1, 0], T as interface DivAssignWith(U = U) [0; 1, 1, 0], T as interface ModAssignWith(U = U) [0; 1, 1, 0], T as interface BitAndAssignWith(U = U) [0; 1, 1, 0], T as interface BitOrAssignWith(U = U) [0; 1, 1, 0], T as interface BitXorAssignWith(U = U) [0; 1, 1, 0], T as interface LeftShiftAssignWith(U = U) [0; 1, 1, 0], T as interface RightShiftAssignWith(U = U) [0; 1, 1, 0]
// CHECK:STDOUT: checking Block {
// CHECK:STDOUT: var x: i32 = N.Foo(0);
// CHECK:STDOUT: return x;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: checking VariableDefinition var x: i32 = N.Foo(0);
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
// CHECK:STDOUT: checking BindingPattern x: i32, expecting i32
// CHECK:STDOUT: checking ExpressionPattern i32, expecting i32
// CHECK:STDOUT: checking IntTypeLiteral i32
// CHECK:STDOUT: (+) stack-push: ValueExpressionAction pos: 0 `i32` (phase_all.carbon:16)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 0 `i32` (phase_all.carbon:16) --->
// CHECK:STDOUT: (+) stack-push: ExpressionAction pos: 0 `i32` (phase_all.carbon:16)
// CHECK:STDOUT: --- step ExpressionAction pos: 0 `i32` (phase_all.carbon:16) --->
// CHECK:STDOUT: (-) stack-pop:  ExpressionAction pos: 0 `i32` (phase_all.carbon:16)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 1 `i32` results: [`i32`]  (phase_all.carbon:16) --->
// CHECK:STDOUT: (-) stack-pop:  ValueExpressionAction pos: 1 `i32` results: [`i32`]  (phase_all.carbon:16)
// CHECK:STDOUT: checking ReturnExpression return x;
// CHECK:STDOUT: checking IdentifierExpression x
// CHECK:STDOUT: ** finished checking function Main
// CHECK:STDOUT: checking CallExpression Main()
// CHECK:STDOUT: checking IdentifierExpression Main
// CHECK:STDOUT: checking TupleLiteral ()
// CHECK:STDOUT: checking call to function of type fn () -> i32
// CHECK:STDOUT: with arguments of type: ()
// CHECK:STDOUT: performing argument deduction for bindings:{{ }}
// CHECK:STDOUT: deduction succeeded with results: {}
// CHECK:STDOUT: ********** resolving unformed variables **********
// CHECK:STDOUT: *** resolving-unformed in decl `interface TestInterface` (phase_all.carbon:7)
// CHECK:STDOUT: *** resolving-unformed in decl `namespace N` (phase_all.carbon:9)
// CHECK:STDOUT: *** resolving-unformed in decl `fn N.Foo` (phase_all.carbon:13)
// CHECK:STDOUT: *** resolving-unformed in stmt `{return (n + 1);}` (phase_all.carbon:13)
// CHECK:STDOUT: *** resolving-unformed in stmt `return (n + 1);` (phase_all.carbon:12)
// CHECK:STDOUT: --- check `n` (phase_all.carbon:12)
// CHECK:STDOUT: *** resolving-unformed in decl `fn Main` (phase_all.carbon:18)
// CHECK:STDOUT: *** resolving-unformed in stmt `{var x: i32 = N.Foo(0);return x;}` (phase_all.carbon:18)
// CHECK:STDOUT: *** resolving-unformed in stmt `var x: i32 = N.Foo(0);` (phase_all.carbon:16)
// CHECK:STDOUT: --- add init `x` (phase_all.carbon:16)
// CHECK:STDOUT: *** resolving-unformed in stmt `return x;` (phase_all.carbon:17)
// CHECK:STDOUT: --- check `x` (phase_all.carbon:17)
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
// CHECK:STDOUT: var x: i32 = N.Foo(0);
// CHECK:STDOUT: return x;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: }
// CHECK:STDOUT: ********** starting execution **********
// CHECK:STDOUT: ********** initializing globals **********
// CHECK:STDOUT: (+) stack-push: DeclarationAction pos: 0 `interface TestInterface` (phase_all.carbon:7)
// CHECK:STDOUT: --- step DeclarationAction pos: 0 `interface TestInterface` (phase_all.carbon:7) --->
// CHECK:STDOUT: (-) stack-pop:  DeclarationAction pos: 0 `interface TestInterface` (phase_all.carbon:7)
// CHECK:STDOUT: (+) stack-push: DeclarationAction pos: 0 `namespace N` (phase_all.carbon:9)
// CHECK:STDOUT: --- step DeclarationAction pos: 0 `namespace N` (phase_all.carbon:9) --->
// CHECK:STDOUT: (-) stack-pop:  DeclarationAction pos: 0 `namespace N` (phase_all.carbon:9)
// CHECK:STDOUT: (+) stack-push: DeclarationAction pos: 0 `fn N.Foo` (phase_all.carbon:13)
// CHECK:STDOUT: --- step DeclarationAction pos: 0 `fn N.Foo` (phase_all.carbon:13) --->
// CHECK:STDOUT: (-) stack-pop:  DeclarationAction pos: 0 `fn N.Foo` (phase_all.carbon:13)
// CHECK:STDOUT: (+) stack-push: DeclarationAction pos: 0 `fn Main` (phase_all.carbon:18)
// CHECK:STDOUT: --- step DeclarationAction pos: 0 `fn Main` (phase_all.carbon:18) --->
// CHECK:STDOUT: (-) stack-pop:  DeclarationAction pos: 0 `fn Main` (phase_all.carbon:18)
// CHECK:STDOUT: ********** calling main function **********
// CHECK:STDOUT: (+) stack-push: ValueExpressionAction pos: 0 `Main()` (<Main()>:0)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 0 `Main()` (<Main()>:0) --->
// CHECK:STDOUT: (+) stack-push: ExpressionAction pos: 0 `Main()` (<Main()>:0)
// CHECK:STDOUT: --- step ExpressionAction pos: 0 `Main()` (<Main()>:0) --->
// CHECK:STDOUT: (+) stack-push: ValueExpressionAction pos: 0 `Main` (<Main()>:0)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 0 `Main` (<Main()>:0) --->
// CHECK:STDOUT: (+) stack-push: ExpressionAction pos: 0 `Main` (<Main()>:0)
// CHECK:STDOUT: --- step ExpressionAction pos: 0 `Main` (<Main()>:0) --->
// CHECK:STDOUT: (-) stack-pop:  ExpressionAction pos: 0 `Main` (<Main()>:0)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 1 `Main` results: [`fun<Main>`]  (<Main()>:0) --->
// CHECK:STDOUT: (-) stack-pop:  ValueExpressionAction pos: 1 `Main` results: [`fun<Main>`]  (<Main()>:0)
// CHECK:STDOUT: --- step ExpressionAction pos: 1 `Main()` results: [`fun<Main>`]  scope: [] (<Main()>:0) --->
// CHECK:STDOUT: calling function: fun<Main>
// CHECK:STDOUT: match pattern ()
// CHECK:STDOUT: from value expression with value ()
// CHECK:STDOUT: (+) stack-push: ScopeAction pos: 0  scope: [] (None)
// CHECK:STDOUT: (+) stack-push: StatementAction pos: 0 `{var x: i32 = N.Foo(0);return x;}` (phase_all.carbon:18)
// CHECK:STDOUT: --- step StatementAction pos: 0 `{var x: i32 = N.Foo(0);return x;}` (phase_all.carbon:18) --->
// CHECK:STDOUT: (+) stack-push: StatementAction pos: 0 `var x: i32 = N.Foo(0);` (phase_all.carbon:16)
// CHECK:STDOUT: --- step StatementAction pos: 0 `var x: i32 = N.Foo(0);` (phase_all.carbon:16) --->
// CHECK:STDOUT: (+) memory-alloc: #1 `Uninit<i32>` uninitialized
// CHECK:STDOUT: (+) stack-push: ExpressionAction pos: 0 `N.Foo(0)` (phase_all.carbon:16)
// CHECK:STDOUT: --- step ExpressionAction pos: 0 `N.Foo(0)` (phase_all.carbon:16) --->
// CHECK:STDOUT: (+) stack-push: ValueExpressionAction pos: 0 `N.Foo` (phase_all.carbon:16)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 0 `N.Foo` (phase_all.carbon:16) --->
// CHECK:STDOUT: (+) stack-push: ExpressionAction pos: 0 `N.Foo` (phase_all.carbon:16)
// CHECK:STDOUT: --- step ExpressionAction pos: 0 `N.Foo` (phase_all.carbon:16) --->
// CHECK:STDOUT: (-) stack-pop:  ExpressionAction pos: 0 `N.Foo` (phase_all.carbon:16)
// CHECK:STDOUT: (+) stack-push: ExpressionAction pos: 0 `Foo` (phase_all.carbon:16)
// CHECK:STDOUT: --- step ExpressionAction pos: 0 `Foo` (phase_all.carbon:16) --->
// CHECK:STDOUT: (-) stack-pop:  ExpressionAction pos: 0 `Foo` (phase_all.carbon:16)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 1 `N.Foo` results: [`fun<N.Foo>`]  (phase_all.carbon:16) --->
// CHECK:STDOUT: (-) stack-pop:  ValueExpressionAction pos: 1 `N.Foo` results: [`fun<N.Foo>`]  (phase_all.carbon:16)
// CHECK:STDOUT: --- step ExpressionAction pos: 1 `N.Foo(0)` results: [`fun<N.Foo>`]  scope: [] (phase_all.carbon:16) --->
// CHECK:STDOUT: (+) stack-push: ExpressionAction pos: 0 `0` (phase_all.carbon:16)
// CHECK:STDOUT: --- step ExpressionAction pos: 0 `0` (phase_all.carbon:16) --->
// CHECK:STDOUT: (-) stack-pop:  ExpressionAction pos: 0 `0` (phase_all.carbon:16)
// CHECK:STDOUT: --- step ExpressionAction pos: 2 `N.Foo(0)` results: [`fun<N.Foo>`, `0`]  scope: [] (phase_all.carbon:16) --->
// CHECK:STDOUT: calling function: fun<N.Foo>
// CHECK:STDOUT: match pattern (Placeholder<n>,)
// CHECK:STDOUT: from value expression with value (0,)
// CHECK:STDOUT: match pattern Placeholder<n>
// CHECK:STDOUT: from value expression with value 0
// CHECK:STDOUT: (+) stack-push: ScopeAction pos: 0  scope: [`n: i32`: `0`] (None)
// CHECK:STDOUT: (+) stack-push: StatementAction pos: 0 `{return (n + 1);}` (phase_all.carbon:13)
// CHECK:STDOUT: --- step StatementAction pos: 0 `{return (n + 1);}` (phase_all.carbon:13) --->
// CHECK:STDOUT: (+) stack-push: StatementAction pos: 0 `return (n + 1);` (phase_all.carbon:12)
// CHECK:STDOUT: --- step StatementAction pos: 0 `return (n + 1);` (phase_all.carbon:12) --->
// CHECK:STDOUT: (+) stack-push: ValueExpressionAction pos: 0 `(n + 1)` (phase_all.carbon:12)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 0 `(n + 1)` (phase_all.carbon:12) --->
// CHECK:STDOUT: (+) stack-push: ExpressionAction pos: 0 `(n + 1)` (phase_all.carbon:12)
// CHECK:STDOUT: --- step ExpressionAction pos: 0 `(n + 1)` (phase_all.carbon:12) --->
// CHECK:STDOUT: (+) stack-push: ValueExpressionAction pos: 0 `n` (phase_all.carbon:12)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 0 `n` (phase_all.carbon:12) --->
// CHECK:STDOUT: (+) stack-push: ExpressionAction pos: 0 `n` (phase_all.carbon:12)
// CHECK:STDOUT: --- step ExpressionAction pos: 0 `n` (phase_all.carbon:12) --->
// CHECK:STDOUT: (-) stack-pop:  ExpressionAction pos: 0 `n` (phase_all.carbon:12)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 1 `n` results: [`0`]  (phase_all.carbon:12) --->
// CHECK:STDOUT: (-) stack-pop:  ValueExpressionAction pos: 1 `n` results: [`0`]  (phase_all.carbon:12)
// CHECK:STDOUT: --- step ExpressionAction pos: 1 `(n + 1)` results: [`0`]  (phase_all.carbon:12) --->
// CHECK:STDOUT: (+) stack-push: ValueExpressionAction pos: 0 `1` (phase_all.carbon:12)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 0 `1` (phase_all.carbon:12) --->
// CHECK:STDOUT: (+) stack-push: ExpressionAction pos: 0 `1` (phase_all.carbon:12)
// CHECK:STDOUT: --- step ExpressionAction pos: 0 `1` (phase_all.carbon:12) --->
// CHECK:STDOUT: (-) stack-pop:  ExpressionAction pos: 0 `1` (phase_all.carbon:12)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 1 `1` results: [`1`]  (phase_all.carbon:12) --->
// CHECK:STDOUT: (-) stack-pop:  ValueExpressionAction pos: 1 `1` results: [`1`]  (phase_all.carbon:12)
// CHECK:STDOUT: --- step ExpressionAction pos: 2 `(n + 1)` results: [`0`, `1`]  (phase_all.carbon:12) --->
// CHECK:STDOUT: (-) stack-pop:  ExpressionAction pos: 2 `(n + 1)` results: [`0`, `1`]  (phase_all.carbon:12)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 1 `(n + 1)` results: [`1`]  (phase_all.carbon:12) --->
// CHECK:STDOUT: (-) stack-pop:  ValueExpressionAction pos: 1 `(n + 1)` results: [`1`]  (phase_all.carbon:12)
// CHECK:STDOUT: --- step StatementAction pos: 1 `return (n + 1);` results: [`1`]  (phase_all.carbon:12) --->
// CHECK:STDOUT: +++ memory-write: #1 `1`
// CHECK:STDOUT: (-) stack-pop:  StatementAction pos: 1 `return (n + 1);` results: [`1`]  (phase_all.carbon:12)
// CHECK:STDOUT: (-) stack-pop:  StatementAction pos: 1 `{return (n + 1);}` scope: [] (phase_all.carbon:13)
// CHECK:STDOUT: (-) stack-pop:  ScopeAction pos: 0  scope: [`n: i32`: `0`] (None)
// CHECK:STDOUT: (+) stack-push: CleanUpAction pos: 0  scope: [`n: i32`: `0`] (stack cleanup:1)
// CHECK:STDOUT: (+) stack-push: CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: --- step CleanUpAction pos: 0  scope: [] (stack cleanup:1) --->
// CHECK:STDOUT: (-) stack-pop:  CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: --- step CleanUpAction pos: 0  scope: [`n: i32`: `0`] (stack cleanup:1) --->
// CHECK:STDOUT: (-) stack-pop:  CleanUpAction pos: 0  scope: [`n: i32`: `0`] (stack cleanup:1)
// CHECK:STDOUT: --- step ExpressionAction pos: 3 `N.Foo(0)` results: [`fun<N.Foo>`, `0`, `1`]  scope: [] (phase_all.carbon:16) --->
// CHECK:STDOUT: (-) stack-pop:  ExpressionAction pos: 3 `N.Foo(0)` results: [`fun<N.Foo>`, `0`, `1`]  scope: [] (phase_all.carbon:16)
// CHECK:STDOUT: (+) stack-push: CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: --- step CleanUpAction pos: 0  scope: [] (stack cleanup:1) --->
// CHECK:STDOUT: (-) stack-pop:  CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: --- step StatementAction pos: 1 `var x: i32 = N.Foo(0);` results: [`1`]  (phase_all.carbon:16) --->
// CHECK:STDOUT: +++ memory-read: #1 `1`
// CHECK:STDOUT: match pattern Placeholder<x>
// CHECK:STDOUT: from initializing expression with value 1
// CHECK:STDOUT: (-) stack-pop:  StatementAction pos: 1 `var x: i32 = N.Foo(0);` results: [`1`]  (phase_all.carbon:16)
// CHECK:STDOUT: --- step StatementAction pos: 1 `{var x: i32 = N.Foo(0);return x;}` scope: [`x: i32`: `lval<Allocation(1)>`] (phase_all.carbon:18) --->
// CHECK:STDOUT: (+) stack-push: StatementAction pos: 0 `return x;` (phase_all.carbon:17)
// CHECK:STDOUT: --- step StatementAction pos: 0 `return x;` (phase_all.carbon:17) --->
// CHECK:STDOUT: (+) stack-push: ValueExpressionAction pos: 0 `x` (phase_all.carbon:17)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 0 `x` (phase_all.carbon:17) --->
// CHECK:STDOUT: (+) stack-push: ExpressionAction pos: 0 `x` (phase_all.carbon:17)
// CHECK:STDOUT: --- step ExpressionAction pos: 0 `x` (phase_all.carbon:17) --->
// CHECK:STDOUT: +++ memory-read: #1 `1`
// CHECK:STDOUT: (-) stack-pop:  ExpressionAction pos: 0 `x` (phase_all.carbon:17)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 1 `x` results: [`ref_expr<Allocation(1)>`]  (phase_all.carbon:17) --->
// CHECK:STDOUT: (-) stack-pop:  ValueExpressionAction pos: 1 `x` results: [`ref_expr<Allocation(1)>`]  (phase_all.carbon:17)
// CHECK:STDOUT: --- step StatementAction pos: 1 `return x;` results: [`1`]  (phase_all.carbon:17) --->
// CHECK:STDOUT: (-) stack-pop:  StatementAction pos: 1 `return x;` results: [`1`]  (phase_all.carbon:17)
// CHECK:STDOUT: (-) stack-pop:  StatementAction pos: 2 `{var x: i32 = N.Foo(0);return x;}` scope: [`x: i32`: `lval<Allocation(1)>`] (phase_all.carbon:18)
// CHECK:STDOUT: (-) stack-pop:  ScopeAction pos: 0  scope: [] (None)
// CHECK:STDOUT: (+) stack-push: CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: (+) stack-push: CleanUpAction pos: 0  scope: [`x: i32`: `lval<Allocation(1)>`] (stack cleanup:1)
// CHECK:STDOUT: --- step CleanUpAction pos: 0  scope: [`x: i32`: `lval<Allocation(1)>`] (stack cleanup:1) --->
// CHECK:STDOUT: +++ memory-read: #1 `1`
// CHECK:STDOUT: (+) stack-push: DestroyAction pos: 0  (None)
// CHECK:STDOUT: --- step DestroyAction pos: 0  (None) --->
// CHECK:STDOUT: (-) stack-pop:  DestroyAction pos: 0  (None)
// CHECK:STDOUT: --- step CleanUpAction pos: 1  scope: [`x: i32`: `lval<Allocation(1)>`] (stack cleanup:1) --->
// CHECK:STDOUT: (-) memory-dealloc: #1 `1`
// CHECK:STDOUT: --- step CleanUpAction pos: 2  scope: [`x: i32`: `lval<Allocation(1)>`] (stack cleanup:1) --->
// CHECK:STDOUT: (-) stack-pop:  CleanUpAction pos: 2  scope: [`x: i32`: `lval<Allocation(1)>`] (stack cleanup:1)
// CHECK:STDOUT: --- step CleanUpAction pos: 0  scope: [] (stack cleanup:1) --->
// CHECK:STDOUT: (-) stack-pop:  CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: --- step ExpressionAction pos: 2 `Main()` results: [`fun<Main>`, `1`]  scope: [] (<Main()>:0) --->
// CHECK:STDOUT: (-) stack-pop:  ExpressionAction pos: 2 `Main()` results: [`fun<Main>`, `1`]  scope: [] (<Main()>:0)
// CHECK:STDOUT: (+) stack-push: CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: --- step CleanUpAction pos: 0  scope: [] (stack cleanup:1) --->
// CHECK:STDOUT: (-) stack-pop:  CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: --- step ValueExpressionAction pos: 1 `Main()` results: [`1`]  (<Main()>:0) --->
// CHECK:STDOUT: (-) stack-pop:  ValueExpressionAction pos: 1 `Main()` results: [`1`]  (<Main()>:0)
// CHECK:STDOUT: interpreter result: 1
// CHECK:STDOUT: ********** printing timing **********
// CHECK:STDOUT: Time elapsed in ExecProgram: {{\d+}}ms
// CHECK:STDOUT: Time elapsed in AnalyzeProgram: {{\d+}}ms
// CHECK:STDOUT: Time elapsed in AddPrelude: {{\d+}}ms
// CHECK:STDOUT: Time elapsed in Parse: {{\d+}}ms
// CHECK:STDOUT: result: 1
