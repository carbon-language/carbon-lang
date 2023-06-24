// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ********** source program **********
// CHECK:STDOUT: fn Main ()-> i32 {
// CHECK:STDOUT: {
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: }
// CHECK:STDOUT: ********** resolving names **********
// CHECK:STDOUT: ********** resolving control flow **********
// CHECK:STDOUT: ********** type checking **********
// CHECK:STDOUT: Omitting prelude type checking traces...
// CHECK:STDOUT: Finished prelude, resuming traces...
// CHECK:STDOUT: ** declaring function Main
// CHECK:STDOUT: checking TuplePattern ()
// CHECK:STDOUT: checking IntTypeLiteral i32
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }

package ExplorerTest api;

// CHECK:STDOUT: --- step exp i32 .0. (full_trace.carbon:[[@LINE+1173]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: ** finished declaring function Main of type fn () -> i32
// CHECK:STDOUT: checking FunctionDeclaration
// CHECK:STDOUT: ** checking function Main
// CHECK:STDOUT: impl declarations:
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
// CHECK:STDOUT: performing argument deduction for bindings:
// CHECK:STDOUT: deduction succeeded with results: {}
// CHECK:STDOUT: ********** resolving unformed variables **********
// CHECK:STDOUT: ********** printing declarations **********
// CHECK:STDOUT: fn Main ()-> i32 {
// CHECK:STDOUT: {
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: }
// CHECK:STDOUT: ********** starting execution **********
// CHECK:STDOUT: ********** initializing globals **********
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface As .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface ImplicitAs .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface __EqualConverter .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl fn __EqualConvert .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl forall [U:! type] U as __EqualConverter where .T = U .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl match_first .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl forall [U1:! type, T1:! As(U1)] (T1) as As((U1)) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl forall [U1:! type, U2:! type, T1:! As(U1), T2:! As(U2)] (T1, T2) as As((U1, U2)) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl forall [U1:! type, U2:! type, U3:! type, T1:! As(U1), T2:! As(U2), T3:! As(U3)] (T1, T2, T3) as As((U1, U2, U3)) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface EqWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint Eq .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl forall [T2:! type, U2:! type, T1:! EqWith(T2), U1:! EqWith(U2)] (T1, U1) as EqWith((T2, U2)) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl bool as EqWith(Self) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as EqWith(Self) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl String as EqWith(Self) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl choice Ordering .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface CompareWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint Ordered .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as CompareWith(Self) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl String as CompareWith(Self) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface LessWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface LessEqWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface GreaterWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface GreaterEqWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as LessWith(Self) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl String as LessWith(Self) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as LessEqWith(Self) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl String as LessEqWith(Self) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as GreaterWith(Self) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl String as GreaterWith(Self) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as GreaterEqWith(Self) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl String as GreaterEqWith(Self) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface Negate .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface AddWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint Add .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface SubWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint Sub .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface MulWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint Mul .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface DivWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint Div .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface ModWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint Mod .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as Negate where .Result = i32 .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as AddWith(i32) where .Result = i32 .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as SubWith(i32) where .Result = i32 .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as MulWith(i32) where .Result = i32 .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as DivWith(i32) where .Result = i32 .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as ModWith(i32) where .Result = i32 .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface BitComplement .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface BitAndWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint BitAnd .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface BitOrWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint BitOr .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface BitXorWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint BitXor .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface LeftShiftWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint LeftShift .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface RightShiftWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint RightShift .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as BitComplement where .Result = i32 .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as BitAndWith(i32) where .Result = i32 .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as BitOrWith(i32) where .Result = i32 .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as BitXorWith(i32) where .Result = i32 .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as LeftShiftWith(i32) where .Result = i32 .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as RightShiftWith(i32) where .Result = i32 .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface AssignWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint Assign .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface AddAssignWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint AddAssign .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface SubAssignWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint SubAssign .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface MulAssignWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint MulAssign .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface DivAssignWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint DivAssign .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface ModAssignWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint ModAssign .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface BitAndAssignWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint BitAssignAnd .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface BitOrAssignWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint BitAssignOr .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface BitXorAssignWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint BitAssignXor .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface LeftShiftAssignWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint LeftShiftAssign .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface RightShiftAssignWith .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl constraint RightShiftAssign .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl forall [T:! type, U:! ImplicitAs(T)] T as AssignWith(U) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl forall [U:! type, T:! AddWith(U) where .Self impls AssignWith(.Self.Result)] T as AddAssignWith(U) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl forall [U:! type, T:! SubWith(U) where .Self impls AssignWith(.Self.Result)] T as SubAssignWith(U) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl forall [U:! type, T:! MulWith(U) where .Self impls AssignWith(.Self.Result)] T as MulAssignWith(U) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl forall [U:! type, T:! DivWith(U) where .Self impls AssignWith(.Self.Result)] T as DivAssignWith(U) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl forall [U:! type, T:! ModWith(U) where .Self impls AssignWith(.Self.Result)] T as ModAssignWith(U) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl forall [U:! type, T:! BitAndWith(U) where .Self impls AssignWith(.Self.Result)] T as BitAndAssignWith(U) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl forall [U:! type, T:! BitOrWith(U) where .Self impls AssignWith(.Self.Result)] T as BitOrAssignWith(U) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl forall [U:! type, T:! BitXorWith(U) where .Self impls AssignWith(.Self.Result)] T as BitXorAssignWith(U) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl forall [U:! type, T:! LeftShiftWith(U) where .Self impls AssignWith(.Self.Result)] T as LeftShiftAssignWith(U) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl forall [U:! type, T:! RightShiftWith(U) where .Self impls AssignWith(.Self.Result)] T as RightShiftAssignWith(U) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface Inc .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl interface Dec .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as Inc .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl impl i32 as Dec .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl fn Assert .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl fn Rand .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl choice OptionalElement .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl class Optional .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl class Heap .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl var heap: Heap .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: {}.(interface ImplicitAs(T = class Heap).Convert)() .0. ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp {}.(interface ImplicitAs(T = class Heap).Convert)() .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: {}.(interface ImplicitAs(T = class Heap).Convert) .0. ## {}.(interface ImplicitAs(T = class Heap).Convert)() .1. ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp {}.(interface ImplicitAs(T = class Heap).Convert) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: {} .0. ## {}.(interface ImplicitAs(T = class Heap).Convert) .1. ## {}.(interface ImplicitAs(T = class Heap).Convert)() .1. ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp {} .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: {}.(interface ImplicitAs(T = class Heap).Convert) .1. {{[[][[]}}{}]] ## {}.(interface ImplicitAs(T = class Heap).Convert)() .1. ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp {}.(interface ImplicitAs(T = class Heap).Convert) .1. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: witness for impl T as ImplicitAs(U) .0. ## {}.(interface ImplicitAs(T = class Heap).Convert) .2. {{[[][[]}}{}]] ## {}.(interface ImplicitAs(T = class Heap).Convert)() .1. ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step witness witness for impl T as ImplicitAs(U) .0. --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: {}.(interface ImplicitAs(T = class Heap).Convert) .2. {{[[][[]}}{}, witness for impl T as ImplicitAs(U)]] ## {}.(interface ImplicitAs(T = class Heap).Convert)() .1. ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp {}.(interface ImplicitAs(T = class Heap).Convert) .2. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: interface As(T = class Heap) .0. ## {}.(interface ImplicitAs(T = class Heap).Convert) .3. {{[[][[]}}{}, witness for impl T as ImplicitAs(U)]] ## {}.(interface ImplicitAs(T = class Heap).Convert)() .1. ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: {}.(interface ImplicitAs(T = class Heap).Convert) .3. {{[[][[]}}{}, witness for impl T as ImplicitAs(U), interface As(T = class Heap)]] ## {}.(interface ImplicitAs(T = class Heap).Convert)() .1. ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp {}.(interface ImplicitAs(T = class Heap).Convert) .3. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: {}.(interface ImplicitAs(T = class Heap).Convert) .4. {{[[][[]}}{}, witness for impl T as ImplicitAs(U), interface As(T = class Heap)]] ## {}.(interface ImplicitAs(T = class Heap).Convert)() .1. ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp {}.(interface ImplicitAs(T = class Heap).Convert) .4. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: {}.(interface ImplicitAs(T = class Heap).Convert)() .1. {{[[][[]}}bound_method<Convert>]] ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp {}.(interface ImplicitAs(T = class Heap).Convert)() .1. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: () .0. ## {}.(interface ImplicitAs(T = class Heap).Convert)() .2. {{[[][[]}}bound_method<Convert>]] ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp () .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: {}.(interface ImplicitAs(T = class Heap).Convert)() .2. {{[[][[]}}bound_method<Convert>, ()]] ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory:
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp {}.(interface ImplicitAs(T = class Heap).Convert)() .2. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: calling function: bound_method<Convert>
// CHECK:STDOUT: match pattern ()
// CHECK:STDOUT: with value ()
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: {return __intrinsic_implicit_as_convert(self, U);} .0. ## .0. {self: Self: lval<Allocation(0)>} ## {}.(interface ImplicitAs(T = class Heap).Convert)() .3. {{[[][[]}}bound_method<Convert>, ()]] {} ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory: 0: {}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step stmt {return __intrinsic_implicit_as_convert(self, U);} .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: return __intrinsic_implicit_as_convert(self, U); .0. ## {return __intrinsic_implicit_as_convert(self, U);} .1. {} ## .0. {self: Self: lval<Allocation(0)>} ## {}.(interface ImplicitAs(T = class Heap).Convert)() .3. {{[[][[]}}bound_method<Convert>, ()]] {} ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory: 0: {}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step stmt return __intrinsic_implicit_as_convert(self, U); .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: __intrinsic_implicit_as_convert(self, U) .0. ## return __intrinsic_implicit_as_convert(self, U); .1. ## {return __intrinsic_implicit_as_convert(self, U);} .1. {} ## .0. {self: Self: lval<Allocation(0)>} ## {}.(interface ImplicitAs(T = class Heap).Convert)() .3. {{[[][[]}}bound_method<Convert>, ()]] {} ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory: 0: {}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp __intrinsic_implicit_as_convert(self, U) .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: self .0. ## return __intrinsic_implicit_as_convert(self, U); .1. ## {return __intrinsic_implicit_as_convert(self, U);} .1. {} ## .0. {self: Self: lval<Allocation(0)>} ## {}.(interface ImplicitAs(T = class Heap).Convert)() .3. {{[[][[]}}bound_method<Convert>, ()]] {} ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory: 0: {}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp self .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: self .0. ## self .1. ## return __intrinsic_implicit_as_convert(self, U); .1. ## {return __intrinsic_implicit_as_convert(self, U);} .1. {} ## .0. {self: Self: lval<Allocation(0)>} ## {}.(interface ImplicitAs(T = class Heap).Convert)() .3. {{[[][[]}}bound_method<Convert>, ()]] {} ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory: 0: {}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp self .0. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: self .1. {{[[][[]}}{}]] ## return __intrinsic_implicit_as_convert(self, U); .1. ## {return __intrinsic_implicit_as_convert(self, U);} .1. {} ## .0. {self: Self: lval<Allocation(0)>} ## {}.(interface ImplicitAs(T = class Heap).Convert)() .3. {{[[][[]}}bound_method<Convert>, ()]] {} ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory: 0: {}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp self .1. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: class Heap .0. ## self .2. {{[[][[]}}{}]] ## return __intrinsic_implicit_as_convert(self, U); .1. ## {return __intrinsic_implicit_as_convert(self, U);} .1. {} ## .0. {self: Self: lval<Allocation(0)>} ## {}.(interface ImplicitAs(T = class Heap).Convert)() .3. {{[[][[]}}bound_method<Convert>, ()]] {} ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory: 0: {}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: self .2. {{[[][[]}}{}, class Heap]] ## return __intrinsic_implicit_as_convert(self, U); .1. ## {return __intrinsic_implicit_as_convert(self, U);} .1. {} ## .0. {self: Self: lval<Allocation(0)>} ## {}.(interface ImplicitAs(T = class Heap).Convert)() .3. {{[[][[]}}bound_method<Convert>, ()]] {} ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory: 0: {}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp self .2. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: return __intrinsic_implicit_as_convert(self, U); .1. {{[[][[]}}Heap{}]] ## {return __intrinsic_implicit_as_convert(self, U);} .1. {} ## .0. {self: Self: lval<Allocation(0)>} ## {}.(interface ImplicitAs(T = class Heap).Convert)() .3. {{[[][[]}}bound_method<Convert>, ()]] {} ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory: 0: {}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step stmt return __intrinsic_implicit_as_convert(self, U); .1. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.0. {} ## clean up.0. {self: Self: lval<Allocation(0)>} ## {}.(interface ImplicitAs(T = class Heap).Convert)() .3. {{[[][[]}}bound_method<Convert>, (), Heap{}]] {} ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory: 0: {}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.0. {self: Self: lval<Allocation(0)>} ## {}.(interface ImplicitAs(T = class Heap).Convert)() .3. {{[[][[]}}bound_method<Convert>, (), Heap{}]] {} ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory: 0: {}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: {}.(interface ImplicitAs(T = class Heap).Convert)() .3. {{[[][[]}}bound_method<Convert>, (), Heap{}]] {} ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1.
// CHECK:STDOUT: memory: 0: {}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp {}.(interface ImplicitAs(T = class Heap).Convert)() .3. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.0. {} ## var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1. {{[[][[]}}Heap{}]]
// CHECK:STDOUT: memory: 0: {}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: var heap: Heap = {}.(interface ImplicitAs(T = class Heap).Convert)();
// CHECK:STDOUT:  .1. {{[[][[]}}Heap{}]]
// CHECK:STDOUT: memory: 0: {}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step decl var heap: Heap .1. ({{.*}}/explorer/data/prelude.carbon:{{.*}}) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
fn Main() -> i32 {
  return 0;
// CHECK:STDOUT: --- step decl fn Main .0. (full_trace.carbon:[[@LINE+82]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: ********** calling main function **********
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp Main() .0. (<Main()>:0) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: Main .0. ## Main() .1.
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp Main .0. (<Main()>:0) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: Main() .1. {{[[][[]}}fun<Main>]]
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp Main() .1. (<Main()>:0) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: () .0. ## Main() .2. {{[[][[]}}fun<Main>]]
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp () .0. (<Main()>:0) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: Main() .2. {{[[][[]}}fun<Main>, ()]]
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp Main() .2. (<Main()>:0) --->
// CHECK:STDOUT: calling function: fun<Main>
// CHECK:STDOUT: match pattern ()
// CHECK:STDOUT: with value ()
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: {return 0;} .0. ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step stmt {return 0;} .0. (full_trace.carbon:[[@LINE+44]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: return 0; .0. ## {return 0;} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step stmt return 0; .0. (full_trace.carbon:[[@LINE-44]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: 0 .0. ## return 0; .1. ## {return 0;} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp 0 .0. (full_trace.carbon:[[@LINE-49]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: return 0; .1. {{[[][[]}}0]] ## {return 0;} .1. {} ## .0. {} ## Main() .3. {{[[][[]}}fun<Main>, ()]] {}
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step stmt return 0; .1. (full_trace.carbon:[[@LINE-54]]) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.0. {} ## clean up.0. {} ## Main() .3. {{[[][[]}}fun<Main>, (), 0]] {}
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.0. {} ## Main() .3. {{[[][[]}}fun<Main>, (), 0]] {}
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: Main() .3. {{[[][[]}}fun<Main>, (), 0]] {}
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: --- step exp Main() .3. (<Main()>:0) --->
// CHECK:STDOUT: {
// CHECK:STDOUT: stack: clean up.0. {}
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: {
// CHECK:STDOUT: stack:
// CHECK:STDOUT: memory: 0: {}, 1: Heap{}
// CHECK:STDOUT: }
// CHECK:STDOUT: interpreter result: 0
// CHECK:STDOUT: ********** printing timing **********
// CHECK:STDOUT: Time elapsed in ExecProgram: 0ms
// CHECK:STDOUT: Time elapsed in AnalyzeProgram: 8ms
// CHECK:STDOUT: Time elapsed in AddPrelude: 2ms
// CHECK:STDOUT: Time elapsed in Parse: 0ms
// CHECK:STDOUT: result: 0
}
