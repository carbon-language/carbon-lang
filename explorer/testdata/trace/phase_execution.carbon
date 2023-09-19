// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExplorerTest api;

class T {}

fn Main() -> i32 {
  var x: T = {};
  return 0;
}

// Place checks after code so that line numbers are stable, reducing merge
// conflicts.
// ARGS: --trace_file=- --trace_phase=execution %s
// AUTOUPDATE

// CHECK:STDOUT: * * * * * * * * * *  starting execution  * * * * * * * * * *
// CHECK:STDOUT: ------------------------------------------------------------
// CHECK:STDOUT:
// CHECK:STDOUT: - - - - -  initializing globals  - - - - -
// CHECK:STDOUT: ------------------------------------------
// CHECK:STDOUT: >[] stack-push: DeclarationAction pos: 0 `class T` (phase_execution.carbon:7)
// CHECK:STDOUT: ->> step DeclarationAction pos: 0 `class T` (phase_execution.carbon:7) --->
// CHECK:STDOUT:
// CHECK:STDOUT: *** declaration at (phase_execution.carbon:7)
// CHECK:STDOUT: ```
// CHECK:STDOUT: class T {
// CHECK:STDOUT: }
// CHECK:STDOUT: ```
// CHECK:STDOUT: <[] stack-pop:  DeclarationAction pos: 0 `class T` (phase_execution.carbon:7)
// CHECK:STDOUT: >[] stack-push: DeclarationAction pos: 0 `fn Main` (phase_execution.carbon:12)
// CHECK:STDOUT: ->> step DeclarationAction pos: 0 `fn Main` (phase_execution.carbon:12) --->
// CHECK:STDOUT:
// CHECK:STDOUT: *** declaration at (phase_execution.carbon:12)
// CHECK:STDOUT: ```
// CHECK:STDOUT: fn Main () -> i32
// CHECK:STDOUT: {
// CHECK:STDOUT:   var x: T = {}.(interface ImplicitAs(T = class T).Convert)();
// CHECK:STDOUT:   return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT: ```
// CHECK:STDOUT: <[] stack-pop:  DeclarationAction pos: 0 `fn Main` (phase_execution.carbon:12)
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
// CHECK:STDOUT: >[] stack-push: StatementAction pos: 0 `{ ... }` (phase_execution.carbon:12)
// CHECK:STDOUT: ->> step StatementAction pos: 0 `{ ... }` (phase_execution.carbon:12) --->
// CHECK:STDOUT:
// CHECK:STDOUT: *** statement at (phase_execution.carbon:12)
// CHECK:STDOUT: ```
// CHECK:STDOUT: {
// CHECK:STDOUT:   var x: T = {}.(interface ImplicitAs(T = class T).Convert)();
// CHECK:STDOUT:   return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT: ```
// CHECK:STDOUT: >[] stack-push: StatementAction pos: 0 `var ... = ...;` (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step StatementAction pos: 0 `var ... = ...;` (phase_execution.carbon:10) --->
// CHECK:STDOUT:
// CHECK:STDOUT: *** statement at (phase_execution.carbon:10)
// CHECK:STDOUT: ```
// CHECK:STDOUT: var x: T = {}.(interface ImplicitAs(T = class T).Convert)();
// CHECK:STDOUT: ```
// CHECK:STDOUT: ++# memory-alloc: #1 `Uninit<class T>` uninitialized
// CHECK:STDOUT: >[] stack-push: ExpressionAction pos: 0 `{}.(interface ImplicitAs(T = class T).Convert)()` (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step ExpressionAction pos: 0 `{}.(interface ImplicitAs(T = class T).Convert)()` (phase_execution.carbon:10) --->
// CHECK:STDOUT: >[] stack-push: ValueExpressionAction pos: 0 `{}.(interface ImplicitAs(T = class T).Convert)` (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 0 `{}.(interface ImplicitAs(T = class T).Convert)` (phase_execution.carbon:10) --->
// CHECK:STDOUT: >[] stack-push: ExpressionAction pos: 0 `{}.(interface ImplicitAs(T = class T).Convert)` (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step ExpressionAction pos: 0 `{}.(interface ImplicitAs(T = class T).Convert)` (phase_execution.carbon:10) --->
// CHECK:STDOUT: >[] stack-push: ValueExpressionAction pos: 0 `{}` (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 0 `{}` (phase_execution.carbon:10) --->
// CHECK:STDOUT: >[] stack-push: ExpressionAction pos: 0 `{}` (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step ExpressionAction pos: 0 `{}` (phase_execution.carbon:10) --->
// CHECK:STDOUT: <[] stack-pop:  ExpressionAction pos: 0 `{}` (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 1 `{}` results: [`{}`]  (phase_execution.carbon:10) --->
// CHECK:STDOUT: <[] stack-pop:  ValueExpressionAction pos: 1 `{}` results: [`{}`]  (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step ExpressionAction pos: 1 `{}.(interface ImplicitAs(T = class T).Convert)` results: [`{}`]  (phase_execution.carbon:10) --->
// CHECK:STDOUT: >[] stack-push: WitnessAction pos: 0 `witness for impl T as ImplicitAs(U)` (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step WitnessAction pos: 0 `witness for impl T as ImplicitAs(U)` (phase_execution.carbon:10) --->
// CHECK:STDOUT: <[] stack-pop:  WitnessAction pos: 0 `witness for impl T as ImplicitAs(U)` (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step ExpressionAction pos: 2 `{}.(interface ImplicitAs(T = class T).Convert)` results: [`{}`, `witness for impl T as ImplicitAs(U)`]  (phase_execution.carbon:10) --->
// CHECK:STDOUT: >[] stack-push: TypeInstantiationAction pos: 0 `interface As(T = class T)` (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step TypeInstantiationAction pos: 0 `interface As(T = class T)` (phase_execution.carbon:10) --->
// CHECK:STDOUT: ->> instantiating type `interface As(T = class T)` (phase_execution.carbon:10)
// CHECK:STDOUT: ->> instantiating type `class T` (phase_execution.carbon:10)
// CHECK:STDOUT: <<- instantiated type `class T` as `class T` (phase_execution.carbon:10)
// CHECK:STDOUT: <<- instantiated type `interface As(T = class T)` as `interface As(T = class T)` (phase_execution.carbon:10)
// CHECK:STDOUT: <[] stack-pop:  TypeInstantiationAction pos: 0 `interface As(T = class T)` (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step ExpressionAction pos: 3 `{}.(interface ImplicitAs(T = class T).Convert)` results: [`{}`, `witness for impl T as ImplicitAs(U)`, `interface As(T = class T)`]  (phase_execution.carbon:10) --->
// CHECK:STDOUT: ->> step ExpressionAction pos: 4 `{}.(interface ImplicitAs(T = class T).Convert)` results: [`{}`, `witness for impl T as ImplicitAs(U)`, `interface As(T = class T)`]  (phase_execution.carbon:10) --->
// CHECK:STDOUT: <[] stack-pop:  ExpressionAction pos: 4 `{}.(interface ImplicitAs(T = class T).Convert)` results: [`{}`, `witness for impl T as ImplicitAs(U)`, `interface As(T = class T)`]  (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 1 `{}.(interface ImplicitAs(T = class T).Convert)` results: [`bound_method<Convert>`]  (phase_execution.carbon:10) --->
// CHECK:STDOUT: <[] stack-pop:  ValueExpressionAction pos: 1 `{}.(interface ImplicitAs(T = class T).Convert)` results: [`bound_method<Convert>`]  (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step ExpressionAction pos: 1 `{}.(interface ImplicitAs(T = class T).Convert)()` results: [`bound_method<Convert>`]  scope: [] (phase_execution.carbon:10) --->
// CHECK:STDOUT: -() calling function: bound_method<Convert>
// CHECK:STDOUT: === match pattern `Placeholder<self>`
// CHECK:STDOUT:     from value expression with value `{}`
// CHECK:STDOUT: === match pattern `()`
// CHECK:STDOUT:     from value expression with value `()`
// CHECK:STDOUT: >[] stack-push: ScopeAction pos: 0  scope: [`self: Self`: `{}`] (None)
// CHECK:STDOUT: >[] stack-push: StatementAction pos: 0 `{ ... }` (prelude.carbon:{{\d+}})
// CHECK:STDOUT: ->> step StatementAction pos: 0 `{ ... }` (prelude.carbon:{{\d+}}) --->
// CHECK:STDOUT:
// CHECK:STDOUT: *** statement at (prelude.carbon:{{\d+}})
// CHECK:STDOUT: ```
// CHECK:STDOUT: {
// CHECK:STDOUT:   return __intrinsic_implicit_as_convert(self, U);
// CHECK:STDOUT: }
// CHECK:STDOUT: ```
// CHECK:STDOUT: >[] stack-push: StatementAction pos: 0 `return ...;` (prelude.carbon:{{\d+}})
// CHECK:STDOUT: ->> step StatementAction pos: 0 `return ...;` (prelude.carbon:{{\d+}}) --->
// CHECK:STDOUT:
// CHECK:STDOUT: *** statement at (prelude.carbon:{{\d+}})
// CHECK:STDOUT: ```
// CHECK:STDOUT: return __intrinsic_implicit_as_convert(self, U);
// CHECK:STDOUT: ```
// CHECK:STDOUT: >[] stack-push: ValueExpressionAction pos: 0 `__intrinsic_implicit_as_convert(self, U)` (prelude.carbon:{{\d+}})
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 0 `__intrinsic_implicit_as_convert(self, U)` (prelude.carbon:{{\d+}}) --->
// CHECK:STDOUT: >[] stack-push: ExpressionAction pos: 0 `__intrinsic_implicit_as_convert(self, U)` (prelude.carbon:{{\d+}})
// CHECK:STDOUT: ->> step ExpressionAction pos: 0 `__intrinsic_implicit_as_convert(self, U)` (prelude.carbon:{{\d+}}) --->
// CHECK:STDOUT: <[] stack-pop:  ExpressionAction pos: 0 `__intrinsic_implicit_as_convert(self, U)` (prelude.carbon:{{\d+}})
// CHECK:STDOUT: >[] stack-push: ExpressionAction pos: 0 `self` (prelude.carbon:{{\d+}})
// CHECK:STDOUT: ->> step ExpressionAction pos: 0 `self` (prelude.carbon:{{\d+}}) --->
// CHECK:STDOUT: >[] stack-push: ValueExpressionAction pos: 0 `self` (prelude.carbon:{{\d+}})
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 0 `self` (prelude.carbon:{{\d+}}) --->
// CHECK:STDOUT: >[] stack-push: ExpressionAction pos: 0 `self` (prelude.carbon:{{\d+}})
// CHECK:STDOUT: ->> step ExpressionAction pos: 0 `self` (prelude.carbon:{{\d+}}) --->
// CHECK:STDOUT: <[] stack-pop:  ExpressionAction pos: 0 `self` (prelude.carbon:{{\d+}})
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 1 `self` results: [`{}`]  (prelude.carbon:{{\d+}}) --->
// CHECK:STDOUT: <[] stack-pop:  ValueExpressionAction pos: 1 `self` results: [`{}`]  (prelude.carbon:{{\d+}})
// CHECK:STDOUT: ->> step ExpressionAction pos: 1 `self` results: [`{}`]  (prelude.carbon:{{\d+}}) --->
// CHECK:STDOUT: >[] stack-push: TypeInstantiationAction pos: 0 `class T` (prelude.carbon:{{\d+}})
// CHECK:STDOUT: ->> step TypeInstantiationAction pos: 0 `class T` (prelude.carbon:{{\d+}}) --->
// CHECK:STDOUT: <[] stack-pop:  TypeInstantiationAction pos: 0 `class T` (prelude.carbon:{{\d+}})
// CHECK:STDOUT: ->> step ExpressionAction pos: 2 `self` results: [`{}`, `class T`]  (prelude.carbon:{{\d+}}) --->
// CHECK:STDOUT: ->> instantiating type `class T` (prelude.carbon:{{\d+}})
// CHECK:STDOUT: <<- instantiated type `class T` as `class T` (prelude.carbon:{{\d+}})
// CHECK:STDOUT: <[] stack-pop:  ExpressionAction pos: 2 `self` results: [`{}`, `class T`]  (prelude.carbon:{{\d+}})
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 1 `__intrinsic_implicit_as_convert(self, U)` results: [`T{}`]  (prelude.carbon:{{\d+}}) --->
// CHECK:STDOUT: <[] stack-pop:  ValueExpressionAction pos: 1 `__intrinsic_implicit_as_convert(self, U)` results: [`T{}`]  (prelude.carbon:{{\d+}})
// CHECK:STDOUT: ->> step StatementAction pos: 1 `return ...;` results: [`T{}`]  (prelude.carbon:{{\d+}}) --->
// CHECK:STDOUT:
// CHECK:STDOUT: *** statement at (prelude.carbon:{{\d+}})
// CHECK:STDOUT: ```
// CHECK:STDOUT: return __intrinsic_implicit_as_convert(self, U);
// CHECK:STDOUT: ```
// CHECK:STDOUT: --> memory-write: #1 `T{}`
// CHECK:STDOUT: <[] stack-pop:  StatementAction pos: 1 `return ...;` results: [`T{}`]  (prelude.carbon:{{\d+}})
// CHECK:STDOUT: <[] stack-pop:  StatementAction pos: 1 `{ ... }` scope: [] (prelude.carbon:{{\d+}})
// CHECK:STDOUT: <[] stack-pop:  ScopeAction pos: 0  scope: [`self: Self`: `{}`] (None)
// CHECK:STDOUT: >[] stack-push: CleanUpAction pos: 0  scope: [`self: Self`: `{}`] (stack cleanup:1)
// CHECK:STDOUT: >[] stack-push: CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: ->> step CleanUpAction pos: 0  scope: [] (stack cleanup:1) --->
// CHECK:STDOUT: <[] stack-pop:  CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: ->> step CleanUpAction pos: 0  scope: [`self: Self`: `{}`] (stack cleanup:1) --->
// CHECK:STDOUT: <[] stack-pop:  CleanUpAction pos: 0  scope: [`self: Self`: `{}`] (stack cleanup:1)
// CHECK:STDOUT: ->> step ExpressionAction pos: 2 `{}.(interface ImplicitAs(T = class T).Convert)()` results: [`bound_method<Convert>`, `T{}`]  scope: [] (phase_execution.carbon:10) --->
// CHECK:STDOUT: <[] stack-pop:  ExpressionAction pos: 2 `{}.(interface ImplicitAs(T = class T).Convert)()` results: [`bound_method<Convert>`, `T{}`]  scope: [] (phase_execution.carbon:10)
// CHECK:STDOUT: >[] stack-push: CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: ->> step CleanUpAction pos: 0  scope: [] (stack cleanup:1) --->
// CHECK:STDOUT: <[] stack-pop:  CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: ->> step StatementAction pos: 1 `var ... = ...;` results: [`T{}`]  (phase_execution.carbon:10) --->
// CHECK:STDOUT:
// CHECK:STDOUT: *** statement at (phase_execution.carbon:10)
// CHECK:STDOUT: ```
// CHECK:STDOUT: var x: T = {}.(interface ImplicitAs(T = class T).Convert)();
// CHECK:STDOUT: ```
// CHECK:STDOUT: <-- memory-read: #1 `T{}`
// CHECK:STDOUT: === match pattern `Placeholder<x>`
// CHECK:STDOUT:     from initializing expression with value `T{}`
// CHECK:STDOUT: <[] stack-pop:  StatementAction pos: 1 `var ... = ...;` results: [`T{}`]  (phase_execution.carbon:10)
// CHECK:STDOUT: ->> step StatementAction pos: 1 `{ ... }` scope: [`x: T`: `lval<Allocation(1)>`] (phase_execution.carbon:12) --->
// CHECK:STDOUT:
// CHECK:STDOUT: *** statement at (phase_execution.carbon:12)
// CHECK:STDOUT: ```
// CHECK:STDOUT: {
// CHECK:STDOUT:   var x: T = {}.(interface ImplicitAs(T = class T).Convert)();
// CHECK:STDOUT:   return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT: ```
// CHECK:STDOUT: >[] stack-push: StatementAction pos: 0 `return ...;` (phase_execution.carbon:11)
// CHECK:STDOUT: ->> step StatementAction pos: 0 `return ...;` (phase_execution.carbon:11) --->
// CHECK:STDOUT:
// CHECK:STDOUT: *** statement at (phase_execution.carbon:11)
// CHECK:STDOUT: ```
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: ```
// CHECK:STDOUT: >[] stack-push: ValueExpressionAction pos: 0 `0` (phase_execution.carbon:11)
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 0 `0` (phase_execution.carbon:11) --->
// CHECK:STDOUT: >[] stack-push: ExpressionAction pos: 0 `0` (phase_execution.carbon:11)
// CHECK:STDOUT: ->> step ExpressionAction pos: 0 `0` (phase_execution.carbon:11) --->
// CHECK:STDOUT: <[] stack-pop:  ExpressionAction pos: 0 `0` (phase_execution.carbon:11)
// CHECK:STDOUT: ->> step ValueExpressionAction pos: 1 `0` results: [`0`]  (phase_execution.carbon:11) --->
// CHECK:STDOUT: <[] stack-pop:  ValueExpressionAction pos: 1 `0` results: [`0`]  (phase_execution.carbon:11)
// CHECK:STDOUT: ->> step StatementAction pos: 1 `return ...;` results: [`0`]  (phase_execution.carbon:11) --->
// CHECK:STDOUT:
// CHECK:STDOUT: *** statement at (phase_execution.carbon:11)
// CHECK:STDOUT: ```
// CHECK:STDOUT: return 0;
// CHECK:STDOUT: ```
// CHECK:STDOUT: <[] stack-pop:  StatementAction pos: 1 `return ...;` results: [`0`]  (phase_execution.carbon:11)
// CHECK:STDOUT: <[] stack-pop:  StatementAction pos: 2 `{ ... }` scope: [`x: T`: `lval<Allocation(1)>`] (phase_execution.carbon:12)
// CHECK:STDOUT: <[] stack-pop:  ScopeAction pos: 0  scope: [] (None)
// CHECK:STDOUT: >[] stack-push: CleanUpAction pos: 0  scope: [] (stack cleanup:1)
// CHECK:STDOUT: >[] stack-push: CleanUpAction pos: 0  scope: [`x: T`: `lval<Allocation(1)>`] (stack cleanup:1)
// CHECK:STDOUT: ->> step CleanUpAction pos: 0  scope: [`x: T`: `lval<Allocation(1)>`] (stack cleanup:1) --->
// CHECK:STDOUT: <-- memory-read: #1 `T{}`
// CHECK:STDOUT: >[] stack-push: DestroyAction pos: 0  (None)
// CHECK:STDOUT: ->> step DestroyAction pos: 0  (None) --->
// CHECK:STDOUT: ->> step DestroyAction pos: 1  (None) --->
// CHECK:STDOUT: ->> step DestroyAction pos: 2  (None) --->
// CHECK:STDOUT: <[] stack-pop:  DestroyAction pos: 2  (None)
// CHECK:STDOUT: ->> step CleanUpAction pos: 1  scope: [`x: T`: `lval<Allocation(1)>`] (stack cleanup:1) --->
// CHECK:STDOUT: --# memory-dealloc: #1 `T{}`
// CHECK:STDOUT: ->> step CleanUpAction pos: 2  scope: [`x: T`: `lval<Allocation(1)>`] (stack cleanup:1) --->
// CHECK:STDOUT: <[] stack-pop:  CleanUpAction pos: 2  scope: [`x: T`: `lval<Allocation(1)>`] (stack cleanup:1)
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
