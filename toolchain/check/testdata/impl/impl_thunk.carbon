// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/impl_thunk.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/impl_thunk.carbon

// --- no_thunk_param_name_differs.carbon

library "[[@TEST_NAME]]";

interface I { fn F(x: Self); }

// This is OK, and does not require a thunk despite the parameter name differing.
class C {
  impl as I {
    fn F(not_x: C);
  }
}

// --- struct_conversion.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F(x: {.a: (), .b: {}}) -> {.c: (), .d: {}};
}

impl () as I {
  //@dump-sem-ir-begin
  fn F(y: {.b: {}, .a: ()}) -> {.d: {}, .c: ()};
  //@dump-sem-ir-end
}

// --- inheritance_conversion.carbon

library "[[@TEST_NAME]]";

base class A {}
base class B { extend base: A; }
class C { extend base: B; }

interface X {
  fn F[addr self: Self*](other: Self*) -> Self*;
}

impl B as X {
  //@dump-sem-ir-begin
  fn F[addr self: A*](other: A*) -> C*;
  //@dump-sem-ir-end
}

// --- inheritance_value_conversion.carbon

library "[[@TEST_NAME]]";

base class A {}
class B { extend base: A; }

interface X {
  fn F[self: Self](other: Self);
}

impl B as X {
  //@dump-sem-ir-begin
  fn F[self: A](other: A);
  //@dump-sem-ir-end
}

// --- inheritance_value_conversion_pointer.carbon

library "[[@TEST_NAME]]";

base class A {}
base class B { extend base: A; }
class C { extend base: B; }

interface X {
  fn F[addr self: Self*](other: Self*) -> Self*;
}

impl B as X {
  //@dump-sem-ir-begin
  fn F[addr self: A*](other: A*) -> C*;
  //@dump-sem-ir-end
}

// --- fail_inheritance_value_conversion_copy_return.carbon

library "[[@TEST_NAME]]";

base class A {}
class B { extend base: A; }

interface X {
  fn F() -> Self;
}

impl A as X {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_inheritance_value_conversion_copy_return.carbon:[[@LINE+10]]:3: error: cannot copy value of type `A` [CopyOfUncopyableType]
  // CHECK:STDERR:   fn F() -> B;
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_inheritance_value_conversion_copy_return.carbon:[[@LINE+7]]:3: note: type `A` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   fn F() -> B;
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_inheritance_value_conversion_copy_return.carbon:[[@LINE-11]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn F() -> Self;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F() -> B;
  //@dump-sem-ir-end
}

// --- fail_param_type_mismatch.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_param_type_mismatch.carbon:[[@LINE+6]]:8: error: cannot implicitly convert expression of type `A` to `B` [ConversionFailure]
  // CHECK:STDERR:   fn F(a: Self);
  // CHECK:STDERR:        ^~~~~~~
  // CHECK:STDERR: fail_param_type_mismatch.carbon:[[@LINE+3]]:8: note: type `A` does not implement interface `Core.ImplicitAs(B)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   fn F(a: Self);
  // CHECK:STDERR:        ^~~~~~~
  fn F(a: Self);
}

class A {}
class B {}

impl A as I {
  // CHECK:STDERR: fail_param_type_mismatch.carbon:[[@LINE+7]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F(a: B);
  // CHECK:STDERR:        ^~~~
  // CHECK:STDERR: fail_param_type_mismatch.carbon:[[@LINE-10]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn F(a: Self);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F(a: B);
}

// --- fail_return_mismatch.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F() -> Self;
}

class A {}
class B {}

impl A as I {
  // CHECK:STDERR: fail_return_mismatch.carbon:[[@LINE+10]]:3: error: cannot implicitly convert expression of type `B` to `A` [ConversionFailure]
  // CHECK:STDERR:   fn F() -> B;
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_return_mismatch.carbon:[[@LINE+7]]:3: note: type `B` does not implement interface `Core.ImplicitAs(A)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   fn F() -> B;
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_return_mismatch.carbon:[[@LINE-13]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn F() -> Self;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F() -> B;
}

// --- return_empty_tuple_mismatch_allowed.carbon

library "[[@TEST_NAME]]";

interface I {
  fn HasReturn() -> Self;
  fn NoReturn();
  fn EmptyTupleReturn() -> ();
}

impl () as I {
  // OK, can `return HasReturn();` in thunk.
  fn HasReturn();

  // OK, exact match.
  fn NoReturn();

  // OK, same as `HasReturn`.
  fn EmptyTupleReturn();
}

// --- fail_return_empty_tuple_mismatch.carbon

library "[[@TEST_NAME]]";

interface I {
  fn NoReturn();
}

impl () as I {
  // TODO: The proposal says to reject this. But should we really do so?
  // CHECK:STDERR: fail_return_empty_tuple_mismatch.carbon:[[@LINE+7]]:3: error: function redeclaration differs because return type is `()` [FunctionRedeclReturnTypeDiffers]
  // CHECK:STDERR:   fn NoReturn() -> ();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_return_empty_tuple_mismatch.carbon:[[@LINE-8]]:3: note: previously declared with no return type [FunctionRedeclReturnTypePreviousNoReturn]
  // CHECK:STDERR:   fn NoReturn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn NoReturn() -> ();
}

class C {}

impl C as I {
  // CHECK:STDERR: fail_return_empty_tuple_mismatch.carbon:[[@LINE+7]]:3: error: function redeclaration differs because return type is `C` [FunctionRedeclReturnTypeDiffers]
  // CHECK:STDERR:   fn NoReturn() -> C;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_return_empty_tuple_mismatch.carbon:[[@LINE-21]]:3: note: previously declared with no return type [FunctionRedeclReturnTypePreviousNoReturn]
  // CHECK:STDERR:   fn NoReturn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn NoReturn() -> C;
}

// --- fail_param_type_incomplete.carbon

library "[[@TEST_NAME]]";

interface I(T:! type) {
  // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE+3]]:8: error: parameter has incomplete type `B` in function definition [IncompleteTypeInFunctionParam]
  // CHECK:STDERR:   fn F(a: T);
  // CHECK:STDERR:        ^~~~
  fn F(a: T);
}

// CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE+3]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class B;
// CHECK:STDERR: ^~~~~~~~
class B;

class C {
  impl as I(B) {
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE+17]]:5: note: while building thunk calling this function [ThunkCallee]
    // CHECK:STDERR:     fn F(c: C);
    // CHECK:STDERR:     ^~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE-14]]:8: error: cannot implicitly convert expression of type `B` to `C` [ConversionFailure]
    // CHECK:STDERR:   fn F(a: T);
    // CHECK:STDERR:        ^~~~
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE-17]]:8: note: type `B` does not implement interface `Core.ImplicitAs(C)` [MissingImplInMemberAccessNote]
    // CHECK:STDERR:   fn F(a: T);
    // CHECK:STDERR:        ^~~~
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE+7]]:10: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR:     fn F(c: C);
    // CHECK:STDERR:          ^~~~
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE-23]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR:   fn F(a: T);
    // CHECK:STDERR:   ^~~~~~~~~~~
    // CHECK:STDERR:
    fn F(c: C);
  }
}

// --- generic_function.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F[T:! type](x: T*) -> T*;
}

impl () as I {
  //@dump-sem-ir-begin
  fn F[U:! Core.Copy](x: U) -> U { return x; }
  //@dump-sem-ir-end
}

// --- fail_todo_generic_method.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F[self: Self, T:! type](x: T*) -> T*;
}

// TODO: This fails because `x.(y)`, where `y` is a generic method, fails with
// an "expression cannot be used as a value" error.
impl () as I {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_generic_method.carbon:[[@LINE+7]]:3: error: expression cannot be used as a value [UseOfNonExprAsValue]
  // CHECK:STDERR:   fn F[self: (), U:! Core.Copy](x: U) -> U { return x; }
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_generic_method.carbon:[[@LINE-10]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn F[self: Self, T:! type](x: T*) -> T*;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F[self: (), U:! Core.Copy](x: U) -> U { return x; }
  //@dump-sem-ir-end
}

// --- generic_interface.carbon

library "[[@TEST_NAME]]";

interface I(T:! type) {
  fn F() -> {.a: T, .b: T};
}

impl () as I({}) {
  //@dump-sem-ir-begin
  fn F() -> {.b: {}, .a: {}};
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- struct_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.a.b.391: type = struct_type {.a: %empty_tuple.type, .b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct_type.c.d.15a: type = struct_type {.c: %empty_tuple.type, .d: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct_type.b.a.40c: type = struct_type {.b: %empty_struct_type, .a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.231: type = pattern_type %struct_type.b.a.40c [concrete]
// CHECK:STDOUT:   %struct_type.d.c.b36: type = struct_type {.d: %empty_struct_type, .c: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.844: type = pattern_type %struct_type.d.c.b36 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.e6fdf8.1: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_48.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.f1aebb.1: %empty_tuple.type.as.I.impl.F.type.e6fdf8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.e6fdf8.2: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_48.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.f1aebb.2: %empty_tuple.type.as.I.impl.F.type.e6fdf8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.c.d.15a = struct_value (%empty_tuple, %empty_struct) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %struct_type.d.c.b36, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.94a: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.db8: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.94a = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.eab: type = ptr_type %struct_type.d.c.b36 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc8_7.2 as %I.ref {
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_48.1: %empty_tuple.type.as.I.impl.F.type.e6fdf8.1 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_48.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.f1aebb.1] {
// CHECK:STDOUT:     %y.patt: %pattern_type.231 = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.231 = value_param_pattern %y.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.844 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.844 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_38.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_38.2: type = converted %.loc10_38.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc10_46.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_46.2: type = converted %.loc10_46.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.d.c: type = struct_type {.d: %empty_struct_type, .c: %empty_tuple.type} [concrete = constants.%struct_type.d.c.b36]
// CHECK:STDOUT:     %y.param: %struct_type.b.a.40c = value_param call_param0
// CHECK:STDOUT:     %.loc10_26: type = splice_block %struct_type.b.a [concrete = constants.%struct_type.b.a.40c] {
// CHECK:STDOUT:       %.loc10_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc10_17.2: type = converted %.loc10_17.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc10_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc10_25.2: type = converted %.loc10_25.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.b.a: type = struct_type {.b: %empty_struct_type, .a: %empty_tuple.type} [concrete = constants.%struct_type.b.a.40c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: %struct_type.b.a.40c = bind_name y, %y.param
// CHECK:STDOUT:     %return.param: ref %struct_type.d.c.b36 = out_param call_param1
// CHECK:STDOUT:     %return: ref %struct_type.d.c.b36 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_48.2: %empty_tuple.type.as.I.impl.F.type.e6fdf8.2 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_48.2 [concrete = constants.%empty_tuple.type.as.I.impl.F.f1aebb.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl.loc10_48.1
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F.loc10_48.1(%y.param: %struct_type.b.a.40c) -> %return.param: %struct_type.d.c.b36;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F.loc10_48.2(%x.param: %struct_type.a.b.391) -> %return.param: %struct_type.c.d.15a [thunk @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_48.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %empty_tuple.type.as.I.impl.F.type.e6fdf8.1 = name_ref F, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_48.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.f1aebb.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_48.1: ref %struct_type.d.c.b36 = temporary_storage
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.call: init %struct_type.d.c.b36 = call %F.ref(%.loc5_9.3) to %.loc10_48.1
// CHECK:STDOUT:   %.loc10_48.2: ref %struct_type.d.c.b36 = temporary %.loc10_48.1, %empty_tuple.type.as.I.impl.F.call
// CHECK:STDOUT:   %.loc10_48.3: ref %empty_tuple.type = struct_access %.loc10_48.2, element1
// CHECK:STDOUT:   %.loc10_48.4: ref %empty_tuple.type = struct_access %return, element1
// CHECK:STDOUT:   %.loc10_48.5: init %empty_tuple.type = tuple_init () to %.loc10_48.4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_48.6: init %empty_tuple.type = converted %.loc10_48.3, %.loc10_48.5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_48.7: ref %empty_struct_type = struct_access %.loc10_48.2, element0
// CHECK:STDOUT:   %.loc10_48.8: ref %empty_struct_type = struct_access %return, element0
// CHECK:STDOUT:   %.loc10_48.9: init %empty_struct_type = struct_init () to %.loc10_48.8 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_48.10: init %empty_struct_type = converted %.loc10_48.7, %.loc10_48.9 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_48.11: init %struct_type.c.d.15a = struct_init (%.loc10_48.6, %.loc10_48.10) to %return [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc10_48.12: init %struct_type.c.d.15a = converted %empty_tuple.type.as.I.impl.F.call, %.loc10_48.11 [concrete = constants.%struct]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_48.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.db8
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc10_48.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.eab = addr_of %.loc10_48.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %.loc10_48.12 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %ptr.6db [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.84a1c2.1: type = fn_type @B.as.X.impl.F.loc14_39.1 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.b849c8.1: %B.as.X.impl.F.type.84a1c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.84a1c2.2: type = fn_type @B.as.X.impl.F.loc14_39.2 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.b849c8.2: %B.as.X.impl.F.type.84a1c2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8a8: %ptr.as.Copy.impl.Op.type.31f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e33: <witness> = impl_witness imports.%Copy.impl_witness_table.53c, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.662: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.057: %ptr.as.Copy.impl.Op.type.662 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e79, (%Copy.impl_witness.e33) [concrete]
// CHECK:STDOUT:   %.56e: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.057, @ptr.as.Copy.impl.Op(%B) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.0e4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.31f) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8a8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.53c = impl_witness_table (%Core.import_ref.0e4), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.X.impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc14_39.1: %B.as.X.impl.F.type.84a1c2.1 = fn_decl @B.as.X.impl.F.loc14_39.1 [concrete = constants.%B.as.X.impl.F.b849c8.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.5f8 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5f8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc14_8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.5f8 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.5f8 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.44a = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.44a = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc14_38: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     %self.param: %ptr.6db = value_param call_param0
// CHECK:STDOUT:     %.loc14_20: type = splice_block %ptr.loc14_20 [concrete = constants.%ptr.6db] {
// CHECK:STDOUT:       %A.ref.loc14_19: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc14_20: type = ptr_type %A.ref.loc14_19 [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.6db = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %ptr.6db = value_param call_param1
// CHECK:STDOUT:     %.loc14_31: type = splice_block %ptr.loc14_31 [concrete = constants.%ptr.6db] {
// CHECK:STDOUT:       %A.ref.loc14_30: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc14_31: type = ptr_type %A.ref.loc14_30 [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: %ptr.6db = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %ptr.019 = out_param call_param2
// CHECK:STDOUT:     %return: ref %ptr.019 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc14_39.2: %B.as.X.impl.F.type.84a1c2.2 = fn_decl @B.as.X.impl.F.loc14_39.2 [concrete = constants.%B.as.X.impl.F.b849c8.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %B.as.X.impl.F.decl.loc14_39.1
// CHECK:STDOUT:   witness = file.%X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc14_39.1(%self.param: %ptr.6db, %other.param: %ptr.6db) -> %ptr.019;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc14_39.2(%self.param: %ptr.e79, %other.param: %ptr.e79) -> %ptr.e79 [thunk @B.as.X.impl.%B.as.X.impl.F.decl.loc14_39.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %B.as.X.impl.F.type.84a1c2.1 = name_ref F, @B.as.X.impl.%B.as.X.impl.F.decl.loc14_39.1 [concrete = constants.%B.as.X.impl.F.b849c8.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.bound: <bound method> = bound_method %.loc9_17.1, %F.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.call: init %ptr.019 = call %B.as.X.impl.F.bound(%.loc9_17.4, %.loc9_31.3)
// CHECK:STDOUT:   %.loc14_39.1: %ptr.019 = value_of_initializer %B.as.X.impl.F.call
// CHECK:STDOUT:   %.loc14_39.2: %ptr.019 = converted %B.as.X.impl.F.call, %.loc14_39.1
// CHECK:STDOUT:   %.loc14_39.3: ref %C = deref %.loc14_39.2
// CHECK:STDOUT:   %.loc14_39.4: ref %B = class_element_access %.loc14_39.3, element0
// CHECK:STDOUT:   %addr.loc14: %ptr.e79 = addr_of %.loc14_39.4
// CHECK:STDOUT:   %.loc14_39.5: %ptr.e79 = converted %B.as.X.impl.F.call, %addr.loc14
// CHECK:STDOUT:   %impl.elem0: %.56e = impl_witness_access constants.%Copy.impl_witness.e33, element0 [concrete = constants.%ptr.as.Copy.impl.Op.057]
// CHECK:STDOUT:   %bound_method.loc14_39.1: <bound method> = bound_method %.loc14_39.5, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%B) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_39.2: <bound method> = bound_method %.loc14_39.5, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.e79 = call %bound_method.loc14_39.2(%.loc14_39.5)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_value_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.84a1c2.1: type = fn_type @B.as.X.impl.F.loc13_26.1 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.b849c8.1: %B.as.X.impl.F.type.84a1c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.84a1c2.2: type = fn_type @B.as.X.impl.F.loc13_26.2 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.b849c8.2: %B.as.X.impl.F.type.84a1c2.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.X.impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc13_26.1: %B.as.X.impl.F.type.84a1c2.1 = fn_decl @B.as.X.impl.F.loc13_26.1 [concrete = constants.%B.as.X.impl.F.b849c8.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c10 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c10 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.c10 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.c10 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref.loc13_14: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %A = value_param call_param1
// CHECK:STDOUT:     %A.ref.loc13_24: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %other: %A = bind_name other, %other.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc13_26.2: %B.as.X.impl.F.type.84a1c2.2 = fn_decl @B.as.X.impl.F.loc13_26.2 [concrete = constants.%B.as.X.impl.F.b849c8.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .F = %B.as.X.impl.F.decl.loc13_26.1
// CHECK:STDOUT:   witness = file.%X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc13_26.1(%self.param: %A, %other.param: %A);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc13_26.2(%self.param: %B, %other.param: %B) [thunk @B.as.X.impl.%B.as.X.impl.F.decl.loc13_26.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %B.as.X.impl.F.type.84a1c2.1 = name_ref F, @B.as.X.impl.%B.as.X.impl.F.decl.loc13_26.1 [concrete = constants.%B.as.X.impl.F.b849c8.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.call: init %empty_tuple.type = call %B.as.X.impl.F.bound(%.loc8_12.3, %.loc8_25.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_value_conversion_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %ptr.6db [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.84a1c2.1: type = fn_type @B.as.X.impl.F.loc14_39.1 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.b849c8.1: %B.as.X.impl.F.type.84a1c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.84a1c2.2: type = fn_type @B.as.X.impl.F.loc14_39.2 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.b849c8.2: %B.as.X.impl.F.type.84a1c2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8a8: %ptr.as.Copy.impl.Op.type.31f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e33: <witness> = impl_witness imports.%Copy.impl_witness_table.53c, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.662: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.057: %ptr.as.Copy.impl.Op.type.662 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e79, (%Copy.impl_witness.e33) [concrete]
// CHECK:STDOUT:   %.56e: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.057, @ptr.as.Copy.impl.Op(%B) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.0e4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.31f) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8a8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.53c = impl_witness_table (%Core.import_ref.0e4), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.X.impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc14_39.1: %B.as.X.impl.F.type.84a1c2.1 = fn_decl @B.as.X.impl.F.loc14_39.1 [concrete = constants.%B.as.X.impl.F.b849c8.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.5f8 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5f8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc14_8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.5f8 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.5f8 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.44a = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.44a = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc14_38: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     %self.param: %ptr.6db = value_param call_param0
// CHECK:STDOUT:     %.loc14_20: type = splice_block %ptr.loc14_20 [concrete = constants.%ptr.6db] {
// CHECK:STDOUT:       %A.ref.loc14_19: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc14_20: type = ptr_type %A.ref.loc14_19 [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.6db = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %ptr.6db = value_param call_param1
// CHECK:STDOUT:     %.loc14_31: type = splice_block %ptr.loc14_31 [concrete = constants.%ptr.6db] {
// CHECK:STDOUT:       %A.ref.loc14_30: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc14_31: type = ptr_type %A.ref.loc14_30 [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: %ptr.6db = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %ptr.019 = out_param call_param2
// CHECK:STDOUT:     %return: ref %ptr.019 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc14_39.2: %B.as.X.impl.F.type.84a1c2.2 = fn_decl @B.as.X.impl.F.loc14_39.2 [concrete = constants.%B.as.X.impl.F.b849c8.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %B.as.X.impl.F.decl.loc14_39.1
// CHECK:STDOUT:   witness = file.%X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc14_39.1(%self.param: %ptr.6db, %other.param: %ptr.6db) -> %ptr.019;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc14_39.2(%self.param: %ptr.e79, %other.param: %ptr.e79) -> %ptr.e79 [thunk @B.as.X.impl.%B.as.X.impl.F.decl.loc14_39.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %B.as.X.impl.F.type.84a1c2.1 = name_ref F, @B.as.X.impl.%B.as.X.impl.F.decl.loc14_39.1 [concrete = constants.%B.as.X.impl.F.b849c8.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.bound: <bound method> = bound_method %.loc9_17.1, %F.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.call: init %ptr.019 = call %B.as.X.impl.F.bound(%.loc9_17.4, %.loc9_31.3)
// CHECK:STDOUT:   %.loc14_39.1: %ptr.019 = value_of_initializer %B.as.X.impl.F.call
// CHECK:STDOUT:   %.loc14_39.2: %ptr.019 = converted %B.as.X.impl.F.call, %.loc14_39.1
// CHECK:STDOUT:   %.loc14_39.3: ref %C = deref %.loc14_39.2
// CHECK:STDOUT:   %.loc14_39.4: ref %B = class_element_access %.loc14_39.3, element0
// CHECK:STDOUT:   %addr.loc14: %ptr.e79 = addr_of %.loc14_39.4
// CHECK:STDOUT:   %.loc14_39.5: %ptr.e79 = converted %B.as.X.impl.F.call, %addr.loc14
// CHECK:STDOUT:   %impl.elem0: %.56e = impl_witness_access constants.%Copy.impl_witness.e33, element0 [concrete = constants.%ptr.as.Copy.impl.Op.057]
// CHECK:STDOUT:   %bound_method.loc14_39.1: <bound method> = bound_method %.loc14_39.5, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%B) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_39.2: <bound method> = bound_method %.loc14_39.5, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.e79 = call %bound_method.loc14_39.2(%.loc14_39.5)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inheritance_value_conversion_copy_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %A.as.X.impl.F.type.dc7bda.1: type = fn_type @A.as.X.impl.F.loc23_14.1 [concrete]
// CHECK:STDOUT:   %A.as.X.impl.F.0afbb1.1: %A.as.X.impl.F.type.dc7bda.1 = struct_value () [concrete]
// CHECK:STDOUT:   %A.as.X.impl.F.type.dc7bda.2: type = fn_type @A.as.X.impl.F.loc23_14.2 [concrete]
// CHECK:STDOUT:   %A.as.X.impl.F.0afbb1.2: %A.as.X.impl.F.type.dc7bda.2 = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %B, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.db1: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.7d9: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.db1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.X.impl: %A.ref as %X.ref {
// CHECK:STDOUT:   %A.as.X.impl.F.decl.loc23_14.1: %A.as.X.impl.F.type.dc7bda.1 = fn_decl @A.as.X.impl.F.loc23_14.1 [concrete = constants.%A.as.X.impl.F.0afbb1.1] {
// CHECK:STDOUT:     %return.patt: %pattern_type.049 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.049 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.as.X.impl.F.decl.loc23_14.2: %A.as.X.impl.F.type.dc7bda.2 = fn_decl @A.as.X.impl.F.loc23_14.2 [concrete = constants.%A.as.X.impl.F.0afbb1.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .F = %A.as.X.impl.F.decl.loc23_14.1
// CHECK:STDOUT:   witness = file.%X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.X.impl.F.loc23_14.1() -> %return.param: %B;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.X.impl.F.loc23_14.2() -> %return.param: %A [thunk @A.as.X.impl.%A.as.X.impl.F.decl.loc23_14.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %A.as.X.impl.F.type.dc7bda.1 = name_ref F, @A.as.X.impl.%A.as.X.impl.F.decl.loc23_14.1 [concrete = constants.%A.as.X.impl.F.0afbb1.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc23_14.1: ref %B = temporary_storage
// CHECK:STDOUT:   %A.as.X.impl.F.call: init %B = call %F.ref() to %.loc23_14.1
// CHECK:STDOUT:   %.loc23_14.2: ref %B = temporary %.loc23_14.1, %A.as.X.impl.F.call
// CHECK:STDOUT:   %.loc23_14.3: ref %A = class_element_access %.loc23_14.2, element0
// CHECK:STDOUT:   %.loc23_14.4: ref %A = converted %A.as.X.impl.F.call, %.loc23_14.3
// CHECK:STDOUT:   %.loc23_14.5: %A = bind_value %.loc23_14.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc23_14.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.7d9
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc23_14.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.e79 = addr_of %.loc23_14.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %U.be8: %Copy.type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.322: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %U.binding.as_type.2a0: type = symbolic_binding_type U, 0, %U.be8 [symbolic]
// CHECK:STDOUT:   %pattern_type.17e4b7.1: type = pattern_type %U.binding.as_type.2a0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.e6fdf8.1: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_34.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.f1aebb.1: %empty_tuple.type.as.I.impl.F.type.e6fdf8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.e6fdf8.2: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_34.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.f1aebb.2: %empty_tuple.type.as.I.impl.F.type.e6fdf8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1cd77c.1: <witness> = require_complete_type %U.binding.as_type.2a0 [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.e15cec.2: <witness> = lookup_impl_witness %U.be8, @Copy [symbolic]
// CHECK:STDOUT:   %.427cc3.2: type = fn_type_with_self_type %Copy.Op.type, %U.be8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.168dc4.2: %.427cc3.2 = impl_witness_access %Copy.lookup_impl_witness.e15cec.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2cedd6.2: <specific function> = specific_impl_function %impl.elem0.168dc4.2, @Copy.Op(%U.be8) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.cb2: <witness> = lookup_impl_witness %ptr.79f, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.79f, (%Copy.lookup_impl_witness.cb2) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.specific_fn: <specific function> = specific_function %empty_tuple.type.as.I.impl.F.f1aebb.1, @empty_tuple.type.as.I.impl.F.loc10_34.1(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.1cc: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.751: %.1cc = impl_witness_access %Copy.lookup_impl_witness.cb2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f44: <specific function> = specific_impl_function %impl.elem0.751, @Copy.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc8_7.2 as %I.ref {
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_34.1: %empty_tuple.type.as.I.impl.F.type.e6fdf8.1 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_34.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.f1aebb.1] {
// CHECK:STDOUT:     %U.patt: %pattern_type.322 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @empty_tuple.type.as.I.impl.F.loc10_34.1.%pattern_type (%pattern_type.17e4b7.1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @empty_tuple.type.as.I.impl.F.loc10_34.1.%pattern_type (%pattern_type.17e4b7.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @empty_tuple.type.as.I.impl.F.loc10_34.1.%pattern_type (%pattern_type.17e4b7.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @empty_tuple.type.as.I.impl.F.loc10_34.1.%pattern_type (%pattern_type.17e4b7.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc10_32: %Copy.type = name_ref U, %U.loc10_8.2 [symbolic = %U.loc10_8.1 (constants.%U.be8)]
// CHECK:STDOUT:     %U.as_type.loc10_32: type = facet_access_type %U.ref.loc10_32 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)]
// CHECK:STDOUT:     %.loc10_32: type = converted %U.ref.loc10_32, %U.as_type.loc10_32 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)]
// CHECK:STDOUT:     %.loc10_16: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc10_8.2: %Copy.type = bind_symbolic_name U, 0 [symbolic = %U.loc10_8.1 (constants.%U.be8)]
// CHECK:STDOUT:     %x.param: @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.2a0) = value_param call_param0
// CHECK:STDOUT:     %.loc10_26.1: type = splice_block %.loc10_26.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)] {
// CHECK:STDOUT:       %U.ref.loc10_26: %Copy.type = name_ref U, %U.loc10_8.2 [symbolic = %U.loc10_8.1 (constants.%U.be8)]
// CHECK:STDOUT:       %U.as_type.loc10_26: type = facet_access_type %U.ref.loc10_26 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)]
// CHECK:STDOUT:       %.loc10_26.2: type = converted %U.ref.loc10_26, %U.as_type.loc10_26 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.2a0) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.2a0) = out_param call_param1
// CHECK:STDOUT:     %return: ref @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.2a0) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_34.2: %empty_tuple.type.as.I.impl.F.type.e6fdf8.2 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_34.2 [concrete = constants.%empty_tuple.type.as.I.impl.F.f1aebb.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl.loc10_34.1
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_tuple.type.as.I.impl.F.loc10_34.1(%U.loc10_8.2: %Copy.type) {
// CHECK:STDOUT:   %U.loc10_8.1: %Copy.type = bind_symbolic_name U, 0 [symbolic = %U.loc10_8.1 (constants.%U.be8)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U.loc10_8.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.17e4b7.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete (constants.%require_complete.1cd77c.1)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc10_8.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.e15cec.2)]
// CHECK:STDOUT:   %.loc10_43.2: type = fn_type_with_self_type constants.%Copy.Op.type, %U.loc10_8.1 [symbolic = %.loc10_43.2 (constants.%.427cc3.2)]
// CHECK:STDOUT:   %impl.elem0.loc10_43.2: @empty_tuple.type.as.I.impl.F.loc10_34.1.%.loc10_43.2 (%.427cc3.2) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_43.2 (constants.%impl.elem0.168dc4.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_43.2: <specific function> = specific_impl_function %impl.elem0.loc10_43.2, @Copy.Op(%U.loc10_8.1) [symbolic = %specific_impl_fn.loc10_43.2 (constants.%specific_impl_fn.2cedd6.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.2a0)) -> %return.param: @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.2a0) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.2a0) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc10_43.1: @empty_tuple.type.as.I.impl.F.loc10_34.1.%.loc10_43.2 (%.427cc3.2) = impl_witness_access constants.%Copy.lookup_impl_witness.e15cec.2, element0 [symbolic = %impl.elem0.loc10_43.2 (constants.%impl.elem0.168dc4.2)]
// CHECK:STDOUT:     %bound_method.loc10_43.1: <bound method> = bound_method %x.ref, %impl.elem0.loc10_43.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_43.1: <specific function> = specific_impl_function %impl.elem0.loc10_43.1, @Copy.Op(constants.%U.be8) [symbolic = %specific_impl_fn.loc10_43.2 (constants.%specific_impl_fn.2cedd6.2)]
// CHECK:STDOUT:     %bound_method.loc10_43.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_43.1
// CHECK:STDOUT:     %.loc10_29: ref @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.2a0) = splice_block %return {}
// CHECK:STDOUT:     %.loc10_43.1: init @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.2a0) = call %bound_method.loc10_43.2(%x.ref) to %.loc10_29
// CHECK:STDOUT:     return %.loc10_43.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_tuple.type.as.I.impl.F.loc10_34.2(%T.loc5_8.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.cb2)]
// CHECK:STDOUT:   %Copy.facet.loc10_34.3: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc10_34.3 (constants.%Copy.facet)]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.specific_fn.loc10_34.2: <specific function> = specific_function constants.%empty_tuple.type.as.I.impl.F.f1aebb.1, @empty_tuple.type.as.I.impl.F.loc10_34.1(%Copy.facet.loc10_34.3) [symbolic = %empty_tuple.type.as.I.impl.F.specific_fn.loc10_34.2 (constants.%empty_tuple.type.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @empty_tuple.type.as.I.impl.F.loc10_34.2.%ptr (%ptr.79f)) -> @empty_tuple.type.as.I.impl.F.loc10_34.2.%ptr (%ptr.79f) [thunk @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_34.1] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %empty_tuple.type.as.I.impl.F.type.e6fdf8.1 = name_ref F, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_34.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.f1aebb.1]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Copy.facet.loc10_34.1: %Copy.type = facet_value constants.%ptr.79f, (constants.%Copy.lookup_impl_witness.cb2) [symbolic = %Copy.facet.loc10_34.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc10_34.1: %Copy.type = converted constants.%ptr.79f, %Copy.facet.loc10_34.1 [symbolic = %Copy.facet.loc10_34.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %Copy.facet.loc10_34.2: %Copy.type = facet_value constants.%ptr.79f, (constants.%Copy.lookup_impl_witness.cb2) [symbolic = %Copy.facet.loc10_34.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc10_34.2: %Copy.type = converted constants.%ptr.79f, %Copy.facet.loc10_34.2 [symbolic = %Copy.facet.loc10_34.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.specific_fn.loc10_34.1: <specific function> = specific_function %F.ref, @empty_tuple.type.as.I.impl.F.loc10_34.1(constants.%Copy.facet) [symbolic = %empty_tuple.type.as.I.impl.F.specific_fn.loc10_34.2 (constants.%empty_tuple.type.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.call: init @empty_tuple.type.as.I.impl.F.loc10_34.2.%ptr (%ptr.79f) = call %empty_tuple.type.as.I.impl.F.specific_fn.loc10_34.1(%x.ref)
// CHECK:STDOUT:     return %empty_tuple.type.as.I.impl.F.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc10_34.1(constants.%U.be8) {
// CHECK:STDOUT:   %U.loc10_8.1 => constants.%U.be8
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type.2a0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.17e4b7.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc10_34.2(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc5_8.1 => constants.%T.8b3
// CHECK:STDOUT:   %ptr => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.afe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc10_34.1(constants.%Copy.facet) {
// CHECK:STDOUT:   %U.loc10_8.1 => constants.%Copy.facet
// CHECK:STDOUT:   %U.binding.as_type => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.afe
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6e5
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.cb2
// CHECK:STDOUT:   %.loc10_43.2 => constants.%.1cc
// CHECK:STDOUT:   %impl.elem0.loc10_43.2 => constants.%impl.elem0.751
// CHECK:STDOUT:   %specific_impl_fn.loc10_43.2 => constants.%specific_impl_fn.f44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_generic_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %U.be8: %Copy.type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.322: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %U.binding.as_type.2a0: type = symbolic_binding_type U, 0, %U.be8 [symbolic]
// CHECK:STDOUT:   %pattern_type.17e4b7.1: type = pattern_type %U.binding.as_type.2a0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.e6fdf8.1: type = fn_type @empty_tuple.type.as.I.impl.F.loc19_44.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.f1aebb.1: %empty_tuple.type.as.I.impl.F.type.e6fdf8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.e6fdf8.2: type = fn_type @empty_tuple.type.as.I.impl.F.loc19_44.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.f1aebb.2: %empty_tuple.type.as.I.impl.F.type.e6fdf8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1cd77c.1: <witness> = require_complete_type %U.binding.as_type.2a0 [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.e15cec.2: <witness> = lookup_impl_witness %U.be8, @Copy [symbolic]
// CHECK:STDOUT:   %.427cc3.2: type = fn_type_with_self_type %Copy.Op.type, %U.be8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.168dc4.2: %.427cc3.2 = impl_witness_access %Copy.lookup_impl_witness.e15cec.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2cedd6.2: <specific function> = specific_impl_function %impl.elem0.168dc4.2, @Copy.Op(%U.be8) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc10_7.2 as %I.ref {
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc19_44.1: %empty_tuple.type.as.I.impl.F.type.e6fdf8.1 = fn_decl @empty_tuple.type.as.I.impl.F.loc19_44.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.f1aebb.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.322 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @empty_tuple.type.as.I.impl.F.loc19_44.1.%pattern_type (%pattern_type.17e4b7.1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @empty_tuple.type.as.I.impl.F.loc19_44.1.%pattern_type (%pattern_type.17e4b7.1) = value_param_pattern %x.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @empty_tuple.type.as.I.impl.F.loc19_44.1.%pattern_type (%pattern_type.17e4b7.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @empty_tuple.type.as.I.impl.F.loc19_44.1.%pattern_type (%pattern_type.17e4b7.1) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc19_42: %Copy.type = name_ref U, %U.loc19_18.2 [symbolic = %U.loc19_18.1 (constants.%U.be8)]
// CHECK:STDOUT:     %U.as_type.loc19_42: type = facet_access_type %U.ref.loc19_42 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)]
// CHECK:STDOUT:     %.loc19_42: type = converted %U.ref.loc19_42, %U.as_type.loc19_42 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc19_15.1: type = splice_block %.loc19_15.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc19_15.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc19_15.3: type = converted %.loc19_15.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %.loc19_26: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc19_18.2: %Copy.type = bind_symbolic_name U, 0 [symbolic = %U.loc19_18.1 (constants.%U.be8)]
// CHECK:STDOUT:     %x.param: @empty_tuple.type.as.I.impl.F.loc19_44.1.%U.binding.as_type (%U.binding.as_type.2a0) = value_param call_param1
// CHECK:STDOUT:     %.loc19_36.1: type = splice_block %.loc19_36.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)] {
// CHECK:STDOUT:       %U.ref.loc19_36: %Copy.type = name_ref U, %U.loc19_18.2 [symbolic = %U.loc19_18.1 (constants.%U.be8)]
// CHECK:STDOUT:       %U.as_type.loc19_36: type = facet_access_type %U.ref.loc19_36 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)]
// CHECK:STDOUT:       %.loc19_36.2: type = converted %U.ref.loc19_36, %U.as_type.loc19_36 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @empty_tuple.type.as.I.impl.F.loc19_44.1.%U.binding.as_type (%U.binding.as_type.2a0) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @empty_tuple.type.as.I.impl.F.loc19_44.1.%U.binding.as_type (%U.binding.as_type.2a0) = out_param call_param2
// CHECK:STDOUT:     %return: ref @empty_tuple.type.as.I.impl.F.loc19_44.1.%U.binding.as_type (%U.binding.as_type.2a0) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc19_44.2: %empty_tuple.type.as.I.impl.F.type.e6fdf8.2 = fn_decl @empty_tuple.type.as.I.impl.F.loc19_44.2 [concrete = constants.%empty_tuple.type.as.I.impl.F.f1aebb.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl.loc19_44.1
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_tuple.type.as.I.impl.F.loc19_44.1(%U.loc19_18.2: %Copy.type) {
// CHECK:STDOUT:   %U.loc19_18.1: %Copy.type = bind_symbolic_name U, 0 [symbolic = %U.loc19_18.1 (constants.%U.be8)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U.loc19_18.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.2a0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.17e4b7.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete (constants.%require_complete.1cd77c.1)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc19_18.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.e15cec.2)]
// CHECK:STDOUT:   %.loc19_53.2: type = fn_type_with_self_type constants.%Copy.Op.type, %U.loc19_18.1 [symbolic = %.loc19_53.2 (constants.%.427cc3.2)]
// CHECK:STDOUT:   %impl.elem0.loc19_53.2: @empty_tuple.type.as.I.impl.F.loc19_44.1.%.loc19_53.2 (%.427cc3.2) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc19_53.2 (constants.%impl.elem0.168dc4.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc19_53.2: <specific function> = specific_impl_function %impl.elem0.loc19_53.2, @Copy.Op(%U.loc19_18.1) [symbolic = %specific_impl_fn.loc19_53.2 (constants.%specific_impl_fn.2cedd6.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: %empty_tuple.type, %x.param: @empty_tuple.type.as.I.impl.F.loc19_44.1.%U.binding.as_type (%U.binding.as_type.2a0)) -> %return.param: @empty_tuple.type.as.I.impl.F.loc19_44.1.%U.binding.as_type (%U.binding.as_type.2a0) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @empty_tuple.type.as.I.impl.F.loc19_44.1.%U.binding.as_type (%U.binding.as_type.2a0) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc19_53.1: @empty_tuple.type.as.I.impl.F.loc19_44.1.%.loc19_53.2 (%.427cc3.2) = impl_witness_access constants.%Copy.lookup_impl_witness.e15cec.2, element0 [symbolic = %impl.elem0.loc19_53.2 (constants.%impl.elem0.168dc4.2)]
// CHECK:STDOUT:     %bound_method.loc19_53.1: <bound method> = bound_method %x.ref, %impl.elem0.loc19_53.1
// CHECK:STDOUT:     %specific_impl_fn.loc19_53.1: <specific function> = specific_impl_function %impl.elem0.loc19_53.1, @Copy.Op(constants.%U.be8) [symbolic = %specific_impl_fn.loc19_53.2 (constants.%specific_impl_fn.2cedd6.2)]
// CHECK:STDOUT:     %bound_method.loc19_53.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc19_53.1
// CHECK:STDOUT:     %.loc19_39: ref @empty_tuple.type.as.I.impl.F.loc19_44.1.%U.binding.as_type (%U.binding.as_type.2a0) = splice_block %return {}
// CHECK:STDOUT:     %.loc19_53.1: init @empty_tuple.type.as.I.impl.F.loc19_44.1.%U.binding.as_type (%U.binding.as_type.2a0) = call %bound_method.loc19_53.2(%x.ref) to %.loc19_39
// CHECK:STDOUT:     return %.loc19_53.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_tuple.type.as.I.impl.F.loc19_44.2(%T.loc5_20.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: %empty_tuple.type, %x.param: @empty_tuple.type.as.I.impl.F.loc19_44.2.%ptr (%ptr.79f)) -> @empty_tuple.type.as.I.impl.F.loc19_44.2.%ptr (%ptr.79f) [thunk @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc19_44.1] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %empty_tuple.type.as.I.impl.F.type.e6fdf8.1 = name_ref F, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc19_44.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.f1aebb.1]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     return <error> to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc19_44.1(constants.%U.be8) {
// CHECK:STDOUT:   %U.loc19_18.1 => constants.%U.be8
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type.2a0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.17e4b7.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc19_44.2(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc5_20.1 => constants.%T.8b3
// CHECK:STDOUT:   %ptr => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.afe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.b.a.1b0: type = struct_type {.b: %empty_struct_type, .a: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %pattern_type.914: type = pattern_type %struct_type.b.a.1b0 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.109800.1: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_29.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.3e8bab.1: %empty_tuple.type.as.I.impl.F.type.109800.1 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.f95: type = struct_type {.a: %empty_struct_type, .b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.109800.2: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_29.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.3e8bab.2: %empty_tuple.type.as.I.impl.F.type.109800.2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.f95 = struct_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %struct_type.b.a.1b0, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.6b8: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.637: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.6b8 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.30d: type = ptr_type %struct_type.b.a.1b0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc8_7.2 as %I.type {
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_29.1: %empty_tuple.type.as.I.impl.F.type.109800.1 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_29.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.3e8bab.1] {
// CHECK:STDOUT:     %return.patt: %pattern_type.914 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.914 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_19.2: type = converted %.loc10_19.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc10_27.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_27.2: type = converted %.loc10_27.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.b.a: type = struct_type {.b: %empty_struct_type, .a: %empty_struct_type} [concrete = constants.%struct_type.b.a.1b0]
// CHECK:STDOUT:     %return.param: ref %struct_type.b.a.1b0 = out_param call_param0
// CHECK:STDOUT:     %return: ref %struct_type.b.a.1b0 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_29.2: %empty_tuple.type.as.I.impl.F.type.109800.2 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_29.2 [concrete = constants.%empty_tuple.type.as.I.impl.F.3e8bab.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl.loc10_29.1
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F.loc10_29.1() -> %return.param: %struct_type.b.a.1b0;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F.loc10_29.2() -> %return.param: %struct_type.a.b.f95 [thunk @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_29.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %empty_tuple.type.as.I.impl.F.type.109800.1 = name_ref F, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_29.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.3e8bab.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_29.1: ref %struct_type.b.a.1b0 = temporary_storage
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.call: init %struct_type.b.a.1b0 = call %F.ref() to %.loc10_29.1
// CHECK:STDOUT:   %.loc10_29.2: ref %struct_type.b.a.1b0 = temporary %.loc10_29.1, %empty_tuple.type.as.I.impl.F.call
// CHECK:STDOUT:   %.loc10_29.3: ref %empty_struct_type = struct_access %.loc10_29.2, element1
// CHECK:STDOUT:   %.loc10_29.4: ref %empty_struct_type = struct_access %return, element1
// CHECK:STDOUT:   %.loc10_29.5: init %empty_struct_type = struct_init () to %.loc10_29.4 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_29.6: init %empty_struct_type = converted %.loc10_29.3, %.loc10_29.5 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_29.7: ref %empty_struct_type = struct_access %.loc10_29.2, element0
// CHECK:STDOUT:   %.loc10_29.8: ref %empty_struct_type = struct_access %return, element0
// CHECK:STDOUT:   %.loc10_29.9: init %empty_struct_type = struct_init () to %.loc10_29.8 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_29.10: init %empty_struct_type = converted %.loc10_29.7, %.loc10_29.9 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_29.11: init %struct_type.a.b.f95 = struct_init (%.loc10_29.6, %.loc10_29.10) to %return [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc10_29.12: init %struct_type.a.b.f95 = converted %empty_tuple.type.as.I.impl.F.call, %.loc10_29.11 [concrete = constants.%struct]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc10_29.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.637
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc10_29.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.30d = addr_of %.loc10_29.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %.loc10_29.12 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
