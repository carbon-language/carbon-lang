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
  fn F[ref self: Self](other: Self*) -> Self*;
}

impl B as X {
  //@dump-sem-ir-begin
  fn F[ref self: A](other: A*) -> C*;
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
  fn F[ref self: Self](other: Self*) -> Self*;
}

impl B as X {
  //@dump-sem-ir-begin
  fn F[ref self: A](other: A*) -> C*;
  //@dump-sem-ir-end
}

// --- inheritance_value_conversion_pointer_addr.carbon

library "[[@TEST_NAME]]";

base class A {}
base class B { extend base: A; }
class C { extend base: B; }

interface X {
  fn F[ref self: Self](other: Self*) -> Self*;
}

impl B as X {
  //@dump-sem-ir-begin
  fn F[ref self: A](other: A*) -> C*;
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

// --- generic_method.carbon

library "[[@TEST_NAME]]";

interface I {
  fn F[self: Self, T:! type](x: T*) -> T*;
}

impl () as I {
  //@dump-sem-ir-begin
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
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.391: type = struct_type {.a: %empty_tuple.type, .b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct_type.c.d.15a: type = struct_type {.c: %empty_tuple.type, .d: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct_type.b.a.40c: type = struct_type {.b: %empty_struct_type, .a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.231: type = pattern_type %struct_type.b.a.40c [concrete]
// CHECK:STDOUT:   %struct_type.d.c.b36: type = struct_type {.d: %empty_struct_type, .c: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %.b7f: Core.Form = init_form %struct_type.d.c.b36, call_param1 [concrete]
// CHECK:STDOUT:   %pattern_type.844: type = pattern_type %struct_type.d.c.b36 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.19e4b8.1: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_48.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.8be29b.1: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.19e4b8.2: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_48.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.8be29b.2: %empty_tuple.type.as.I.impl.F.type.19e4b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.c.d.15a = struct_value (%empty_tuple, %empty_struct) [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc8_7.2 as %I.ref {
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_48.1: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_48.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.1] {
// CHECK:STDOUT:     %y.patt: %pattern_type.231 = value_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.231 = value_param_pattern %y.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.844 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.844 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_38.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc10_38.2: type = converted %.loc10_38.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc10_46.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc10_46.2: type = converted %.loc10_46.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.d.c: type = struct_type {.d: %empty_struct_type, .c: %empty_tuple.type} [concrete = constants.%struct_type.d.c.b36]
// CHECK:STDOUT:     %.loc10_47: Core.Form = init_form %struct_type.d.c, call_param1 [concrete = constants.%.b7f]
// CHECK:STDOUT:     %y.param: %struct_type.b.a.40c = value_param call_param0
// CHECK:STDOUT:     %.loc10_26: type = splice_block %struct_type.b.a [concrete = constants.%struct_type.b.a.40c] {
// CHECK:STDOUT:       %.loc10_17.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc10_17.2: type = converted %.loc10_17.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc10_25.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc10_25.2: type = converted %.loc10_25.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.b.a: type = struct_type {.b: %empty_struct_type, .a: %empty_tuple.type} [concrete = constants.%struct_type.b.a.40c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: %struct_type.b.a.40c = value_binding y, %y.param
// CHECK:STDOUT:     %return.param: ref %struct_type.d.c.b36 = out_param call_param1
// CHECK:STDOUT:     %return: ref %struct_type.d.c.b36 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_48.2: %empty_tuple.type.as.I.impl.F.type.19e4b8.2 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_48.2 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl.loc10_48.1
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F.loc10_48.1(%y.param: %struct_type.b.a.40c) -> out %return.param: %struct_type.d.c.b36;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F.loc10_48.2(%x.param: %struct_type.a.b.391) -> out %return.param: %struct_type.c.d.15a [thunk @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_48.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = name_ref F, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_48.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_48.1: ref %struct_type.d.c.b36 = temporary_storage
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.call: init %struct_type.d.c.b36 to %.loc10_48.1 = call %F.ref(%.loc5_9.3)
// CHECK:STDOUT:   %.loc10_48.2: ref %struct_type.d.c.b36 = temporary %.loc10_48.1, %empty_tuple.type.as.I.impl.F.call
// CHECK:STDOUT:   %.loc10_48.3: ref %empty_tuple.type = struct_access %.loc10_48.2, element1
// CHECK:STDOUT:   %.loc10_48.4: ref %empty_tuple.type = struct_access %return.param, element1
// CHECK:STDOUT:   %.loc10_48.5: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_48.6: init %empty_tuple.type = converted %.loc10_48.3, %.loc10_48.5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_48.7: ref %empty_struct_type = struct_access %.loc10_48.2, element0
// CHECK:STDOUT:   %.loc10_48.8: ref %empty_struct_type = struct_access %return.param, element0
// CHECK:STDOUT:   %.loc10_48.9: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_48.10: init %empty_struct_type = converted %.loc10_48.7, %.loc10_48.9 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_48.11: init %struct_type.c.d.15a to %return.param = struct_init (%.loc10_48.6, %.loc10_48.10) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc10_48.12: init %struct_type.c.d.15a = converted %empty_tuple.type.as.I.impl.F.call, %.loc10_48.11 [concrete = constants.%struct]
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %.loc10_48.2, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%.loc10_48.2)
// CHECK:STDOUT:   return %.loc10_48.12 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %struct_type.d.c.b36) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.1ab: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %ptr.643: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.f29: type = pattern_type %ptr.643 [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.c5e: Core.Form = init_form %ptr.31e, call_param2 [concrete]
// CHECK:STDOUT:   %pattern_type.506: type = pattern_type %ptr.31e [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.421a66.1: type = fn_type @B.as.X.impl.F.loc14_37.1 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.8ec460.1: %B.as.X.impl.F.type.421a66.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.27c: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.421a66.2: type = fn_type @B.as.X.impl.F.loc14_37.2 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.8ec460.2: %B.as.X.impl.F.type.421a66.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.672: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.0fc: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.20b: %ptr.as.Copy.impl.Op.type.0fc = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.27c, (%Copy.impl_witness.672) [concrete]
// CHECK:STDOUT:   %.116: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.20b, @ptr.as.Copy.impl.Op(%B) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.X.impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc14_37.1: %B.as.X.impl.F.type.421a66.1 = fn_decl @B.as.X.impl.F.loc14_37.1 [concrete = constants.%B.as.X.impl.F.8ec460.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1ab = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1ab = ref_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.f29 = value_binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.f29 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.506 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.506 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc14_36: type = ptr_type %C.ref [concrete = constants.%ptr.31e]
// CHECK:STDOUT:     %.loc14_36: Core.Form = init_form %ptr.loc14_36, call_param2 [concrete = constants.%.c5e]
// CHECK:STDOUT:     %self.param: ref %A = ref_param call_param0
// CHECK:STDOUT:     %A.ref.loc14_18: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %self: ref %A = ref_binding self, %self.param
// CHECK:STDOUT:     %other.param: %ptr.643 = value_param call_param1
// CHECK:STDOUT:     %.loc14_29: type = splice_block %ptr.loc14_29 [concrete = constants.%ptr.643] {
// CHECK:STDOUT:       %A.ref.loc14_28: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc14_29: type = ptr_type %A.ref.loc14_28 [concrete = constants.%ptr.643]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: %ptr.643 = value_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref %ptr.31e = out_param call_param2
// CHECK:STDOUT:     %return: ref %ptr.31e = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc14_37.2: %B.as.X.impl.F.type.421a66.2 = fn_decl @B.as.X.impl.F.loc14_37.2 [concrete = constants.%B.as.X.impl.F.8ec460.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %B.as.X.impl.F.decl.loc14_37.1
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc14_37.1(%self.param: %A, %other.param: %ptr.643) -> out %return.param: %ptr.31e;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc14_37.2(%self.param: %B, %other.param: %ptr.27c) -> out %return.param: %ptr.27c [thunk @B.as.X.impl.%B.as.X.impl.F.decl.loc14_37.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %B.as.X.impl.F.type.421a66.1 = name_ref F, @B.as.X.impl.%B.as.X.impl.F.decl.loc14_37.1 [concrete = constants.%B.as.X.impl.F.8ec460.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.call: init %ptr.31e = call %B.as.X.impl.F.bound(%.loc9_16.2, %.loc9_29.3)
// CHECK:STDOUT:   %.loc14_37.1: %ptr.31e = value_of_initializer %B.as.X.impl.F.call
// CHECK:STDOUT:   %.loc14_37.2: %ptr.31e = converted %B.as.X.impl.F.call, %.loc14_37.1
// CHECK:STDOUT:   %.loc14_37.3: ref %C = deref %.loc14_37.2
// CHECK:STDOUT:   %.loc14_37.4: ref %B = class_element_access %.loc14_37.3, element0
// CHECK:STDOUT:   %addr.loc14: %ptr.27c = addr_of %.loc14_37.4
// CHECK:STDOUT:   %.loc14_37.5: %ptr.27c = converted %B.as.X.impl.F.call, %addr.loc14
// CHECK:STDOUT:   %impl.elem0: %.116 = impl_witness_access constants.%Copy.impl_witness.672, element0 [concrete = constants.%ptr.as.Copy.impl.Op.20b]
// CHECK:STDOUT:   %bound_method.loc14_37.1: <bound method> = bound_method %.loc14_37.5, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%B) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_37.2: <bound method> = bound_method %.loc14_37.5, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.27c = call %bound_method.loc14_37.2(%.loc14_37.5)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_value_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.1ab: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.421a66.1: type = fn_type @B.as.X.impl.F.loc13_26.1 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.8ec460.1: %B.as.X.impl.F.type.421a66.1 = struct_value () [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.421a66.2: type = fn_type @B.as.X.impl.F.loc13_26.2 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.8ec460.2: %B.as.X.impl.F.type.421a66.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.X.impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc13_26.1: %B.as.X.impl.F.type.421a66.1 = fn_decl @B.as.X.impl.F.loc13_26.1 [concrete = constants.%B.as.X.impl.F.8ec460.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1ab = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1ab = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.1ab = value_binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.1ab = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref.loc13_14: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = value_binding self, %self.param
// CHECK:STDOUT:     %other.param: %A = value_param call_param1
// CHECK:STDOUT:     %A.ref.loc13_24: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %other: %A = value_binding other, %other.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc13_26.2: %B.as.X.impl.F.type.421a66.2 = fn_decl @B.as.X.impl.F.loc13_26.2 [concrete = constants.%B.as.X.impl.F.8ec460.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .F = %B.as.X.impl.F.decl.loc13_26.1
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc13_26.1(%self.param: %A, %other.param: %A);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc13_26.2(%self.param: %B, %other.param: %B) [thunk @B.as.X.impl.%B.as.X.impl.F.decl.loc13_26.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %B.as.X.impl.F.type.421a66.1 = name_ref F, @B.as.X.impl.%B.as.X.impl.F.decl.loc13_26.1 [concrete = constants.%B.as.X.impl.F.8ec460.1]
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
// CHECK:STDOUT:   %pattern_type.1ab: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %ptr.643: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.f29: type = pattern_type %ptr.643 [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.c5e: Core.Form = init_form %ptr.31e, call_param2 [concrete]
// CHECK:STDOUT:   %pattern_type.506: type = pattern_type %ptr.31e [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.421a66.1: type = fn_type @B.as.X.impl.F.loc14_37.1 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.8ec460.1: %B.as.X.impl.F.type.421a66.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.27c: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.421a66.2: type = fn_type @B.as.X.impl.F.loc14_37.2 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.8ec460.2: %B.as.X.impl.F.type.421a66.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.672: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.0fc: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.20b: %ptr.as.Copy.impl.Op.type.0fc = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.27c, (%Copy.impl_witness.672) [concrete]
// CHECK:STDOUT:   %.116: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.20b, @ptr.as.Copy.impl.Op(%B) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.X.impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc14_37.1: %B.as.X.impl.F.type.421a66.1 = fn_decl @B.as.X.impl.F.loc14_37.1 [concrete = constants.%B.as.X.impl.F.8ec460.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1ab = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1ab = ref_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.f29 = value_binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.f29 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.506 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.506 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc14_36: type = ptr_type %C.ref [concrete = constants.%ptr.31e]
// CHECK:STDOUT:     %.loc14_36: Core.Form = init_form %ptr.loc14_36, call_param2 [concrete = constants.%.c5e]
// CHECK:STDOUT:     %self.param: ref %A = ref_param call_param0
// CHECK:STDOUT:     %A.ref.loc14_18: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %self: ref %A = ref_binding self, %self.param
// CHECK:STDOUT:     %other.param: %ptr.643 = value_param call_param1
// CHECK:STDOUT:     %.loc14_29: type = splice_block %ptr.loc14_29 [concrete = constants.%ptr.643] {
// CHECK:STDOUT:       %A.ref.loc14_28: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc14_29: type = ptr_type %A.ref.loc14_28 [concrete = constants.%ptr.643]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: %ptr.643 = value_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref %ptr.31e = out_param call_param2
// CHECK:STDOUT:     %return: ref %ptr.31e = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc14_37.2: %B.as.X.impl.F.type.421a66.2 = fn_decl @B.as.X.impl.F.loc14_37.2 [concrete = constants.%B.as.X.impl.F.8ec460.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %B.as.X.impl.F.decl.loc14_37.1
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc14_37.1(%self.param: %A, %other.param: %ptr.643) -> out %return.param: %ptr.31e;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc14_37.2(%self.param: %B, %other.param: %ptr.27c) -> out %return.param: %ptr.27c [thunk @B.as.X.impl.%B.as.X.impl.F.decl.loc14_37.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %B.as.X.impl.F.type.421a66.1 = name_ref F, @B.as.X.impl.%B.as.X.impl.F.decl.loc14_37.1 [concrete = constants.%B.as.X.impl.F.8ec460.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.call: init %ptr.31e = call %B.as.X.impl.F.bound(%.loc9_16.2, %.loc9_29.3)
// CHECK:STDOUT:   %.loc14_37.1: %ptr.31e = value_of_initializer %B.as.X.impl.F.call
// CHECK:STDOUT:   %.loc14_37.2: %ptr.31e = converted %B.as.X.impl.F.call, %.loc14_37.1
// CHECK:STDOUT:   %.loc14_37.3: ref %C = deref %.loc14_37.2
// CHECK:STDOUT:   %.loc14_37.4: ref %B = class_element_access %.loc14_37.3, element0
// CHECK:STDOUT:   %addr.loc14: %ptr.27c = addr_of %.loc14_37.4
// CHECK:STDOUT:   %.loc14_37.5: %ptr.27c = converted %B.as.X.impl.F.call, %addr.loc14
// CHECK:STDOUT:   %impl.elem0: %.116 = impl_witness_access constants.%Copy.impl_witness.672, element0 [concrete = constants.%ptr.as.Copy.impl.Op.20b]
// CHECK:STDOUT:   %bound_method.loc14_37.1: <bound method> = bound_method %.loc14_37.5, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%B) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_37.2: <bound method> = bound_method %.loc14_37.5, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.27c = call %bound_method.loc14_37.2(%.loc14_37.5)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_value_conversion_pointer_addr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.1ab: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %ptr.643: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.f29: type = pattern_type %ptr.643 [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.c5e: Core.Form = init_form %ptr.31e, call_param2 [concrete]
// CHECK:STDOUT:   %pattern_type.506: type = pattern_type %ptr.31e [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.421a66.1: type = fn_type @B.as.X.impl.F.loc14_37.1 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.8ec460.1: %B.as.X.impl.F.type.421a66.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.27c: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.421a66.2: type = fn_type @B.as.X.impl.F.loc14_37.2 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.8ec460.2: %B.as.X.impl.F.type.421a66.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.672: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.0fc: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.20b: %ptr.as.Copy.impl.Op.type.0fc = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.27c, (%Copy.impl_witness.672) [concrete]
// CHECK:STDOUT:   %.116: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.20b, @ptr.as.Copy.impl.Op(%B) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.X.impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc14_37.1: %B.as.X.impl.F.type.421a66.1 = fn_decl @B.as.X.impl.F.loc14_37.1 [concrete = constants.%B.as.X.impl.F.8ec460.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1ab = ref_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1ab = ref_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.f29 = value_binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.f29 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.506 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.506 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc14_36: type = ptr_type %C.ref [concrete = constants.%ptr.31e]
// CHECK:STDOUT:     %.loc14_36: Core.Form = init_form %ptr.loc14_36, call_param2 [concrete = constants.%.c5e]
// CHECK:STDOUT:     %self.param: ref %A = ref_param call_param0
// CHECK:STDOUT:     %A.ref.loc14_18: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %self: ref %A = ref_binding self, %self.param
// CHECK:STDOUT:     %other.param: %ptr.643 = value_param call_param1
// CHECK:STDOUT:     %.loc14_29: type = splice_block %ptr.loc14_29 [concrete = constants.%ptr.643] {
// CHECK:STDOUT:       %A.ref.loc14_28: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc14_29: type = ptr_type %A.ref.loc14_28 [concrete = constants.%ptr.643]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: %ptr.643 = value_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref %ptr.31e = out_param call_param2
// CHECK:STDOUT:     %return: ref %ptr.31e = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc14_37.2: %B.as.X.impl.F.type.421a66.2 = fn_decl @B.as.X.impl.F.loc14_37.2 [concrete = constants.%B.as.X.impl.F.8ec460.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %B.as.X.impl.F.decl.loc14_37.1
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc14_37.1(%self.param: %A, %other.param: %ptr.643) -> out %return.param: %ptr.31e;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc14_37.2(%self.param: %B, %other.param: %ptr.27c) -> out %return.param: %ptr.27c [thunk @B.as.X.impl.%B.as.X.impl.F.decl.loc14_37.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %B.as.X.impl.F.type.421a66.1 = name_ref F, @B.as.X.impl.%B.as.X.impl.F.decl.loc14_37.1 [concrete = constants.%B.as.X.impl.F.8ec460.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.call: init %ptr.31e = call %B.as.X.impl.F.bound(%.loc9_16.2, %.loc9_29.3)
// CHECK:STDOUT:   %.loc14_37.1: %ptr.31e = value_of_initializer %B.as.X.impl.F.call
// CHECK:STDOUT:   %.loc14_37.2: %ptr.31e = converted %B.as.X.impl.F.call, %.loc14_37.1
// CHECK:STDOUT:   %.loc14_37.3: ref %C = deref %.loc14_37.2
// CHECK:STDOUT:   %.loc14_37.4: ref %B = class_element_access %.loc14_37.3, element0
// CHECK:STDOUT:   %addr.loc14: %ptr.27c = addr_of %.loc14_37.4
// CHECK:STDOUT:   %.loc14_37.5: %ptr.27c = converted %B.as.X.impl.F.call, %addr.loc14
// CHECK:STDOUT:   %impl.elem0: %.116 = impl_witness_access constants.%Copy.impl_witness.672, element0 [concrete = constants.%ptr.as.Copy.impl.Op.20b]
// CHECK:STDOUT:   %bound_method.loc14_37.1: <bound method> = bound_method %.loc14_37.5, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%B) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_37.2: <bound method> = bound_method %.loc14_37.5, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.27c = call %bound_method.loc14_37.2(%.loc14_37.5)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inheritance_value_conversion_copy_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.e82: Core.Form = init_form %B, call_param0 [concrete]
// CHECK:STDOUT:   %pattern_type.1f4: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %A.as.X.impl.F.type.170a02.1: type = fn_type @A.as.X.impl.F.loc23_14.1 [concrete]
// CHECK:STDOUT:   %A.as.X.impl.F.e6cf46.1: %A.as.X.impl.F.type.170a02.1 = struct_value () [concrete]
// CHECK:STDOUT:   %A.as.X.impl.F.type.170a02.2: type = fn_type @A.as.X.impl.F.loc23_14.2 [concrete]
// CHECK:STDOUT:   %A.as.X.impl.F.e6cf46.2: %A.as.X.impl.F.type.170a02.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.X.impl: %A.ref as %X.ref {
// CHECK:STDOUT:   %A.as.X.impl.F.decl.loc23_14.1: %A.as.X.impl.F.type.170a02.1 = fn_decl @A.as.X.impl.F.loc23_14.1 [concrete = constants.%A.as.X.impl.F.e6cf46.1] {
// CHECK:STDOUT:     %return.patt: %pattern_type.1f4 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.1f4 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc23: Core.Form = init_form %B.ref, call_param0 [concrete = constants.%.e82]
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %A.as.X.impl.F.decl.loc23_14.2: %A.as.X.impl.F.type.170a02.2 = fn_decl @A.as.X.impl.F.loc23_14.2 [concrete = constants.%A.as.X.impl.F.e6cf46.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .F = %A.as.X.impl.F.decl.loc23_14.1
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.X.impl.F.loc23_14.1() -> out %return.param: %B;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.X.impl.F.loc23_14.2() -> out %return.param: %A [thunk @A.as.X.impl.%A.as.X.impl.F.decl.loc23_14.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %A.as.X.impl.F.type.170a02.1 = name_ref F, @A.as.X.impl.%A.as.X.impl.F.decl.loc23_14.1 [concrete = constants.%A.as.X.impl.F.e6cf46.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc23_14.1: ref %B = temporary_storage
// CHECK:STDOUT:   %A.as.X.impl.F.call: init %B to %.loc23_14.1 = call %F.ref()
// CHECK:STDOUT:   %.loc23_14.2: ref %B = temporary %.loc23_14.1, %A.as.X.impl.F.call
// CHECK:STDOUT:   %.loc23_14.3: ref %A = class_element_access %.loc23_14.2, element0
// CHECK:STDOUT:   %.loc23_14.4: ref %A = converted %A.as.X.impl.F.call, %.loc23_14.3
// CHECK:STDOUT:   %.loc23_14.5: %A = acquire_value %.loc23_14.4
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %.loc23_14.2, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%.loc23_14.2)
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %U.035: %Copy.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %U.binding.as_type.14b: type = symbolic_binding_type U, 0, %U.035 [symbolic]
// CHECK:STDOUT:   %pattern_type.9b9f0c.1: type = pattern_type %U.binding.as_type.14b [symbolic]
// CHECK:STDOUT:   %.6c815b.1: Core.Form = init_form %U.binding.as_type.14b, call_param1 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.19e4b8.1: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_34.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.8be29b.1: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %.4fe: Core.Form = init_form %ptr.e8f, call_param1 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.19e4b8.2: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_34.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.8be29b.2: %empty_tuple.type.as.I.impl.F.type.19e4b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.67ca8d.1: <witness> = require_complete_type %U.binding.as_type.14b [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.58dce0.2: <witness> = lookup_impl_witness %U.035, @Copy [symbolic]
// CHECK:STDOUT:   %.72e61c.2: type = fn_type_with_self_type %Copy.Op.type, %U.035 [symbolic]
// CHECK:STDOUT:   %impl.elem0.07b224.2: %.72e61c.2 = impl_witness_access %Copy.lookup_impl_witness.58dce0.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2c9874.2: <specific function> = specific_impl_function %impl.elem0.07b224.2, @Copy.Op(%U.035) [symbolic]
// CHECK:STDOUT:   %.2f2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2e6: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.2e6) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.specific_fn: <specific function> = specific_function %empty_tuple.type.as.I.impl.F.8be29b.1, @empty_tuple.type.as.I.impl.F.loc10_34.1(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.b46: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.201: %.b46 = impl_witness_access %Copy.lookup_impl_witness.2e6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b84: <specific function> = specific_impl_function %impl.elem0.201, @Copy.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_34.1: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_34.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.1] {
// CHECK:STDOUT:     %U.patt: %pattern_type.ce2 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @empty_tuple.type.as.I.impl.F.loc10_34.1.%pattern_type (%pattern_type.9b9f0c.1) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @empty_tuple.type.as.I.impl.F.loc10_34.1.%pattern_type (%pattern_type.9b9f0c.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @empty_tuple.type.as.I.impl.F.loc10_34.1.%pattern_type (%pattern_type.9b9f0c.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @empty_tuple.type.as.I.impl.F.loc10_34.1.%pattern_type (%pattern_type.9b9f0c.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc10_32: %Copy.type = name_ref U, %U.loc10_8.2 [symbolic = %U.loc10_8.1 (constants.%U.035)]
// CHECK:STDOUT:     %U.as_type.loc10_32: type = facet_access_type %U.ref.loc10_32 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:     %.loc10_32.3: type = converted %U.ref.loc10_32, %U.as_type.loc10_32 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:     %.loc10_32.4: Core.Form = init_form %.loc10_32.3, call_param1 [symbolic = %.loc10_32.2 (constants.%.6c815b.1)]
// CHECK:STDOUT:     %.loc10_16: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc10_8.2: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc10_8.1 (constants.%U.035)]
// CHECK:STDOUT:     %x.param: @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.14b) = value_param call_param0
// CHECK:STDOUT:     %.loc10_26.1: type = splice_block %.loc10_26.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)] {
// CHECK:STDOUT:       %U.ref.loc10_26: %Copy.type = name_ref U, %U.loc10_8.2 [symbolic = %U.loc10_8.1 (constants.%U.035)]
// CHECK:STDOUT:       %U.as_type.loc10_26: type = facet_access_type %U.ref.loc10_26 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:       %.loc10_26.2: type = converted %U.ref.loc10_26, %U.as_type.loc10_26 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.14b) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.14b) = out_param call_param1
// CHECK:STDOUT:     %return: ref @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.14b) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_34.2: %empty_tuple.type.as.I.impl.F.type.19e4b8.2 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_34.2 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl.loc10_34.1
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_tuple.type.as.I.impl.F.loc10_34.1(%U.loc10_8.2: %Copy.type) {
// CHECK:STDOUT:   %U.loc10_8.1: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc10_8.1 (constants.%U.035)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U.loc10_8.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.9b9f0c.1)]
// CHECK:STDOUT:   %.loc10_32.2: Core.Form = init_form %U.binding.as_type, call_param1 [symbolic = %.loc10_32.2 (constants.%.6c815b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete (constants.%require_complete.67ca8d.1)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc10_8.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58dce0.2)]
// CHECK:STDOUT:   %.loc10_43: type = fn_type_with_self_type constants.%Copy.Op.type, %U.loc10_8.1 [symbolic = %.loc10_43 (constants.%.72e61c.2)]
// CHECK:STDOUT:   %impl.elem0.loc10_43.2: @empty_tuple.type.as.I.impl.F.loc10_34.1.%.loc10_43 (%.72e61c.2) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_43.2 (constants.%impl.elem0.07b224.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_43.2: <specific function> = specific_impl_function %impl.elem0.loc10_43.2, @Copy.Op(%U.loc10_8.1) [symbolic = %specific_impl_fn.loc10_43.2 (constants.%specific_impl_fn.2c9874.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.14b)) -> out %return.param: @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.14b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.14b) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc10_43.1: @empty_tuple.type.as.I.impl.F.loc10_34.1.%.loc10_43 (%.72e61c.2) = impl_witness_access constants.%Copy.lookup_impl_witness.58dce0.2, element0 [symbolic = %impl.elem0.loc10_43.2 (constants.%impl.elem0.07b224.2)]
// CHECK:STDOUT:     %bound_method.loc10_43.1: <bound method> = bound_method %x.ref, %impl.elem0.loc10_43.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_43.1: <specific function> = specific_impl_function %impl.elem0.loc10_43.1, @Copy.Op(constants.%U.035) [symbolic = %specific_impl_fn.loc10_43.2 (constants.%specific_impl_fn.2c9874.2)]
// CHECK:STDOUT:     %bound_method.loc10_43.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_43.1
// CHECK:STDOUT:     %.loc10_32.1: ref @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.14b) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.Op.call: init @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.binding.as_type (%U.binding.as_type.14b) to %.loc10_32.1 = call %bound_method.loc10_43.2(%x.ref)
// CHECK:STDOUT:     return %Copy.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_tuple.type.as.I.impl.F.loc10_34.2(%T.loc5_8.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_34.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc5_8.1) [symbolic = %.loc10_34.3 (constants.%.2f2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2e6)]
// CHECK:STDOUT:   %Copy.facet.loc10_34.3: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc10_34.3 (constants.%Copy.facet)]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.specific_fn.loc10_34.2: <specific function> = specific_function constants.%empty_tuple.type.as.I.impl.F.8be29b.1, @empty_tuple.type.as.I.impl.F.loc10_34.1(%Copy.facet.loc10_34.3) [symbolic = %empty_tuple.type.as.I.impl.F.specific_fn.loc10_34.2 (constants.%empty_tuple.type.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @empty_tuple.type.as.I.impl.F.loc10_34.2.%ptr (%ptr.e8f)) -> out %return.param: @empty_tuple.type.as.I.impl.F.loc10_34.2.%ptr (%ptr.e8f) [thunk @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_34.1] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = name_ref F, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_34.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.1]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Copy.facet.loc10_34.1: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.2e6) [symbolic = %Copy.facet.loc10_34.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc10_34.1: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc10_34.1 [symbolic = %Copy.facet.loc10_34.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %Copy.facet.loc10_34.2: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.2e6) [symbolic = %Copy.facet.loc10_34.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc10_34.2: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc10_34.2 [symbolic = %Copy.facet.loc10_34.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.specific_fn.loc10_34.1: <specific function> = specific_function %F.ref, @empty_tuple.type.as.I.impl.F.loc10_34.1(constants.%Copy.facet) [symbolic = %empty_tuple.type.as.I.impl.F.specific_fn.loc10_34.2 (constants.%empty_tuple.type.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.call: init @empty_tuple.type.as.I.impl.F.loc10_34.2.%ptr (%ptr.e8f) = call %empty_tuple.type.as.I.impl.F.specific_fn.loc10_34.1(%x.ref)
// CHECK:STDOUT:     return %empty_tuple.type.as.I.impl.F.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc10_34.1(constants.%U.035) {
// CHECK:STDOUT:   %U.loc10_8.1 => constants.%U.035
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type.14b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9b9f0c.1
// CHECK:STDOUT:   %.loc10_32.2 => constants.%.6c815b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc10_34.2(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc5_8.1 => constants.%T.67d
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc5 => constants.%.4fe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc10_34.1(constants.%Copy.facet) {
// CHECK:STDOUT:   %U.loc10_8.1 => constants.%Copy.facet
// CHECK:STDOUT:   %U.binding.as_type => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc10_32.2 => constants.%.4fe
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ef1
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.2e6
// CHECK:STDOUT:   %.loc10_43 => constants.%.b46
// CHECK:STDOUT:   %impl.elem0.loc10_43.2 => constants.%impl.elem0.201
// CHECK:STDOUT:   %specific_impl_fn.loc10_43.2 => constants.%specific_impl_fn.b84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %U.035: %Copy.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %U.binding.as_type.14b: type = symbolic_binding_type U, 0, %U.035 [symbolic]
// CHECK:STDOUT:   %pattern_type.9b9f0c.1: type = pattern_type %U.binding.as_type.14b [symbolic]
// CHECK:STDOUT:   %.0e8: Core.Form = init_form %U.binding.as_type.14b, call_param2 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.19e4b8.1: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_44.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.8be29b.1: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %.924: Core.Form = init_form %ptr.e8f, call_param2 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.19e4b8.2: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_44.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.8be29b.2: %empty_tuple.type.as.I.impl.F.type.19e4b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.67ca8d.1: <witness> = require_complete_type %U.binding.as_type.14b [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.58dce0.2: <witness> = lookup_impl_witness %U.035, @Copy [symbolic]
// CHECK:STDOUT:   %.72e61c.2: type = fn_type_with_self_type %Copy.Op.type, %U.035 [symbolic]
// CHECK:STDOUT:   %impl.elem0.07b224.2: %.72e61c.2 = impl_witness_access %Copy.lookup_impl_witness.58dce0.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2c9874.2: <specific function> = specific_impl_function %impl.elem0.07b224.2, @Copy.Op(%U.035) [symbolic]
// CHECK:STDOUT:   %.2f2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2e6: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.2e6) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.specific_fn: <specific function> = specific_function %empty_tuple.type.as.I.impl.F.8be29b.1, @empty_tuple.type.as.I.impl.F.loc10_44.1(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.b46: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.201: %.b46 = impl_witness_access %Copy.lookup_impl_witness.2e6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b84: <specific function> = specific_impl_function %impl.elem0.201, @Copy.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_44.1: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_44.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.ce2 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @empty_tuple.type.as.I.impl.F.loc10_44.1.%pattern_type (%pattern_type.9b9f0c.1) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @empty_tuple.type.as.I.impl.F.loc10_44.1.%pattern_type (%pattern_type.9b9f0c.1) = value_param_pattern %x.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @empty_tuple.type.as.I.impl.F.loc10_44.1.%pattern_type (%pattern_type.9b9f0c.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @empty_tuple.type.as.I.impl.F.loc10_44.1.%pattern_type (%pattern_type.9b9f0c.1) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc10_42: %Copy.type = name_ref U, %U.loc10_18.2 [symbolic = %U.loc10_18.1 (constants.%U.035)]
// CHECK:STDOUT:     %U.as_type.loc10_42: type = facet_access_type %U.ref.loc10_42 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:     %.loc10_42.3: type = converted %U.ref.loc10_42, %U.as_type.loc10_42 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:     %.loc10_42.4: Core.Form = init_form %.loc10_42.3, call_param2 [symbolic = %.loc10_42.2 (constants.%.0e8)]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc10_15.1: type = splice_block %.loc10_15.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc10_15.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc10_15.3: type = converted %.loc10_15.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %empty_tuple.type = value_binding self, %self.param
// CHECK:STDOUT:     %.loc10_26: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc10_18.2: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc10_18.1 (constants.%U.035)]
// CHECK:STDOUT:     %x.param: @empty_tuple.type.as.I.impl.F.loc10_44.1.%U.binding.as_type (%U.binding.as_type.14b) = value_param call_param1
// CHECK:STDOUT:     %.loc10_36.1: type = splice_block %.loc10_36.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)] {
// CHECK:STDOUT:       %U.ref.loc10_36: %Copy.type = name_ref U, %U.loc10_18.2 [symbolic = %U.loc10_18.1 (constants.%U.035)]
// CHECK:STDOUT:       %U.as_type.loc10_36: type = facet_access_type %U.ref.loc10_36 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:       %.loc10_36.2: type = converted %U.ref.loc10_36, %U.as_type.loc10_36 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @empty_tuple.type.as.I.impl.F.loc10_44.1.%U.binding.as_type (%U.binding.as_type.14b) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @empty_tuple.type.as.I.impl.F.loc10_44.1.%U.binding.as_type (%U.binding.as_type.14b) = out_param call_param2
// CHECK:STDOUT:     %return: ref @empty_tuple.type.as.I.impl.F.loc10_44.1.%U.binding.as_type (%U.binding.as_type.14b) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_44.2: %empty_tuple.type.as.I.impl.F.type.19e4b8.2 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_44.2 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl.loc10_44.1
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_tuple.type.as.I.impl.F.loc10_44.1(%U.loc10_18.2: %Copy.type) {
// CHECK:STDOUT:   %U.loc10_18.1: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc10_18.1 (constants.%U.035)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U.loc10_18.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.9b9f0c.1)]
// CHECK:STDOUT:   %.loc10_42.2: Core.Form = init_form %U.binding.as_type, call_param2 [symbolic = %.loc10_42.2 (constants.%.0e8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete (constants.%require_complete.67ca8d.1)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc10_18.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58dce0.2)]
// CHECK:STDOUT:   %.loc10_53: type = fn_type_with_self_type constants.%Copy.Op.type, %U.loc10_18.1 [symbolic = %.loc10_53 (constants.%.72e61c.2)]
// CHECK:STDOUT:   %impl.elem0.loc10_53.2: @empty_tuple.type.as.I.impl.F.loc10_44.1.%.loc10_53 (%.72e61c.2) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_53.2 (constants.%impl.elem0.07b224.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_53.2: <specific function> = specific_impl_function %impl.elem0.loc10_53.2, @Copy.Op(%U.loc10_18.1) [symbolic = %specific_impl_fn.loc10_53.2 (constants.%specific_impl_fn.2c9874.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: %empty_tuple.type, %x.param: @empty_tuple.type.as.I.impl.F.loc10_44.1.%U.binding.as_type (%U.binding.as_type.14b)) -> out %return.param: @empty_tuple.type.as.I.impl.F.loc10_44.1.%U.binding.as_type (%U.binding.as_type.14b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @empty_tuple.type.as.I.impl.F.loc10_44.1.%U.binding.as_type (%U.binding.as_type.14b) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc10_53.1: @empty_tuple.type.as.I.impl.F.loc10_44.1.%.loc10_53 (%.72e61c.2) = impl_witness_access constants.%Copy.lookup_impl_witness.58dce0.2, element0 [symbolic = %impl.elem0.loc10_53.2 (constants.%impl.elem0.07b224.2)]
// CHECK:STDOUT:     %bound_method.loc10_53.1: <bound method> = bound_method %x.ref, %impl.elem0.loc10_53.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_53.1: <specific function> = specific_impl_function %impl.elem0.loc10_53.1, @Copy.Op(constants.%U.035) [symbolic = %specific_impl_fn.loc10_53.2 (constants.%specific_impl_fn.2c9874.2)]
// CHECK:STDOUT:     %bound_method.loc10_53.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_53.1
// CHECK:STDOUT:     %.loc10_42.1: ref @empty_tuple.type.as.I.impl.F.loc10_44.1.%U.binding.as_type (%U.binding.as_type.14b) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.Op.call: init @empty_tuple.type.as.I.impl.F.loc10_44.1.%U.binding.as_type (%U.binding.as_type.14b) to %.loc10_42.1 = call %bound_method.loc10_53.2(%x.ref)
// CHECK:STDOUT:     return %Copy.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_tuple.type.as.I.impl.F.loc10_44.2(%T.loc5_20.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_44.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc5_20.1) [symbolic = %.loc10_44.3 (constants.%.2f2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2e6)]
// CHECK:STDOUT:   %Copy.facet.loc10_44.3: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc10_44.3 (constants.%Copy.facet)]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.specific_fn.loc10_44.2: <specific function> = specific_function constants.%empty_tuple.type.as.I.impl.F.8be29b.1, @empty_tuple.type.as.I.impl.F.loc10_44.1(%Copy.facet.loc10_44.3) [symbolic = %empty_tuple.type.as.I.impl.F.specific_fn.loc10_44.2 (constants.%empty_tuple.type.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: %empty_tuple.type, %x.param: @empty_tuple.type.as.I.impl.F.loc10_44.2.%ptr (%ptr.e8f)) -> out %return.param: @empty_tuple.type.as.I.impl.F.loc10_44.2.%ptr (%ptr.e8f) [thunk @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_44.1] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %empty_tuple.type.as.I.impl.F.type.19e4b8.1 = name_ref F, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_44.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.8be29b.1]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.bound: <bound method> = bound_method %self.ref, %F.ref
// CHECK:STDOUT:     %Copy.facet.loc10_44.1: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.2e6) [symbolic = %Copy.facet.loc10_44.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc10_44.1: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc10_44.1 [symbolic = %Copy.facet.loc10_44.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %Copy.facet.loc10_44.2: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.2e6) [symbolic = %Copy.facet.loc10_44.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc10_44.2: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc10_44.2 [symbolic = %Copy.facet.loc10_44.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.specific_fn.loc10_44.1: <specific function> = specific_function %F.ref, @empty_tuple.type.as.I.impl.F.loc10_44.1(constants.%Copy.facet) [symbolic = %empty_tuple.type.as.I.impl.F.specific_fn.loc10_44.2 (constants.%empty_tuple.type.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %self.ref, %empty_tuple.type.as.I.impl.F.specific_fn.loc10_44.1
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.call: init @empty_tuple.type.as.I.impl.F.loc10_44.2.%ptr (%ptr.e8f) = call %bound_method(%self.ref, %x.ref)
// CHECK:STDOUT:     return %empty_tuple.type.as.I.impl.F.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc10_44.1(constants.%U.035) {
// CHECK:STDOUT:   %U.loc10_18.1 => constants.%U.035
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type.14b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9b9f0c.1
// CHECK:STDOUT:   %.loc10_42.2 => constants.%.0e8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc10_44.2(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc5_20.1 => constants.%T.67d
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc5 => constants.%.924
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc10_44.1(constants.%Copy.facet) {
// CHECK:STDOUT:   %U.loc10_18.1 => constants.%Copy.facet
// CHECK:STDOUT:   %U.binding.as_type => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc10_42.2 => constants.%.924
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ef1
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.2e6
// CHECK:STDOUT:   %.loc10_53 => constants.%.b46
// CHECK:STDOUT:   %impl.elem0.loc10_53.2 => constants.%impl.elem0.201
// CHECK:STDOUT:   %specific_impl_fn.loc10_53.2 => constants.%specific_impl_fn.b84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.b.a.1b0: type = struct_type {.b: %empty_struct_type, .a: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %.a97: Core.Form = init_form %struct_type.b.a.1b0, call_param0 [concrete]
// CHECK:STDOUT:   %pattern_type.914: type = pattern_type %struct_type.b.a.1b0 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.e82c13.1: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_29.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.5f0cef.1: %empty_tuple.type.as.I.impl.F.type.e82c13.1 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.f95: type = struct_type {.a: %empty_struct_type, .b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.e82c13.2: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_29.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.5f0cef.2: %empty_tuple.type.as.I.impl.F.type.e82c13.2 = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.f95 = struct_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc8_7.2 as %I.type {
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_29.1: %empty_tuple.type.as.I.impl.F.type.e82c13.1 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_29.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.5f0cef.1] {
// CHECK:STDOUT:     %return.patt: %pattern_type.914 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.914 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_19.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc10_19.2: type = converted %.loc10_19.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc10_27.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc10_27.2: type = converted %.loc10_27.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.b.a: type = struct_type {.b: %empty_struct_type, .a: %empty_struct_type} [concrete = constants.%struct_type.b.a.1b0]
// CHECK:STDOUT:     %.loc10_28: Core.Form = init_form %struct_type.b.a, call_param0 [concrete = constants.%.a97]
// CHECK:STDOUT:     %return.param: ref %struct_type.b.a.1b0 = out_param call_param0
// CHECK:STDOUT:     %return: ref %struct_type.b.a.1b0 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_29.2: %empty_tuple.type.as.I.impl.F.type.e82c13.2 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_29.2 [concrete = constants.%empty_tuple.type.as.I.impl.F.5f0cef.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl.loc10_29.1
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F.loc10_29.1() -> out %return.param: %struct_type.b.a.1b0;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F.loc10_29.2() -> out %return.param: %struct_type.a.b.f95 [thunk @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_29.1] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %empty_tuple.type.as.I.impl.F.type.e82c13.1 = name_ref F, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_29.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.5f0cef.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_29.1: ref %struct_type.b.a.1b0 = temporary_storage
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.call: init %struct_type.b.a.1b0 to %.loc10_29.1 = call %F.ref()
// CHECK:STDOUT:   %.loc10_29.2: ref %struct_type.b.a.1b0 = temporary %.loc10_29.1, %empty_tuple.type.as.I.impl.F.call
// CHECK:STDOUT:   %.loc10_29.3: ref %empty_struct_type = struct_access %.loc10_29.2, element1
// CHECK:STDOUT:   %.loc10_29.4: ref %empty_struct_type = struct_access %return.param, element1
// CHECK:STDOUT:   %.loc10_29.5: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_29.6: init %empty_struct_type = converted %.loc10_29.3, %.loc10_29.5 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_29.7: ref %empty_struct_type = struct_access %.loc10_29.2, element0
// CHECK:STDOUT:   %.loc10_29.8: ref %empty_struct_type = struct_access %return.param, element0
// CHECK:STDOUT:   %.loc10_29.9: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_29.10: init %empty_struct_type = converted %.loc10_29.7, %.loc10_29.9 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_29.11: init %struct_type.a.b.f95 to %return.param = struct_init (%.loc10_29.6, %.loc10_29.10) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc10_29.12: init %struct_type.a.b.f95 = converted %empty_tuple.type.as.I.impl.F.call, %.loc10_29.11 [concrete = constants.%struct]
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %.loc10_29.2, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%.loc10_29.2)
// CHECK:STDOUT:   return %.loc10_29.12 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %struct_type.b.a.1b0) = "no_op";
// CHECK:STDOUT:
