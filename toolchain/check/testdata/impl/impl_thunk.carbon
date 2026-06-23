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
  fn F(ref self, other: Self*) -> Self*;
}

impl B as X {
  //@dump-sem-ir-begin
  fn F(ref self: A, other: A*) -> C*;
  //@dump-sem-ir-end
}

// --- inheritance_value_conversion.carbon

library "[[@TEST_NAME]]";

base class A {}
class B { extend base: A; }

interface X {
  fn F(self, other: Self);
}

impl B as X {
  //@dump-sem-ir-begin
  fn F(self: A, other: A);
  //@dump-sem-ir-end
}

// --- inheritance_value_conversion_pointer.carbon

library "[[@TEST_NAME]]";

base class A {}
base class B { extend base: A; }
class C { extend base: B; }

interface X {
  fn F(ref self, other: Self*) -> Self*;
}

impl B as X {
  //@dump-sem-ir-begin
  fn F(ref self: A, other: A*) -> C*;
  //@dump-sem-ir-end
}

// --- inheritance_value_conversion_pointer_addr.carbon

library "[[@TEST_NAME]]";

base class A {}
base class B { extend base: A; }
class C { extend base: B; }

interface X {
  fn F(ref self, other: Self*) -> Self*;
}

impl B as X {
  //@dump-sem-ir-begin
  fn F(ref self: A, other: A*) -> C*;
  //@dump-sem-ir-end
}

// --- inheritance_ref_conversion.carbon

library "[[@TEST_NAME]]";

base class A {}
base class B { extend base: A; }

interface X {
  fn F(self, ref other: Self);
}

impl B as X {
  //@dump-sem-ir-begin
  fn F(self: A, ref other: A);
  //@dump-sem-ir-end
}

// --- form_parameter.carbon

interface I {
  let T:! type;
  let Fm:! Core.Form;
  fn F(x:? Fm, y: {.a: (), .b: ()});
}

class C {
  impl as I where .T = () and .Fm = form(ref ()) {
    //@dump-sem-ir-begin
    fn F(x:? form(ref ()), y: {.b: (), .a: ()});
    //@dump-sem-ir-end
  }
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
  // CHECK:STDERR: fail_inheritance_value_conversion_copy_return.carbon:[[@LINE+7]]:3: note: type `A` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
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
// CHECK:STDERR: fail_param_type_mismatch.carbon: error: cannot implicitly convert expression of type `A` to `B` [ConversionFailure]
// CHECK:STDERR: fail_param_type_mismatch.carbon: note: type `A` does not implement interface `Core.ImplicitAs(B)` [MissingImplInMemberAccessInContext]

library "[[@TEST_NAME]]";

interface I {
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
  // CHECK:STDERR: fail_return_mismatch.carbon:[[@LINE+7]]:3: note: type `B` does not implement interface `Core.ImplicitAs(A)` [MissingImplInMemberAccessInContext]
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
// CHECK:STDERR: fail_param_type_incomplete.carbon: error: parameter has incomplete type `B` in function definition [IncompleteTypeInFunctionParam]

library "[[@TEST_NAME]]";

interface I(T:! type) {
  fn F(a: T);
}

// CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE+3]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class B;
// CHECK:STDERR: ^~~~~~~~
class B;

class C {
  impl as I(B) {
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE+13]]:5: note: while building thunk calling this function [ThunkCallee]
    // CHECK:STDERR:     fn F(c: C);
    // CHECK:STDERR:     ^~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_param_type_incomplete.carbon: error: cannot implicitly convert expression of type `B` to `C` [ConversionFailure]
    // CHECK:STDERR: fail_param_type_incomplete.carbon: note: type `B` does not implement interface `Core.ImplicitAs(C)` [MissingImplInMemberAccessInContext]
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE+7]]:10: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR:     fn F(c: C);
    // CHECK:STDERR:          ^~~~
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE-19]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
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
  fn F[T:! type](self, x: T*) -> T*;
}

impl () as I {
  //@dump-sem-ir-begin
  fn F[U:! Core.Copy](unused self: (), x: U) -> U { return x; }
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
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.391: type = struct_type {.a: %empty_tuple.type, .b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct_type.c.d.15a: type = struct_type {.c: %empty_tuple.type, .d: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @empty_tuple.type.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %struct_type.b.a.40c: type = struct_type {.b: %empty_struct_type, .a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.231: type = pattern_type %struct_type.b.a.40c [concrete]
// CHECK:STDOUT:   %y.param_patt: %pattern_type.231 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.231 = at_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:   %struct_type.d.c.b36: type = struct_type {.d: %empty_struct_type, .c: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %.4b1: Core.Form = init_form %struct_type.d.c.b36 [concrete]
// CHECK:STDOUT:   %pattern_type.844: type = pattern_type %struct_type.d.c.b36 [concrete]
// CHECK:STDOUT:   %return.param_patt.4a9: %pattern_type.844 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.706: %pattern_type.844 = return_slot_pattern %return.param_patt.4a9, %struct_type.d.c.b36 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.48cdab.1: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_48.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.bd5be1.1: %empty_tuple.type.as.I.impl.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_tuple.type, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.48cdab.2: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_48.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.bd5be1.2: %empty_tuple.type.as.I.impl.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.c.d.15a = struct_value (%empty_tuple, %empty_struct) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc10_48.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc8_7.2 as %I.ref {
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_48.1: %empty_tuple.type.as.I.impl.F.type.48cdab.1 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_48.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.bd5be1.1] {
// CHECK:STDOUT:     %y.param_patt: %pattern_type.231 = value_param_pattern [concrete = constants.%y.param_patt]
// CHECK:STDOUT:     %y.patt: %pattern_type.231 = at_binding_pattern y, %y.param_patt [concrete = constants.%y.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.844 = out_param_pattern [concrete = constants.%return.param_patt.4a9]
// CHECK:STDOUT:     %return.patt: %pattern_type.844 = return_slot_pattern %return.param_patt, %struct_type.d.c [concrete = constants.%return.patt.706]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_38.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc10_38.2: type = converted %.loc10_38.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc10_46.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc10_46.2: type = converted %.loc10_46.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.d.c: type = struct_type {.d: %empty_struct_type, .c: %empty_tuple.type} [concrete = constants.%struct_type.d.c.b36]
// CHECK:STDOUT:     %.loc10_47: Core.Form = init_form %struct_type.d.c [concrete = constants.%.4b1]
// CHECK:STDOUT:     %y.param: %struct_type.b.a.40c = value_param call_param0
// CHECK:STDOUT:     %.loc10_26: type = splice_block %struct_type.b.a [concrete = constants.%struct_type.b.a.40c] {
// CHECK:STDOUT:       %.loc10_17.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc10_17.2: type = converted %.loc10_17.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc10_25.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc10_25.2: type = converted %.loc10_25.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.b.a: type = struct_type {.b: %empty_struct_type, .a: %empty_tuple.type} [concrete = constants.%struct_type.b.a.40c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: %struct_type.b.a.40c = wrapper_binding y, %y.param
// CHECK:STDOUT:     %return.param: ref %struct_type.d.c.b36 = out_param call_param1
// CHECK:STDOUT:     %return: ref %struct_type.d.c.b36 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_48.2: %empty_tuple.type.as.I.impl.F.type.48cdab.2 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_48.2 [concrete = constants.%empty_tuple.type.as.I.impl.F.bd5be1.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl.loc10_48.1
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F.loc10_48.1(%y.param: %struct_type.b.a.40c) -> out %return.param: %struct_type.d.c.b36;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F.loc10_48.2(%x.param: %struct_type.a.b.391) -> out %return.param: %struct_type.c.d.15a [thunk @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_48.1 for @I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %empty_tuple.type.as.I.impl.F.type.48cdab.1 = name_ref F, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_48.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.bd5be1.1]
// CHECK:STDOUT:   %.loc10_48.1: ref %struct_type.d.c.b36 = temporary_storage
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.call: init %struct_type.d.c.b36 to %.loc10_48.1 = call %F.ref(%.5)
// CHECK:STDOUT:   %.loc10_48.2: ref %struct_type.d.c.b36 = temporary %.loc10_48.1, %empty_tuple.type.as.I.impl.F.call
// CHECK:STDOUT:   %.loc10_48.3: ref %empty_tuple.type = struct_access %.loc10_48.2, element1
// CHECK:STDOUT:   %.loc10_48.4: ref %empty_tuple.type = struct_access %return.param, element1
// CHECK:STDOUT:   %.loc10_48.5: init %empty_tuple.type to %.loc10_48.4 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_48.6: init %empty_tuple.type = converted %.loc10_48.3, %.loc10_48.5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_48.7: init %empty_tuple.type to %.loc10_48.4 = in_place_init %.loc10_48.6 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc10_48.8: ref %empty_struct_type = struct_access %.loc10_48.2, element0
// CHECK:STDOUT:   %.loc10_48.9: ref %empty_struct_type = struct_access %return.param, element0
// CHECK:STDOUT:   %.loc10_48.10: init %empty_struct_type to %.loc10_48.9 = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_48.11: init %empty_struct_type = converted %.loc10_48.8, %.loc10_48.10 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_48.12: init %empty_struct_type to %.loc10_48.9 = in_place_init %.loc10_48.11 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_48.13: init %struct_type.c.d.15a to %return.param = struct_init (%.loc10_48.7, %.loc10_48.12) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc10_48.14: init %struct_type.c.d.15a = converted %empty_tuple.type.as.I.impl.F.call, %.loc10_48.13 [concrete = constants.%struct]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc10_48.2, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc10_48.2)
// CHECK:STDOUT:   return %.loc10_48.14 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_48.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_48.2(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_48.3(%self.param: ref %struct_type.d.c.b36) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %X.impl_witness: <witness> = impl_witness @B.as.X.impl.%X.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %self.param_patt.8fb: %pattern_type.9ef = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.680: %pattern_type.9ef = at_binding_pattern self, %self.param_patt.8fb [concrete]
// CHECK:STDOUT:   %ptr.d43: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.8a5: type = pattern_type %ptr.d43 [concrete]
// CHECK:STDOUT:   %other.param_patt.7dd: %pattern_type.8a5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %other.patt.95f: %pattern_type.8a5 = at_binding_pattern other, %other.param_patt.7dd [concrete]
// CHECK:STDOUT:   %ptr.6b6: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.8fa: Core.Form = init_form %ptr.6b6 [concrete]
// CHECK:STDOUT:   %pattern_type.fcb: type = pattern_type %ptr.6b6 [concrete]
// CHECK:STDOUT:   %return.param_patt.ada: %pattern_type.fcb = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.783: %pattern_type.fcb = return_slot_pattern %return.param_patt.ada, %ptr.6b6 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.48cdab.1: type = fn_type @B.as.X.impl.F.loc14_37.1 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.bd5be1.1: %B.as.X.impl.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %X.facet: %X.type = facet_value %B, (%X.impl_witness) [concrete]
// CHECK:STDOUT:   %ptr.432: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.48cdab.2: type = fn_type @B.as.X.impl.F.loc14_37.2 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.bd5be1.2: %B.as.X.impl.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.09a: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.d4b: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.70e: %ptr.as.Copy.impl.Op.type.d4b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.432, (%Copy.impl_witness.09a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.78a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.fa1: type = fn_type_with_self_type %Copy.WithSelf.Op.type.78a, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.70e, @ptr.as.Copy.impl.Op(%B) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.X.impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc14_37.1: %B.as.X.impl.F.type.48cdab.1 = fn_decl @B.as.X.impl.F.loc14_37.1 [concrete = constants.%B.as.X.impl.F.bd5be1.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9ef = ref_param_pattern [concrete = constants.%self.param_patt.8fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.680]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.8a5 = value_param_pattern [concrete = constants.%other.param_patt.7dd]
// CHECK:STDOUT:     %other.patt: %pattern_type.8a5 = at_binding_pattern other, %other.param_patt [concrete = constants.%other.patt.95f]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.fcb = out_param_pattern [concrete = constants.%return.param_patt.ada]
// CHECK:STDOUT:     %return.patt: %pattern_type.fcb = return_slot_pattern %return.param_patt, %ptr.loc14_36 [concrete = constants.%return.patt.783]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc14_36: type = ptr_type %C.ref [concrete = constants.%ptr.6b6]
// CHECK:STDOUT:     %.loc14_36: Core.Form = init_form %ptr.loc14_36 [concrete = constants.%.8fa]
// CHECK:STDOUT:     %self.param: ref %A = ref_param call_param0
// CHECK:STDOUT:     %A.ref.loc14_18: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %self: ref %A = wrapper_binding self, %self.param
// CHECK:STDOUT:     %other.param: %ptr.d43 = value_param call_param1
// CHECK:STDOUT:     %.loc14_29: type = splice_block %ptr.loc14_29 [concrete = constants.%ptr.d43] {
// CHECK:STDOUT:       %A.ref.loc14_28: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc14_29: type = ptr_type %A.ref.loc14_28 [concrete = constants.%ptr.d43]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: %ptr.d43 = wrapper_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref %ptr.6b6 = out_param call_param2
// CHECK:STDOUT:     %return: ref %ptr.6b6 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc14_37.2: %B.as.X.impl.F.type.48cdab.2 = fn_decl @B.as.X.impl.F.loc14_37.2 [concrete = constants.%B.as.X.impl.F.bd5be1.2] {
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
// CHECK:STDOUT:   extend %X.ref
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc14_37.1(%self.param: ref %A, %other.param: %ptr.d43) -> out %return.param: %ptr.6b6;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc14_37.2(%self.param: ref %B, %other.param: %ptr.432) -> out %return.param: %ptr.432 [thunk @B.as.X.impl.%B.as.X.impl.F.decl.loc14_37.1 for @X.WithSelf.%X.WithSelf.F.decl, @X.WithSelf.F(constants.%X.facet)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %B.as.X.impl.F.type.48cdab.1 = name_ref F, @B.as.X.impl.%B.as.X.impl.F.decl.loc14_37.1 [concrete = constants.%B.as.X.impl.F.bd5be1.1]
// CHECK:STDOUT:   %B.as.X.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.call: init %ptr.6b6 = call %B.as.X.impl.F.bound(%.6, %.9)
// CHECK:STDOUT:   %.loc14_37.1: %ptr.6b6 = value_of_initializer %B.as.X.impl.F.call
// CHECK:STDOUT:   %.loc14_37.2: %ptr.6b6 = converted %B.as.X.impl.F.call, %.loc14_37.1
// CHECK:STDOUT:   %.loc14_37.3: ref %C = deref %.loc14_37.2
// CHECK:STDOUT:   %.loc14_37.4: ref %B = class_element_access %.loc14_37.3, element0
// CHECK:STDOUT:   %addr.loc14: %ptr.432 = addr_of %.loc14_37.4
// CHECK:STDOUT:   %.loc14_37.5: %ptr.432 = converted %B.as.X.impl.F.call, %addr.loc14
// CHECK:STDOUT:   %impl.elem0: %.fa1 = impl_witness_access constants.%Copy.impl_witness.09a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.70e]
// CHECK:STDOUT:   %bound_method.loc14_37.1: <bound method> = bound_method %.loc14_37.5, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%B) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_37.2: <bound method> = bound_method %.loc14_37.5, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.432 = call %bound_method.loc14_37.2(%.loc14_37.5)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_value_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %X.impl_witness: <witness> = impl_witness @B.as.X.impl.%X.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %self.param_patt.04e: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.312: %pattern_type.9ef = at_binding_pattern self, %self.param_patt.04e [concrete]
// CHECK:STDOUT:   %other.param_patt.484: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %other.patt.9f6: %pattern_type.9ef = at_binding_pattern other, %other.param_patt.484 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.48cdab.1: type = fn_type @B.as.X.impl.F.loc13_26.1 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.bd5be1.1: %B.as.X.impl.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %X.facet: %X.type = facet_value %B, (%X.impl_witness) [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.48cdab.2: type = fn_type @B.as.X.impl.F.loc13_26.2 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.bd5be1.2: %B.as.X.impl.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.X.impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc13_26.1: %B.as.X.impl.F.type.48cdab.1 = fn_decl @B.as.X.impl.F.loc13_26.1 [concrete = constants.%B.as.X.impl.F.bd5be1.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9ef = value_param_pattern [concrete = constants.%self.param_patt.04e]
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.312]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.9ef = value_param_pattern [concrete = constants.%other.param_patt.484]
// CHECK:STDOUT:     %other.patt: %pattern_type.9ef = at_binding_pattern other, %other.param_patt [concrete = constants.%other.patt.9f6]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref.loc13_14: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = wrapper_binding self, %self.param
// CHECK:STDOUT:     %other.param: %A = value_param call_param1
// CHECK:STDOUT:     %A.ref.loc13_24: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %other: %A = wrapper_binding other, %other.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc13_26.2: %B.as.X.impl.F.type.48cdab.2 = fn_decl @B.as.X.impl.F.loc13_26.2 [concrete = constants.%B.as.X.impl.F.bd5be1.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .F = %B.as.X.impl.F.decl.loc13_26.1
// CHECK:STDOUT:   extend %X.ref
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc13_26.1(%self.param: %A, %other.param: %A);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc13_26.2(%self.param: %B, %other.param: %B) [thunk @B.as.X.impl.%B.as.X.impl.F.decl.loc13_26.1 for @X.WithSelf.%X.WithSelf.F.decl, @X.WithSelf.F(constants.%X.facet)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %B.as.X.impl.F.type.48cdab.1 = name_ref F, @B.as.X.impl.%B.as.X.impl.F.decl.loc13_26.1 [concrete = constants.%B.as.X.impl.F.bd5be1.1]
// CHECK:STDOUT:   %B.as.X.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.call: init %empty_tuple.type = call %B.as.X.impl.F.bound(%.6, %.10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_value_conversion_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %X.impl_witness: <witness> = impl_witness @B.as.X.impl.%X.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %self.param_patt.8fb: %pattern_type.9ef = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.680: %pattern_type.9ef = at_binding_pattern self, %self.param_patt.8fb [concrete]
// CHECK:STDOUT:   %ptr.d43: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.8a5: type = pattern_type %ptr.d43 [concrete]
// CHECK:STDOUT:   %other.param_patt.7dd: %pattern_type.8a5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %other.patt.95f: %pattern_type.8a5 = at_binding_pattern other, %other.param_patt.7dd [concrete]
// CHECK:STDOUT:   %ptr.6b6: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.8fa: Core.Form = init_form %ptr.6b6 [concrete]
// CHECK:STDOUT:   %pattern_type.fcb: type = pattern_type %ptr.6b6 [concrete]
// CHECK:STDOUT:   %return.param_patt.ada: %pattern_type.fcb = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.783: %pattern_type.fcb = return_slot_pattern %return.param_patt.ada, %ptr.6b6 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.48cdab.1: type = fn_type @B.as.X.impl.F.loc14_37.1 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.bd5be1.1: %B.as.X.impl.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %X.facet: %X.type = facet_value %B, (%X.impl_witness) [concrete]
// CHECK:STDOUT:   %ptr.432: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.48cdab.2: type = fn_type @B.as.X.impl.F.loc14_37.2 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.bd5be1.2: %B.as.X.impl.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.09a: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.d4b: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.70e: %ptr.as.Copy.impl.Op.type.d4b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.432, (%Copy.impl_witness.09a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.78a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.fa1: type = fn_type_with_self_type %Copy.WithSelf.Op.type.78a, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.70e, @ptr.as.Copy.impl.Op(%B) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.X.impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc14_37.1: %B.as.X.impl.F.type.48cdab.1 = fn_decl @B.as.X.impl.F.loc14_37.1 [concrete = constants.%B.as.X.impl.F.bd5be1.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9ef = ref_param_pattern [concrete = constants.%self.param_patt.8fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.680]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.8a5 = value_param_pattern [concrete = constants.%other.param_patt.7dd]
// CHECK:STDOUT:     %other.patt: %pattern_type.8a5 = at_binding_pattern other, %other.param_patt [concrete = constants.%other.patt.95f]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.fcb = out_param_pattern [concrete = constants.%return.param_patt.ada]
// CHECK:STDOUT:     %return.patt: %pattern_type.fcb = return_slot_pattern %return.param_patt, %ptr.loc14_36 [concrete = constants.%return.patt.783]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc14_36: type = ptr_type %C.ref [concrete = constants.%ptr.6b6]
// CHECK:STDOUT:     %.loc14_36: Core.Form = init_form %ptr.loc14_36 [concrete = constants.%.8fa]
// CHECK:STDOUT:     %self.param: ref %A = ref_param call_param0
// CHECK:STDOUT:     %A.ref.loc14_18: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %self: ref %A = wrapper_binding self, %self.param
// CHECK:STDOUT:     %other.param: %ptr.d43 = value_param call_param1
// CHECK:STDOUT:     %.loc14_29: type = splice_block %ptr.loc14_29 [concrete = constants.%ptr.d43] {
// CHECK:STDOUT:       %A.ref.loc14_28: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc14_29: type = ptr_type %A.ref.loc14_28 [concrete = constants.%ptr.d43]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: %ptr.d43 = wrapper_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref %ptr.6b6 = out_param call_param2
// CHECK:STDOUT:     %return: ref %ptr.6b6 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc14_37.2: %B.as.X.impl.F.type.48cdab.2 = fn_decl @B.as.X.impl.F.loc14_37.2 [concrete = constants.%B.as.X.impl.F.bd5be1.2] {
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
// CHECK:STDOUT:   extend %X.ref
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc14_37.1(%self.param: ref %A, %other.param: %ptr.d43) -> out %return.param: %ptr.6b6;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc14_37.2(%self.param: ref %B, %other.param: %ptr.432) -> out %return.param: %ptr.432 [thunk @B.as.X.impl.%B.as.X.impl.F.decl.loc14_37.1 for @X.WithSelf.%X.WithSelf.F.decl, @X.WithSelf.F(constants.%X.facet)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %B.as.X.impl.F.type.48cdab.1 = name_ref F, @B.as.X.impl.%B.as.X.impl.F.decl.loc14_37.1 [concrete = constants.%B.as.X.impl.F.bd5be1.1]
// CHECK:STDOUT:   %B.as.X.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.call: init %ptr.6b6 = call %B.as.X.impl.F.bound(%.6, %.9)
// CHECK:STDOUT:   %.loc14_37.1: %ptr.6b6 = value_of_initializer %B.as.X.impl.F.call
// CHECK:STDOUT:   %.loc14_37.2: %ptr.6b6 = converted %B.as.X.impl.F.call, %.loc14_37.1
// CHECK:STDOUT:   %.loc14_37.3: ref %C = deref %.loc14_37.2
// CHECK:STDOUT:   %.loc14_37.4: ref %B = class_element_access %.loc14_37.3, element0
// CHECK:STDOUT:   %addr.loc14: %ptr.432 = addr_of %.loc14_37.4
// CHECK:STDOUT:   %.loc14_37.5: %ptr.432 = converted %B.as.X.impl.F.call, %addr.loc14
// CHECK:STDOUT:   %impl.elem0: %.fa1 = impl_witness_access constants.%Copy.impl_witness.09a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.70e]
// CHECK:STDOUT:   %bound_method.loc14_37.1: <bound method> = bound_method %.loc14_37.5, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%B) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_37.2: <bound method> = bound_method %.loc14_37.5, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.432 = call %bound_method.loc14_37.2(%.loc14_37.5)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_value_conversion_pointer_addr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %X.impl_witness: <witness> = impl_witness @B.as.X.impl.%X.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %self.param_patt.8fb: %pattern_type.9ef = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.680: %pattern_type.9ef = at_binding_pattern self, %self.param_patt.8fb [concrete]
// CHECK:STDOUT:   %ptr.d43: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.8a5: type = pattern_type %ptr.d43 [concrete]
// CHECK:STDOUT:   %other.param_patt.7dd: %pattern_type.8a5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %other.patt.95f: %pattern_type.8a5 = at_binding_pattern other, %other.param_patt.7dd [concrete]
// CHECK:STDOUT:   %ptr.6b6: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.8fa: Core.Form = init_form %ptr.6b6 [concrete]
// CHECK:STDOUT:   %pattern_type.fcb: type = pattern_type %ptr.6b6 [concrete]
// CHECK:STDOUT:   %return.param_patt.ada: %pattern_type.fcb = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.783: %pattern_type.fcb = return_slot_pattern %return.param_patt.ada, %ptr.6b6 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.48cdab.1: type = fn_type @B.as.X.impl.F.loc14_37.1 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.bd5be1.1: %B.as.X.impl.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %X.facet: %X.type = facet_value %B, (%X.impl_witness) [concrete]
// CHECK:STDOUT:   %ptr.432: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.48cdab.2: type = fn_type @B.as.X.impl.F.loc14_37.2 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.bd5be1.2: %B.as.X.impl.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.09a: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.d4b: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%B) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.70e: %ptr.as.Copy.impl.Op.type.d4b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.432, (%Copy.impl_witness.09a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.78a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.fa1: type = fn_type_with_self_type %Copy.WithSelf.Op.type.78a, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.70e, @ptr.as.Copy.impl.Op(%B) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.X.impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc14_37.1: %B.as.X.impl.F.type.48cdab.1 = fn_decl @B.as.X.impl.F.loc14_37.1 [concrete = constants.%B.as.X.impl.F.bd5be1.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9ef = ref_param_pattern [concrete = constants.%self.param_patt.8fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.680]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.8a5 = value_param_pattern [concrete = constants.%other.param_patt.7dd]
// CHECK:STDOUT:     %other.patt: %pattern_type.8a5 = at_binding_pattern other, %other.param_patt [concrete = constants.%other.patt.95f]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.fcb = out_param_pattern [concrete = constants.%return.param_patt.ada]
// CHECK:STDOUT:     %return.patt: %pattern_type.fcb = return_slot_pattern %return.param_patt, %ptr.loc14_36 [concrete = constants.%return.patt.783]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc14_36: type = ptr_type %C.ref [concrete = constants.%ptr.6b6]
// CHECK:STDOUT:     %.loc14_36: Core.Form = init_form %ptr.loc14_36 [concrete = constants.%.8fa]
// CHECK:STDOUT:     %self.param: ref %A = ref_param call_param0
// CHECK:STDOUT:     %A.ref.loc14_18: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %self: ref %A = wrapper_binding self, %self.param
// CHECK:STDOUT:     %other.param: %ptr.d43 = value_param call_param1
// CHECK:STDOUT:     %.loc14_29: type = splice_block %ptr.loc14_29 [concrete = constants.%ptr.d43] {
// CHECK:STDOUT:       %A.ref.loc14_28: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc14_29: type = ptr_type %A.ref.loc14_28 [concrete = constants.%ptr.d43]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: %ptr.d43 = wrapper_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref %ptr.6b6 = out_param call_param2
// CHECK:STDOUT:     %return: ref %ptr.6b6 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc14_37.2: %B.as.X.impl.F.type.48cdab.2 = fn_decl @B.as.X.impl.F.loc14_37.2 [concrete = constants.%B.as.X.impl.F.bd5be1.2] {
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
// CHECK:STDOUT:   extend %X.ref
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc14_37.1(%self.param: ref %A, %other.param: %ptr.d43) -> out %return.param: %ptr.6b6;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc14_37.2(%self.param: ref %B, %other.param: %ptr.432) -> out %return.param: %ptr.432 [thunk @B.as.X.impl.%B.as.X.impl.F.decl.loc14_37.1 for @X.WithSelf.%X.WithSelf.F.decl, @X.WithSelf.F(constants.%X.facet)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %B.as.X.impl.F.type.48cdab.1 = name_ref F, @B.as.X.impl.%B.as.X.impl.F.decl.loc14_37.1 [concrete = constants.%B.as.X.impl.F.bd5be1.1]
// CHECK:STDOUT:   %B.as.X.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.call: init %ptr.6b6 = call %B.as.X.impl.F.bound(%.6, %.9)
// CHECK:STDOUT:   %.loc14_37.1: %ptr.6b6 = value_of_initializer %B.as.X.impl.F.call
// CHECK:STDOUT:   %.loc14_37.2: %ptr.6b6 = converted %B.as.X.impl.F.call, %.loc14_37.1
// CHECK:STDOUT:   %.loc14_37.3: ref %C = deref %.loc14_37.2
// CHECK:STDOUT:   %.loc14_37.4: ref %B = class_element_access %.loc14_37.3, element0
// CHECK:STDOUT:   %addr.loc14: %ptr.432 = addr_of %.loc14_37.4
// CHECK:STDOUT:   %.loc14_37.5: %ptr.432 = converted %B.as.X.impl.F.call, %addr.loc14
// CHECK:STDOUT:   %impl.elem0: %.fa1 = impl_witness_access constants.%Copy.impl_witness.09a, element0 [concrete = constants.%ptr.as.Copy.impl.Op.70e]
// CHECK:STDOUT:   %bound_method.loc14_37.1: <bound method> = bound_method %.loc14_37.5, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%B) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_37.2: <bound method> = bound_method %.loc14_37.5, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.432 = call %bound_method.loc14_37.2(%.loc14_37.5)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_ref_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %X.impl_witness: <witness> = impl_witness @B.as.X.impl.%X.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %self.param_patt.04e: %pattern_type.9ef = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.312: %pattern_type.9ef = at_binding_pattern self, %self.param_patt.04e [concrete]
// CHECK:STDOUT:   %other.param_patt.c59: %pattern_type.9ef = ref_param_pattern [concrete]
// CHECK:STDOUT:   %other.patt.c26: %pattern_type.9ef = at_binding_pattern other, %other.param_patt.c59 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.48cdab.1: type = fn_type @B.as.X.impl.F.loc13_30.1 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.bd5be1.1: %B.as.X.impl.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %X.facet: %X.type = facet_value %B, (%X.impl_witness) [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.type.48cdab.2: type = fn_type @B.as.X.impl.F.loc13_30.2 [concrete]
// CHECK:STDOUT:   %B.as.X.impl.F.bd5be1.2: %B.as.X.impl.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @B.as.X.impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc13_30.1: %B.as.X.impl.F.type.48cdab.1 = fn_decl @B.as.X.impl.F.loc13_30.1 [concrete = constants.%B.as.X.impl.F.bd5be1.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9ef = value_param_pattern [concrete = constants.%self.param_patt.04e]
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.312]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.9ef = ref_param_pattern [concrete = constants.%other.param_patt.c59]
// CHECK:STDOUT:     %other.patt: %pattern_type.9ef = at_binding_pattern other, %other.param_patt [concrete = constants.%other.patt.c26]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref.loc13_14: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = wrapper_binding self, %self.param
// CHECK:STDOUT:     %other.param: ref %A = ref_param call_param1
// CHECK:STDOUT:     %A.ref.loc13_28: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %other: ref %A = wrapper_binding other, %other.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.as.X.impl.F.decl.loc13_30.2: %B.as.X.impl.F.type.48cdab.2 = fn_decl @B.as.X.impl.F.loc13_30.2 [concrete = constants.%B.as.X.impl.F.bd5be1.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .F = %B.as.X.impl.F.decl.loc13_30.1
// CHECK:STDOUT:   extend %X.ref
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc13_30.1(%self.param: %A, %other.param: ref %A);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.as.X.impl.F.loc13_30.2(%self.param: %B, %other.param: ref %B) [thunk @B.as.X.impl.%B.as.X.impl.F.decl.loc13_30.1 for @X.WithSelf.%X.WithSelf.F.decl, @X.WithSelf.F(constants.%X.facet)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %B.as.X.impl.F.type.48cdab.1 = name_ref F, @B.as.X.impl.%B.as.X.impl.F.decl.loc13_30.1 [concrete = constants.%B.as.X.impl.F.bd5be1.1]
// CHECK:STDOUT:   %B.as.X.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %B.as.X.impl.F.call: init %empty_tuple.type = call %B.as.X.impl.F.bound(%.6, %.9)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- form_parameter.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %empty_tuple.type, .b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.9f9: Core.Form = ref_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.param_patt.363cc0.1: %pattern_type.cb1 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.b18d0b.1: %pattern_type.cb1 = at_binding_pattern x, %x.param_patt.363cc0.1 [concrete]
// CHECK:STDOUT:   %struct_type.b.a: type = struct_type {.b: %empty_tuple.type, .a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.120: type = pattern_type %struct_type.b.a [concrete]
// CHECK:STDOUT:   %y.param_patt.618: %pattern_type.120 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.b7b: %pattern_type.120 = at_binding_pattern y, %y.param_patt.618 [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type.a24f58.1: type = fn_type @C.as.I.impl.F.loc11_48.1 [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.4c1488.1: %C.as.I.impl.F.type.a24f58.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.type.a24f58.2: type = fn_type @C.as.I.impl.F.loc11_48.2 [concrete]
// CHECK:STDOUT:   %C.as.I.impl.F.4c1488.2: %C.as.I.impl.F.type.a24f58.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %Self.ref as %.loc9_13 {
// CHECK:STDOUT:   %C.as.I.impl.F.decl.loc11_48.1: %C.as.I.impl.F.type.a24f58.1 = fn_decl @C.as.I.impl.F.loc11_48.1 [concrete = constants.%C.as.I.impl.F.4c1488.1] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = ref_param_pattern [concrete = constants.%x.param_patt.363cc0.1]
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.b18d0b.1]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.120 = value_param_pattern [concrete = constants.%y.param_patt.618]
// CHECK:STDOUT:     %y.patt: %pattern_type.120 = at_binding_pattern y, %y.param_patt [concrete = constants.%y.patt.b7b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: ref %empty_tuple.type = ref_param call_param0
// CHECK:STDOUT:     %.loc11_19.1: Core.Form = splice_block %.loc11_19.2 [concrete = constants.%.9f9] {
// CHECK:STDOUT:       %.loc11_24.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc11_24.2: type = converted %.loc11_24.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc11_19.2: Core.Form = ref_form %.loc11_24.2 [concrete = constants.%.9f9]
// CHECK:STDOUT:       %.loc11_19.3: type = type_component_of %.loc11_19.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %empty_tuple.type = wrapper_binding x, %x.param
// CHECK:STDOUT:     %y.param: %struct_type.b.a = value_param call_param1
// CHECK:STDOUT:     %.loc11_46: type = splice_block %struct_type.b.a [concrete = constants.%struct_type.b.a] {
// CHECK:STDOUT:       %.loc11_37.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc11_37.2: type = converted %.loc11_37.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc11_45.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc11_45.2: type = converted %.loc11_45.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.b.a: type = struct_type {.b: %empty_tuple.type, .a: %empty_tuple.type} [concrete = constants.%struct_type.b.a]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: %struct_type.b.a = wrapper_binding y, %y.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.as.I.impl.F.decl.loc11_48.2: %C.as.I.impl.F.type.a24f58.2 = fn_decl @C.as.I.impl.F.loc11_48.2 [concrete = constants.%C.as.I.impl.F.4c1488.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.I.impl.F.decl.loc11_48.1
// CHECK:STDOUT:   extend %.loc9_13
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.F.loc11_48.1(%x.param: ref %empty_tuple.type, %y.param: %struct_type.b.a);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.I.impl.F.loc11_48.2(%x.param: ref %empty_tuple.type, %y.param: %struct_type.a.b) [thunk @C.as.I.impl.%C.as.I.impl.F.decl.loc11_48.1 for @I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %C.as.I.impl.F.type.a24f58.1 = name_ref F, @C.as.I.impl.%C.as.I.impl.F.decl.loc11_48.1 [concrete = constants.%C.as.I.impl.F.4c1488.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.as.I.impl.F.call: init %empty_tuple.type = call %F.ref(%x.param, %.5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inheritance_value_conversion_copy_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %X.impl_witness: <witness> = impl_witness @A.as.X.impl.%X.impl_witness_table [concrete]
// CHECK:STDOUT:   %.d83: Core.Form = init_form %B [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %return.param_patt.934: %pattern_type.e39 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.911: %pattern_type.e39 = return_slot_pattern %return.param_patt.934, %B [concrete]
// CHECK:STDOUT:   %A.as.X.impl.F.type.48cdab.1: type = fn_type @A.as.X.impl.F.loc23_14.1 [concrete]
// CHECK:STDOUT:   %A.as.X.impl.F.bd5be1.1: %A.as.X.impl.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %X.facet: %X.type = facet_value %A, (%X.impl_witness) [concrete]
// CHECK:STDOUT:   %A.as.X.impl.F.type.48cdab.2: type = fn_type @A.as.X.impl.F.loc23_14.2 [concrete]
// CHECK:STDOUT:   %A.as.X.impl.F.bd5be1.2: %A.as.X.impl.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc23_14.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @A.as.X.impl: %A.ref as %X.ref {
// CHECK:STDOUT:   %A.as.X.impl.F.decl.loc23_14.1: %A.as.X.impl.F.type.48cdab.1 = fn_decl @A.as.X.impl.F.loc23_14.1 [concrete = constants.%A.as.X.impl.F.bd5be1.1] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e39 = out_param_pattern [concrete = constants.%return.param_patt.934]
// CHECK:STDOUT:     %return.patt: %pattern_type.e39 = return_slot_pattern %return.param_patt, %B.ref [concrete = constants.%return.patt.911]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc23: Core.Form = init_form %B.ref [concrete = constants.%.d83]
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %A.as.X.impl.F.decl.loc23_14.2: %A.as.X.impl.F.type.48cdab.2 = fn_decl @A.as.X.impl.F.loc23_14.2 [concrete = constants.%A.as.X.impl.F.bd5be1.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .F = %A.as.X.impl.F.decl.loc23_14.1
// CHECK:STDOUT:   extend %X.ref
// CHECK:STDOUT:   witness = %X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.X.impl.F.loc23_14.1() -> out %return.param: %B;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.as.X.impl.F.loc23_14.2() -> out %return.param: %A [thunk @A.as.X.impl.%A.as.X.impl.F.decl.loc23_14.1 for @X.WithSelf.%X.WithSelf.F.decl, @X.WithSelf.F(constants.%X.facet)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %A.as.X.impl.F.type.48cdab.1 = name_ref F, @A.as.X.impl.%A.as.X.impl.F.decl.loc23_14.1 [concrete = constants.%A.as.X.impl.F.bd5be1.1]
// CHECK:STDOUT:   %.loc23_14.1: ref %B = temporary_storage
// CHECK:STDOUT:   %A.as.X.impl.F.call: init %B to %.loc23_14.1 = call %F.ref()
// CHECK:STDOUT:   %.loc23_14.2: init %A = as_compatible %A.as.X.impl.F.call
// CHECK:STDOUT:   %.loc23_14.3: ref %A = temporary %.loc23_14.1, %.loc23_14.2
// CHECK:STDOUT:   %.loc23_14.4: ref %A = class_element_access %.loc23_14.3, element0
// CHECK:STDOUT:   %.loc23_14.5: ref %A = converted %A.as.X.impl.F.call, %.loc23_14.4
// CHECK:STDOUT:   %.loc23_14.6: %A = acquire_value %.loc23_14.5
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc23_14.3, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc23_14.3)
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_14.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_14.2(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @empty_tuple.type.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %U.patt.fe6: %pattern_type.e81 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.f84: %Copy.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type.c1c: type = facet_access_type %U.f84 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %x.param_patt.1f4: %pattern_type.2c66b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.f93: %pattern_type.2c66b4.2 = at_binding_pattern x, %x.param_patt.1f4 [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %return.param_patt.bf4475.2: %pattern_type.2c66b4.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f44: %pattern_type.2c66b4.2 = return_slot_pattern %return.param_patt.bf4475.2, %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.48cdab.1: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_34.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.bd5be1.1: %empty_tuple.type.as.I.impl.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_tuple.type, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %x.param_patt.b5c: %pattern_type.4f4 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.b28: %pattern_type.4f4 = at_binding_pattern x, %x.param_patt.b5c [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [symbolic]
// CHECK:STDOUT:   %return.param_patt.27f: %pattern_type.4f4 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.d67: %pattern_type.4f4 = return_slot_pattern %return.param_patt.27f, %ptr.e8f [symbolic]
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.48cdab.2: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_34.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.bd5be1.2: %empty_tuple.type.as.I.impl.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d83f06.1: <witness> = require_complete_type %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c332.2: <witness> = lookup_impl_witness %U.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c85f21.3: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%U.f84) [symbolic]
// CHECK:STDOUT:   %.c29330.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c85f21.3, %U.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bd2456.2: %.c29330.2 = impl_witness_access %Copy.lookup_impl_witness.d9c332.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e2af7e.2: <specific function> = specific_impl_function %impl.elem0.bd2456.2, @Copy.WithSelf.Op(%U.f84) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.1da: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %.0e9: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.1da) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.specific_fn: <specific function> = specific_function %empty_tuple.type.as.I.impl.F.bd5be1.1, @empty_tuple.type.as.I.impl.F.loc10_34.1(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.884: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.be5: type = fn_type_with_self_type %Copy.WithSelf.Op.type.884, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.484: %.be5 = impl_witness_access %Copy.lookup_impl_witness.1da, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a76: <specific function> = specific_impl_function %impl.elem0.484, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_34.1: %empty_tuple.type.as.I.impl.F.type.48cdab.1 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_34.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.bd5be1.1] {
// CHECK:STDOUT:     %U.patt.loc10_9.1: %pattern_type.e81 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc10_9.2 (constants.%U.patt.fe6)]
// CHECK:STDOUT:     %x.param_patt.loc10_24.1: @empty_tuple.type.as.I.impl.F.loc10_34.1.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %x.param_patt.loc10_24.2 (constants.%x.param_patt.1f4)]
// CHECK:STDOUT:     %x.patt.loc10_24.1: @empty_tuple.type.as.I.impl.F.loc10_34.1.%pattern_type (%pattern_type.2c66b4.2) = at_binding_pattern x, %x.param_patt.loc10_24.1 [symbolic = %x.patt.loc10_24.2 (constants.%x.patt.f93)]
// CHECK:STDOUT:     %return.param_patt.loc10_32.1: @empty_tuple.type.as.I.impl.F.loc10_34.1.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc10_32.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:     %return.patt.loc10_29.1: @empty_tuple.type.as.I.impl.F.loc10_34.1.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc10_32.1, %.loc10_32.3 [symbolic = %return.patt.loc10_29.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc10_32: %Copy.type = name_ref U, %U.loc10_9.2 [symbolic = %U.loc10_9.1 (constants.%U.f84)]
// CHECK:STDOUT:     %U.as_type.loc10_32: type = facet_access_type %U.ref.loc10_32 [symbolic = %U.as_type.loc10_26.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:     %.loc10_32.3: type = converted %U.ref.loc10_32, %U.as_type.loc10_32 [symbolic = %U.as_type.loc10_26.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:     %.loc10_32.4: Core.Form = init_form %.loc10_32.3 [symbolic = %.loc10_32.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:     %.loc10_16: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc10_9.2: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc10_9.1 (constants.%U.f84)]
// CHECK:STDOUT:     %x.param: @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.as_type.loc10_26.1 (%U.as_type.c1c) = value_param call_param0
// CHECK:STDOUT:     %.loc10_26.1: type = splice_block %.loc10_26.2 [symbolic = %U.as_type.loc10_26.1 (constants.%U.as_type.c1c)] {
// CHECK:STDOUT:       %U.ref.loc10_26: %Copy.type = name_ref U, %U.loc10_9.2 [symbolic = %U.loc10_9.1 (constants.%U.f84)]
// CHECK:STDOUT:       %U.as_type.loc10_26.2: type = facet_access_type %U.ref.loc10_26 [symbolic = %U.as_type.loc10_26.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:       %.loc10_26.2: type = converted %U.ref.loc10_26, %U.as_type.loc10_26.2 [symbolic = %U.as_type.loc10_26.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.as_type.loc10_26.1 (%U.as_type.c1c) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.as_type.loc10_26.1 (%U.as_type.c1c) = out_param call_param1
// CHECK:STDOUT:     %return: ref @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.as_type.loc10_26.1 (%U.as_type.c1c) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_34.2: %empty_tuple.type.as.I.impl.F.type.48cdab.2 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_34.2 [concrete = constants.%empty_tuple.type.as.I.impl.F.bd5be1.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl.loc10_34.1
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_tuple.type.as.I.impl.F.loc10_34.1(%U.loc10_9.2: %Copy.type) {
// CHECK:STDOUT:   %U.patt.loc10_9.2: %pattern_type.e81 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc10_9.2 (constants.%U.patt.fe6)]
// CHECK:STDOUT:   %U.loc10_9.1: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc10_9.1 (constants.%U.f84)]
// CHECK:STDOUT:   %U.as_type.loc10_26.1: type = facet_access_type %U.loc10_9.1 [symbolic = %U.as_type.loc10_26.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.as_type.loc10_26.1 [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %x.param_patt.loc10_24.2: @empty_tuple.type.as.I.impl.F.loc10_34.1.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %x.param_patt.loc10_24.2 (constants.%x.param_patt.1f4)]
// CHECK:STDOUT:   %x.patt.loc10_24.2: @empty_tuple.type.as.I.impl.F.loc10_34.1.%pattern_type (%pattern_type.2c66b4.2) = at_binding_pattern x, %x.param_patt.loc10_24.2 [symbolic = %x.patt.loc10_24.2 (constants.%x.patt.f93)]
// CHECK:STDOUT:   %.loc10_32.2: Core.Form = init_form %U.as_type.loc10_26.1 [symbolic = %.loc10_32.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:   %return.param_patt.loc10_32.2: @empty_tuple.type.as.I.impl.F.loc10_34.1.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc10_32.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:   %return.patt.loc10_29.2: @empty_tuple.type.as.I.impl.F.loc10_34.1.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc10_32.2, %U.as_type.loc10_26.1 [symbolic = %return.patt.loc10_29.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.as_type.loc10_26.1 [symbolic = %require_complete (constants.%require_complete.d83f06.1)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%U.loc10_9.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c85f21.3)]
// CHECK:STDOUT:   %.loc10_43.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %U.loc10_9.1 [symbolic = %.loc10_43.3 (constants.%.c29330.2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc10_9.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c332.2)]
// CHECK:STDOUT:   %impl.elem0.loc10_43.2: @empty_tuple.type.as.I.impl.F.loc10_34.1.%.loc10_43.3 (%.c29330.2) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_43.2 (constants.%impl.elem0.bd2456.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_43.2: <specific function> = specific_impl_function %impl.elem0.loc10_43.2, @Copy.WithSelf.Op(%U.loc10_9.1) [symbolic = %specific_impl_fn.loc10_43.2 (constants.%specific_impl_fn.e2af7e.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.as_type.loc10_26.1 (%U.as_type.c1c)) -> out %return.param: @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.as_type.loc10_26.1 (%U.as_type.c1c) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.as_type.loc10_26.1 (%U.as_type.c1c) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc10_43.1: @empty_tuple.type.as.I.impl.F.loc10_34.1.%.loc10_43.3 (%.c29330.2) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c332.2, element0 [symbolic = %impl.elem0.loc10_43.2 (constants.%impl.elem0.bd2456.2)]
// CHECK:STDOUT:     %bound_method.loc10_43.1: <bound method> = bound_method %x.ref, %impl.elem0.loc10_43.1
// CHECK:STDOUT:     %.loc10_43.1: %Copy.type = converted constants.%U.as_type.c1c, constants.%U.f84 [symbolic = %U.loc10_9.1 (constants.%U.f84)]
// CHECK:STDOUT:     %.loc10_43.2: %Copy.type = converted constants.%U.as_type.c1c, constants.%U.f84 [symbolic = %U.loc10_9.1 (constants.%U.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_43.1: <specific function> = specific_impl_function %impl.elem0.loc10_43.1, @Copy.WithSelf.Op(constants.%U.f84) [symbolic = %specific_impl_fn.loc10_43.2 (constants.%specific_impl_fn.e2af7e.2)]
// CHECK:STDOUT:     %bound_method.loc10_43.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_43.1
// CHECK:STDOUT:     %.loc10_32.1: ref @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.as_type.loc10_26.1 (%U.as_type.c1c) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @empty_tuple.type.as.I.impl.F.loc10_34.1.%U.as_type.loc10_26.1 (%U.as_type.c1c) to %.loc10_32.1 = call %bound_method.loc10_43.2(%x.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_tuple.type.as.I.impl.F.loc10_34.2(%T.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_34.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.1) [symbolic = %.loc10_34.3 (constants.%.0e9)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.1da)]
// CHECK:STDOUT:   %Copy.facet.loc10_34.3: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc10_34.3 (constants.%Copy.facet)]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.specific_fn.loc10_34.2: <specific function> = specific_function constants.%empty_tuple.type.as.I.impl.F.bd5be1.1, @empty_tuple.type.as.I.impl.F.loc10_34.1(%Copy.facet.loc10_34.3) [symbolic = %empty_tuple.type.as.I.impl.F.specific_fn.loc10_34.2 (constants.%empty_tuple.type.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @empty_tuple.type.as.I.impl.F.loc10_34.2.%ptr (%ptr.e8f)) -> out %return.param: @empty_tuple.type.as.I.impl.F.loc10_34.2.%ptr (%ptr.e8f) [thunk @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_34.1 for @I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet, constants.%T.67d)] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %empty_tuple.type.as.I.impl.F.type.48cdab.1 = name_ref F, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_34.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.bd5be1.1]
// CHECK:STDOUT:     %Copy.facet.loc10_34.1: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc10_34.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc10_34.1: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc10_34.1 [symbolic = %Copy.facet.loc10_34.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %Copy.facet.loc10_34.2: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc10_34.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc10_34.2: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc10_34.2 [symbolic = %Copy.facet.loc10_34.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.specific_fn.loc10_34.1: <specific function> = specific_function %F.ref, @empty_tuple.type.as.I.impl.F.loc10_34.1(constants.%Copy.facet) [symbolic = %empty_tuple.type.as.I.impl.F.specific_fn.loc10_34.2 (constants.%empty_tuple.type.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.call: init @empty_tuple.type.as.I.impl.F.loc10_34.2.%ptr (%ptr.e8f) = call %empty_tuple.type.as.I.impl.F.specific_fn.loc10_34.1(%x.param)
// CHECK:STDOUT:     return %empty_tuple.type.as.I.impl.F.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc10_34.1(constants.%U.f84) {
// CHECK:STDOUT:   %U.patt.loc10_9.2 => constants.%U.patt.fe6
// CHECK:STDOUT:   %U.loc10_9.1 => constants.%U.f84
// CHECK:STDOUT:   %U.as_type.loc10_26.1 => constants.%U.as_type.c1c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %x.param_patt.loc10_24.2 => constants.%x.param_patt.1f4
// CHECK:STDOUT:   %x.patt.loc10_24.2 => constants.%x.patt.f93
// CHECK:STDOUT:   %.loc10_32.2 => constants.%.1ce700.2
// CHECK:STDOUT:   %return.param_patt.loc10_32.2 => constants.%return.param_patt.bf4475.2
// CHECK:STDOUT:   %return.patt.loc10_29.2 => constants.%return.patt.f44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc10_34.2(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.1 => constants.%T.67d
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %x.param_patt => constants.%x.param_patt.b5c
// CHECK:STDOUT:   %x.patt.2 => constants.%x.patt.b28
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.27f
// CHECK:STDOUT:   %return.patt => constants.%return.patt.d67
// CHECK:STDOUT:   %.loc5 => constants.%.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc10_34.1(constants.%Copy.facet) {
// CHECK:STDOUT:   %U.patt.loc10_9.2 => constants.%U.patt.fe6
// CHECK:STDOUT:   %U.loc10_9.1 => constants.%Copy.facet
// CHECK:STDOUT:   %U.as_type.loc10_26.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %x.param_patt.loc10_24.2 => constants.%x.param_patt.b5c
// CHECK:STDOUT:   %x.patt.loc10_24.2 => constants.%x.patt.b28
// CHECK:STDOUT:   %.loc10_32.2 => constants.%.cb6
// CHECK:STDOUT:   %return.param_patt.loc10_32.2 => constants.%return.param_patt.27f
// CHECK:STDOUT:   %return.patt.loc10_29.2 => constants.%return.patt.d67
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ef1
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.884
// CHECK:STDOUT:   %.loc10_43.3 => constants.%.be5
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.1da
// CHECK:STDOUT:   %impl.elem0.loc10_43.2 => constants.%impl.elem0.484
// CHECK:STDOUT:   %specific_impl_fn.loc10_43.2 => constants.%specific_impl_fn.a76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @empty_tuple.type.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %U.patt.fe6: %pattern_type.e81 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.f84: %Copy.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %self.param_patt.154: %pattern_type.cb1 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.8c1: %pattern_type.cb1 = at_binding_pattern self, %self.param_patt.154 [concrete]
// CHECK:STDOUT:   %U.as_type.c1c: type = facet_access_type %U.f84 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %x.param_patt.1f4: %pattern_type.2c66b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.f93: %pattern_type.2c66b4.2 = at_binding_pattern x, %x.param_patt.1f4 [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %return.param_patt.bf4475.2: %pattern_type.2c66b4.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f44: %pattern_type.2c66b4.2 = return_slot_pattern %return.param_patt.bf4475.2, %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.48cdab.1: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_51.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.bd5be1.1: %empty_tuple.type.as.I.impl.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_tuple.type, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %x.param_patt.b5c: %pattern_type.4f4 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.b28: %pattern_type.4f4 = at_binding_pattern x, %x.param_patt.b5c [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [symbolic]
// CHECK:STDOUT:   %return.param_patt.27f: %pattern_type.4f4 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.d67: %pattern_type.4f4 = return_slot_pattern %return.param_patt.27f, %ptr.e8f [symbolic]
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.48cdab.2: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_51.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.bd5be1.2: %empty_tuple.type.as.I.impl.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d83f06.1: <witness> = require_complete_type %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c332.2: <witness> = lookup_impl_witness %U.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c85f21.3: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%U.f84) [symbolic]
// CHECK:STDOUT:   %.c29330.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c85f21.3, %U.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bd2456.2: %.c29330.2 = impl_witness_access %Copy.lookup_impl_witness.d9c332.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e2af7e.2: <specific function> = specific_impl_function %impl.elem0.bd2456.2, @Copy.WithSelf.Op(%U.f84) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.1da: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %.0e9: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.1da) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.specific_fn: <specific function> = specific_function %empty_tuple.type.as.I.impl.F.bd5be1.1, @empty_tuple.type.as.I.impl.F.loc10_51.1(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.884: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.be5: type = fn_type_with_self_type %Copy.WithSelf.Op.type.884, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.484: %.be5 = impl_witness_access %Copy.lookup_impl_witness.1da, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a76: <specific function> = specific_impl_function %impl.elem0.484, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_51.1: %empty_tuple.type.as.I.impl.F.type.48cdab.1 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_51.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.bd5be1.1] {
// CHECK:STDOUT:     %U.patt.loc10_9.1: %pattern_type.e81 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc10_9.2 (constants.%U.patt.fe6)]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = value_param_pattern [concrete = constants.%self.param_patt.154]
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.8c1]
// CHECK:STDOUT:     %x.param_patt.loc10_41.1: @empty_tuple.type.as.I.impl.F.loc10_51.1.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %x.param_patt.loc10_41.2 (constants.%x.param_patt.1f4)]
// CHECK:STDOUT:     %x.patt.loc10_41.1: @empty_tuple.type.as.I.impl.F.loc10_51.1.%pattern_type (%pattern_type.2c66b4.2) = at_binding_pattern x, %x.param_patt.loc10_41.1 [symbolic = %x.patt.loc10_41.2 (constants.%x.patt.f93)]
// CHECK:STDOUT:     %return.param_patt.loc10_49.1: @empty_tuple.type.as.I.impl.F.loc10_51.1.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc10_49.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:     %return.patt.loc10_46.1: @empty_tuple.type.as.I.impl.F.loc10_51.1.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc10_49.1, %.loc10_49.3 [symbolic = %return.patt.loc10_46.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc10_49: %Copy.type = name_ref U, %U.loc10_9.2 [symbolic = %U.loc10_9.1 (constants.%U.f84)]
// CHECK:STDOUT:     %U.as_type.loc10_49: type = facet_access_type %U.ref.loc10_49 [symbolic = %U.as_type.loc10_43.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:     %.loc10_49.3: type = converted %U.ref.loc10_49, %U.as_type.loc10_49 [symbolic = %U.as_type.loc10_43.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:     %.loc10_49.4: Core.Form = init_form %.loc10_49.3 [symbolic = %.loc10_49.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:     %.loc10_16: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc10_9.2: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc10_9.1 (constants.%U.f84)]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc10_37.1: type = splice_block %.loc10_37.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc10_37.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc10_37.3: type = converted %.loc10_37.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %empty_tuple.type = wrapper_binding self, %self.param
// CHECK:STDOUT:     %x.param: @empty_tuple.type.as.I.impl.F.loc10_51.1.%U.as_type.loc10_43.1 (%U.as_type.c1c) = value_param call_param1
// CHECK:STDOUT:     %.loc10_43.1: type = splice_block %.loc10_43.2 [symbolic = %U.as_type.loc10_43.1 (constants.%U.as_type.c1c)] {
// CHECK:STDOUT:       %U.ref.loc10_43: %Copy.type = name_ref U, %U.loc10_9.2 [symbolic = %U.loc10_9.1 (constants.%U.f84)]
// CHECK:STDOUT:       %U.as_type.loc10_43.2: type = facet_access_type %U.ref.loc10_43 [symbolic = %U.as_type.loc10_43.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:       %.loc10_43.2: type = converted %U.ref.loc10_43, %U.as_type.loc10_43.2 [symbolic = %U.as_type.loc10_43.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @empty_tuple.type.as.I.impl.F.loc10_51.1.%U.as_type.loc10_43.1 (%U.as_type.c1c) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @empty_tuple.type.as.I.impl.F.loc10_51.1.%U.as_type.loc10_43.1 (%U.as_type.c1c) = out_param call_param2
// CHECK:STDOUT:     %return: ref @empty_tuple.type.as.I.impl.F.loc10_51.1.%U.as_type.loc10_43.1 (%U.as_type.c1c) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_51.2: %empty_tuple.type.as.I.impl.F.type.48cdab.2 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_51.2 [concrete = constants.%empty_tuple.type.as.I.impl.F.bd5be1.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl.loc10_51.1
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_tuple.type.as.I.impl.F.loc10_51.1(%U.loc10_9.2: %Copy.type) {
// CHECK:STDOUT:   %U.patt.loc10_9.2: %pattern_type.e81 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc10_9.2 (constants.%U.patt.fe6)]
// CHECK:STDOUT:   %U.loc10_9.1: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc10_9.1 (constants.%U.f84)]
// CHECK:STDOUT:   %U.as_type.loc10_43.1: type = facet_access_type %U.loc10_9.1 [symbolic = %U.as_type.loc10_43.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.as_type.loc10_43.1 [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %x.param_patt.loc10_41.2: @empty_tuple.type.as.I.impl.F.loc10_51.1.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %x.param_patt.loc10_41.2 (constants.%x.param_patt.1f4)]
// CHECK:STDOUT:   %x.patt.loc10_41.2: @empty_tuple.type.as.I.impl.F.loc10_51.1.%pattern_type (%pattern_type.2c66b4.2) = at_binding_pattern x, %x.param_patt.loc10_41.2 [symbolic = %x.patt.loc10_41.2 (constants.%x.patt.f93)]
// CHECK:STDOUT:   %.loc10_49.2: Core.Form = init_form %U.as_type.loc10_43.1 [symbolic = %.loc10_49.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:   %return.param_patt.loc10_49.2: @empty_tuple.type.as.I.impl.F.loc10_51.1.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc10_49.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:   %return.patt.loc10_46.2: @empty_tuple.type.as.I.impl.F.loc10_51.1.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc10_49.2, %U.as_type.loc10_43.1 [symbolic = %return.patt.loc10_46.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.as_type.loc10_43.1 [symbolic = %require_complete (constants.%require_complete.d83f06.1)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%U.loc10_9.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c85f21.3)]
// CHECK:STDOUT:   %.loc10_60.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %U.loc10_9.1 [symbolic = %.loc10_60.3 (constants.%.c29330.2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc10_9.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c332.2)]
// CHECK:STDOUT:   %impl.elem0.loc10_60.2: @empty_tuple.type.as.I.impl.F.loc10_51.1.%.loc10_60.3 (%.c29330.2) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_60.2 (constants.%impl.elem0.bd2456.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_60.2: <specific function> = specific_impl_function %impl.elem0.loc10_60.2, @Copy.WithSelf.Op(%U.loc10_9.1) [symbolic = %specific_impl_fn.loc10_60.2 (constants.%specific_impl_fn.e2af7e.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: %empty_tuple.type, %x.param: @empty_tuple.type.as.I.impl.F.loc10_51.1.%U.as_type.loc10_43.1 (%U.as_type.c1c)) -> out %return.param: @empty_tuple.type.as.I.impl.F.loc10_51.1.%U.as_type.loc10_43.1 (%U.as_type.c1c) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @empty_tuple.type.as.I.impl.F.loc10_51.1.%U.as_type.loc10_43.1 (%U.as_type.c1c) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc10_60.1: @empty_tuple.type.as.I.impl.F.loc10_51.1.%.loc10_60.3 (%.c29330.2) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c332.2, element0 [symbolic = %impl.elem0.loc10_60.2 (constants.%impl.elem0.bd2456.2)]
// CHECK:STDOUT:     %bound_method.loc10_60.1: <bound method> = bound_method %x.ref, %impl.elem0.loc10_60.1
// CHECK:STDOUT:     %.loc10_60.1: %Copy.type = converted constants.%U.as_type.c1c, constants.%U.f84 [symbolic = %U.loc10_9.1 (constants.%U.f84)]
// CHECK:STDOUT:     %.loc10_60.2: %Copy.type = converted constants.%U.as_type.c1c, constants.%U.f84 [symbolic = %U.loc10_9.1 (constants.%U.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_60.1: <specific function> = specific_impl_function %impl.elem0.loc10_60.1, @Copy.WithSelf.Op(constants.%U.f84) [symbolic = %specific_impl_fn.loc10_60.2 (constants.%specific_impl_fn.e2af7e.2)]
// CHECK:STDOUT:     %bound_method.loc10_60.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_60.1
// CHECK:STDOUT:     %.loc10_49.1: ref @empty_tuple.type.as.I.impl.F.loc10_51.1.%U.as_type.loc10_43.1 (%U.as_type.c1c) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @empty_tuple.type.as.I.impl.F.loc10_51.1.%U.as_type.loc10_43.1 (%U.as_type.c1c) to %.loc10_49.1 = call %bound_method.loc10_60.2(%x.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @empty_tuple.type.as.I.impl.F.loc10_51.2(%T.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_51.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.1) [symbolic = %.loc10_51.3 (constants.%.0e9)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.1da)]
// CHECK:STDOUT:   %Copy.facet.loc10_51.3: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc10_51.3 (constants.%Copy.facet)]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.specific_fn.loc10_51.2: <specific function> = specific_function constants.%empty_tuple.type.as.I.impl.F.bd5be1.1, @empty_tuple.type.as.I.impl.F.loc10_51.1(%Copy.facet.loc10_51.3) [symbolic = %empty_tuple.type.as.I.impl.F.specific_fn.loc10_51.2 (constants.%empty_tuple.type.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: %empty_tuple.type, %x.param: @empty_tuple.type.as.I.impl.F.loc10_51.2.%ptr (%ptr.e8f)) -> out %return.param: @empty_tuple.type.as.I.impl.F.loc10_51.2.%ptr (%ptr.e8f) [thunk @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_51.1 for @I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%I.facet, constants.%T.67d)] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %empty_tuple.type.as.I.impl.F.type.48cdab.1 = name_ref F, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_51.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.bd5be1.1]
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.bound: <bound method> = bound_method %self.param, %F.ref
// CHECK:STDOUT:     %Copy.facet.loc10_51.1: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc10_51.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc10_51.1: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc10_51.1 [symbolic = %Copy.facet.loc10_51.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %Copy.facet.loc10_51.2: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc10_51.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc10_51.2: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc10_51.2 [symbolic = %Copy.facet.loc10_51.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.specific_fn.loc10_51.1: <specific function> = specific_function %F.ref, @empty_tuple.type.as.I.impl.F.loc10_51.1(constants.%Copy.facet) [symbolic = %empty_tuple.type.as.I.impl.F.specific_fn.loc10_51.2 (constants.%empty_tuple.type.as.I.impl.F.specific_fn)]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %self.param, %empty_tuple.type.as.I.impl.F.specific_fn.loc10_51.1
// CHECK:STDOUT:     %empty_tuple.type.as.I.impl.F.call: init @empty_tuple.type.as.I.impl.F.loc10_51.2.%ptr (%ptr.e8f) = call %bound_method(%self.param, %x.param)
// CHECK:STDOUT:     return %empty_tuple.type.as.I.impl.F.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc10_51.1(constants.%U.f84) {
// CHECK:STDOUT:   %U.patt.loc10_9.2 => constants.%U.patt.fe6
// CHECK:STDOUT:   %U.loc10_9.1 => constants.%U.f84
// CHECK:STDOUT:   %U.as_type.loc10_43.1 => constants.%U.as_type.c1c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %x.param_patt.loc10_41.2 => constants.%x.param_patt.1f4
// CHECK:STDOUT:   %x.patt.loc10_41.2 => constants.%x.patt.f93
// CHECK:STDOUT:   %.loc10_49.2 => constants.%.1ce700.2
// CHECK:STDOUT:   %return.param_patt.loc10_49.2 => constants.%return.param_patt.bf4475.2
// CHECK:STDOUT:   %return.patt.loc10_46.2 => constants.%return.patt.f44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc10_51.2(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.1 => constants.%T.67d
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %x.param_patt => constants.%x.param_patt.b5c
// CHECK:STDOUT:   %x.patt.2 => constants.%x.patt.b28
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.27f
// CHECK:STDOUT:   %return.patt => constants.%return.patt.d67
// CHECK:STDOUT:   %.loc5 => constants.%.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @empty_tuple.type.as.I.impl.F.loc10_51.1(constants.%Copy.facet) {
// CHECK:STDOUT:   %U.patt.loc10_9.2 => constants.%U.patt.fe6
// CHECK:STDOUT:   %U.loc10_9.1 => constants.%Copy.facet
// CHECK:STDOUT:   %U.as_type.loc10_43.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %x.param_patt.loc10_41.2 => constants.%x.param_patt.b5c
// CHECK:STDOUT:   %x.patt.loc10_41.2 => constants.%x.patt.b28
// CHECK:STDOUT:   %.loc10_49.2 => constants.%.cb6
// CHECK:STDOUT:   %return.param_patt.loc10_49.2 => constants.%return.param_patt.27f
// CHECK:STDOUT:   %return.patt.loc10_46.2 => constants.%return.patt.d67
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ef1
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.884
// CHECK:STDOUT:   %.loc10_60.3 => constants.%.be5
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.1da
// CHECK:STDOUT:   %impl.elem0.loc10_60.2 => constants.%impl.elem0.484
// CHECK:STDOUT:   %specific_impl_fn.loc10_60.2 => constants.%specific_impl_fn.a76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.81d: type = facet_type <@I, @I(%empty_struct_type)> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @empty_tuple.type.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %struct_type.b.a.1b0: type = struct_type {.b: %empty_struct_type, .a: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %.3fc: Core.Form = init_form %struct_type.b.a.1b0 [concrete]
// CHECK:STDOUT:   %pattern_type.914: type = pattern_type %struct_type.b.a.1b0 [concrete]
// CHECK:STDOUT:   %return.param_patt.2c3: %pattern_type.914 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.5c9: %pattern_type.914 = return_slot_pattern %return.param_patt.2c3, %struct_type.b.a.1b0 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.48cdab.1: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_29.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.bd5be1.1: %empty_tuple.type.as.I.impl.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.81d = facet_value %empty_tuple.type, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.f95: type = struct_type {.a: %empty_struct_type, .b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.type.48cdab.2: type = fn_type @empty_tuple.type.as.I.impl.F.loc10_29.2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.bd5be1.2: %empty_tuple.type.as.I.impl.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.f95 = struct_value (%empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc10_29.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @empty_tuple.type.as.I.impl: %.loc8_7.2 as %I.type {
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_29.1: %empty_tuple.type.as.I.impl.F.type.48cdab.1 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_29.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.bd5be1.1] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.914 = out_param_pattern [concrete = constants.%return.param_patt.2c3]
// CHECK:STDOUT:     %return.patt: %pattern_type.914 = return_slot_pattern %return.param_patt, %struct_type.b.a [concrete = constants.%return.patt.5c9]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_19.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc10_19.2: type = converted %.loc10_19.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc10_27.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc10_27.2: type = converted %.loc10_27.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.b.a: type = struct_type {.b: %empty_struct_type, .a: %empty_struct_type} [concrete = constants.%struct_type.b.a.1b0]
// CHECK:STDOUT:     %.loc10_28: Core.Form = init_form %struct_type.b.a [concrete = constants.%.3fc]
// CHECK:STDOUT:     %return.param: ref %struct_type.b.a.1b0 = out_param call_param0
// CHECK:STDOUT:     %return: ref %struct_type.b.a.1b0 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.decl.loc10_29.2: %empty_tuple.type.as.I.impl.F.type.48cdab.2 = fn_decl @empty_tuple.type.as.I.impl.F.loc10_29.2 [concrete = constants.%empty_tuple.type.as.I.impl.F.bd5be1.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %empty_tuple.type.as.I.impl.F.decl.loc10_29.1
// CHECK:STDOUT:   extend %I.type
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F.loc10_29.1() -> out %return.param: %struct_type.b.a.1b0;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @empty_tuple.type.as.I.impl.F.loc10_29.2() -> out %return.param: %struct_type.a.b.f95 [thunk @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_29.1 for @I.WithSelf.%I.WithSelf.F.decl, @I.WithSelf.F(constants.%empty_struct_type, constants.%I.facet)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %empty_tuple.type.as.I.impl.F.type.48cdab.1 = name_ref F, @empty_tuple.type.as.I.impl.%empty_tuple.type.as.I.impl.F.decl.loc10_29.1 [concrete = constants.%empty_tuple.type.as.I.impl.F.bd5be1.1]
// CHECK:STDOUT:   %.loc10_29.1: ref %struct_type.b.a.1b0 = temporary_storage
// CHECK:STDOUT:   %empty_tuple.type.as.I.impl.F.call: init %struct_type.b.a.1b0 to %.loc10_29.1 = call %F.ref()
// CHECK:STDOUT:   %.loc10_29.2: ref %struct_type.b.a.1b0 = temporary %.loc10_29.1, %empty_tuple.type.as.I.impl.F.call
// CHECK:STDOUT:   %.loc10_29.3: ref %empty_struct_type = struct_access %.loc10_29.2, element1
// CHECK:STDOUT:   %.loc10_29.4: ref %empty_struct_type = struct_access %return.param, element1
// CHECK:STDOUT:   %.loc10_29.5: init %empty_struct_type to %.loc10_29.4 = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_29.6: init %empty_struct_type = converted %.loc10_29.3, %.loc10_29.5 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_29.7: init %empty_struct_type to %.loc10_29.4 = in_place_init %.loc10_29.6 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_29.8: ref %empty_struct_type = struct_access %.loc10_29.2, element0
// CHECK:STDOUT:   %.loc10_29.9: ref %empty_struct_type = struct_access %return.param, element0
// CHECK:STDOUT:   %.loc10_29.10: init %empty_struct_type to %.loc10_29.9 = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_29.11: init %empty_struct_type = converted %.loc10_29.8, %.loc10_29.10 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_29.12: init %empty_struct_type to %.loc10_29.9 = in_place_init %.loc10_29.11 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_29.13: init %struct_type.a.b.f95 to %return.param = struct_init (%.loc10_29.7, %.loc10_29.12) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc10_29.14: init %struct_type.a.b.f95 = converted %empty_tuple.type.as.I.impl.F.call, %.loc10_29.13 [concrete = constants.%struct]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc10_29.2, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc10_29.2)
// CHECK:STDOUT:   return %.loc10_29.14 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_29.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_29.2(%self.param: ref %struct_type.b.a.1b0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
