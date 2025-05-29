// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/impl_thunk.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/impl_thunk.carbon

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
  fn F(y: {.b: {}, .a: ()}) -> {.d: {}, .c: ()};
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
  fn F[addr self: A*](other: A*) -> C*;
}

// --- inheritance_value_conversion.carbon

library "[[@TEST_NAME]]";

base class A {}
class B { extend base: A; }

interface X {
  fn F[self: Self](other: Self);
}

impl B as X {
  fn F[self: A](other: A);
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
  fn F[addr self: A*](other: A*) -> C*;
}

// --- fail_inheritance_value_conversion_copy_return.carbon

library "[[@TEST_NAME]]";

base class A {}
class B { extend base: A; }

interface X {
  fn F() -> Self;
}

impl A as X {
  // CHECK:STDERR: fail_inheritance_value_conversion_copy_return.carbon:[[@LINE+7]]:3: error: cannot copy value of type `A` [CopyOfUncopyableType]
  // CHECK:STDERR:   fn F() -> B;
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_inheritance_value_conversion_copy_return.carbon:[[@LINE-7]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn F() -> Self;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F() -> B;
}

// --- fail_param_type_mismatch.carbon

library "[[@TEST_NAME]]";

interface I {
  // CHECK:STDERR: fail_param_type_mismatch.carbon:[[@LINE+3]]:8: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
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
  // CHECK:STDERR: fail_return_mismatch.carbon:[[@LINE+7]]:3: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
  // CHECK:STDERR:   fn F() -> B;
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_return_mismatch.carbon:[[@LINE-10]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
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
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE+14]]:5: note: while building thunk calling this function [ThunkCallee]
    // CHECK:STDERR:     fn F(c: C);
    // CHECK:STDERR:     ^~~~~~~~~~~
    // CHECK:STDERR:
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE-14]]:8: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
    // CHECK:STDERR:   fn F(a: T);
    // CHECK:STDERR:        ^~~~
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE+7]]:10: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR:     fn F(c: C);
    // CHECK:STDERR:          ^~~~
    // CHECK:STDERR: fail_param_type_incomplete.carbon:[[@LINE-20]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR:   fn F(a: T);
    // CHECK:STDERR:   ^~~~~~~~~~~
    // CHECK:STDERR:
    fn F(c: C);
  }
}

// CHECK:STDOUT: --- no_thunk_param_name_differs.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.type.f36: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.4c3: %F.type.f36 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.1 [concrete = constants.%F.bc6] {
// CHECK:STDOUT:     %x.patt: @F.1.%pattern_type (%pattern_type.6de) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.1.%pattern_type (%pattern_type.6de) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: @F.1.%Self.as_type.loc4_23.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc4_23.1: type = splice_block %.loc4_23.2 [symbolic = %Self.as_type.loc4_23.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc4_23.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc4_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc4_23.2: type = converted %Self.ref, %Self.as_type.loc4_23.2 [symbolic = %Self.as_type.loc4_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.1.%Self.as_type.loc4_23.1 (%Self.as_type) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %I.ref {
// CHECK:STDOUT:   %F.decl: %F.type.f36 = fn_decl @F.2 [concrete = constants.%F.4c3] {
// CHECK:STDOUT:     %not_x.patt: %pattern_type.c48 = binding_pattern not_x [concrete]
// CHECK:STDOUT:     %not_x.param_patt: %pattern_type.c48 = value_param_pattern %not_x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %not_x.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %not_x: %C = bind_name not_x, %not_x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @C.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@impl.%F.decl), @impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc4_23.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc4_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc4_23.1 [symbolic = %pattern_type (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.1.%Self.as_type.loc4_23.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%not_x.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc4_23.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc4_23.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- struct_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.a.b.391: type = struct_type {.a: %empty_tuple.type, .b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %pattern_type.bec: type = pattern_type %struct_type.a.b.391 [concrete]
// CHECK:STDOUT:   %struct_type.c.d.15a: type = struct_type {.c: %empty_tuple.type, .d: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %pattern_type.d63: type = pattern_type %struct_type.c.d.15a [concrete]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %struct_type.b.a.40c: type = struct_type {.b: %empty_struct_type, .a: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.231: type = pattern_type %struct_type.b.a.40c [concrete]
// CHECK:STDOUT:   %struct_type.d.c.b36: type = struct_type {.d: %empty_struct_type, .c: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type.844: type = pattern_type %struct_type.d.c.b36 [concrete]
// CHECK:STDOUT:   %F.type.39e918.1: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c04b92.1: %F.type.39e918.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_tuple.type, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %F.type.39e918.2: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.c04b92.2: %F.type.39e918.2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.c.d.15a = struct_value (%empty_tuple, %empty_struct) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@impl.%F.decl.loc9_48.2), @impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.1 [concrete = constants.%F.bc6] {
// CHECK:STDOUT:     %x.patt: %pattern_type.bec = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.bec = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d63 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d63 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_38.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_38.2: type = converted %.loc5_38.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc5_46.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_46.2: type = converted %.loc5_46.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.c.d: type = struct_type {.c: %empty_tuple.type, .d: %empty_struct_type} [concrete = constants.%struct_type.c.d.15a]
// CHECK:STDOUT:     %x.param: %struct_type.a.b.391 = value_param call_param0
// CHECK:STDOUT:     %.loc5_26: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.391] {
// CHECK:STDOUT:       %.loc5_17.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc5_17.2: type = converted %.loc5_17.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc5_25.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc5_25.2: type = converted %.loc5_25.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %struct_type.a.b: type = struct_type {.a: %empty_tuple.type, .b: %empty_struct_type} [concrete = constants.%struct_type.a.b.391]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %struct_type.a.b.391 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %struct_type.c.d.15a = out_param call_param1
// CHECK:STDOUT:     %return: ref %struct_type.c.d.15a = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc8_7.2 as %I.ref {
// CHECK:STDOUT:   %F.decl.loc9_48.1: %F.type.39e918.1 = fn_decl @F.2 [concrete = constants.%F.c04b92.1] {
// CHECK:STDOUT:     %y.patt: %pattern_type.231 = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.231 = value_param_pattern %y.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.844 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.844 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_38.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc9_38.2: type = converted %.loc9_38.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc9_46.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_46.2: type = converted %.loc9_46.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.d.c: type = struct_type {.d: %empty_struct_type, .c: %empty_tuple.type} [concrete = constants.%struct_type.d.c.b36]
// CHECK:STDOUT:     %y.param: %struct_type.b.a.40c = value_param call_param0
// CHECK:STDOUT:     %.loc9_26: type = splice_block %struct_type.b.a [concrete = constants.%struct_type.b.a.40c] {
// CHECK:STDOUT:       %.loc9_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc9_17.2: type = converted %.loc9_17.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:       %.loc9_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc9_25.2: type = converted %.loc9_25.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %struct_type.b.a: type = struct_type {.b: %empty_struct_type, .a: %empty_tuple.type} [concrete = constants.%struct_type.b.a.40c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %y: %struct_type.b.a.40c = bind_name y, %y.param
// CHECK:STDOUT:     %return.param: ref %struct_type.d.c.b36 = out_param call_param1
// CHECK:STDOUT:     %return: ref %struct_type.d.c.b36 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc9_48.2: %F.type.39e918.2 = fn_decl @F.3 [concrete = constants.%F.c04b92.2] {
// CHECK:STDOUT:     %x.patt: %pattern_type.bec = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.bec = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d63 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d63 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %struct_type.a.b.391 = value_param call_param0
// CHECK:STDOUT:     %x: %struct_type.a.b.391 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %struct_type.c.d.15a = out_param call_param1
// CHECK:STDOUT:     %return: ref %struct_type.c.d.15a = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl.loc9_48.1
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn(%x.param: %struct_type.a.b.391) -> %return.param: %struct_type.c.d.15a;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%y.param: %struct_type.b.a.40c) -> %return.param: %struct_type.d.c.b36;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3(%x.param: %struct_type.a.b.391) -> %return.param: %struct_type.c.d.15a {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %struct_type.a.b.391 = name_ref x, %x.param
// CHECK:STDOUT:   %.loc9_48.1: ref %struct_type.d.c.b36 = temporary_storage
// CHECK:STDOUT:   %.loc5_9.1: %empty_struct_type = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc5_9.2: %empty_tuple.type = struct_access %x.ref, element0
// CHECK:STDOUT:   %struct: %struct_type.b.a.40c = struct_value (%.loc5_9.1, %.loc5_9.2)
// CHECK:STDOUT:   %.loc5_9.3: %struct_type.b.a.40c = converted %x.ref, %struct
// CHECK:STDOUT:   %F.call: init %struct_type.d.c.b36 = call @impl.%F.decl.loc9_48.1(%.loc5_9.3) to %.loc9_48.1
// CHECK:STDOUT:   %.loc9_48.2: ref %struct_type.d.c.b36 = temporary %.loc9_48.1, %F.call
// CHECK:STDOUT:   %.loc9_48.3: ref %empty_tuple.type = struct_access %.loc9_48.2, element1
// CHECK:STDOUT:   %.loc9_48.4: ref %empty_tuple.type = struct_access %return, element1
// CHECK:STDOUT:   %.loc9_48.5: init %empty_tuple.type = tuple_init () to %.loc9_48.4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_48.6: init %empty_tuple.type = converted %.loc9_48.3, %.loc9_48.5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_48.7: ref %empty_struct_type = struct_access %.loc9_48.2, element0
// CHECK:STDOUT:   %.loc9_48.8: ref %empty_struct_type = struct_access %return, element0
// CHECK:STDOUT:   %.loc9_48.9: init %empty_struct_type = struct_init () to %.loc9_48.8 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_48.10: init %empty_struct_type = converted %.loc9_48.7, %.loc9_48.9 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_48.11: init %struct_type.c.d.15a = struct_init (%.loc9_48.6, %.loc9_48.10) to %return [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc9_48.12: init %struct_type.c.d.15a = converted %F.call, %.loc9_48.11 [concrete = constants.%struct]
// CHECK:STDOUT:   return %.loc9_48.12 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %struct_type.base.953: type = struct_type {.base: %A} [concrete]
// CHECK:STDOUT:   %complete_type.020: <witness> = complete_type_witness %struct_type.base.953 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B [concrete]
// CHECK:STDOUT:   %struct_type.base.0ff: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.98e: <witness> = complete_type_witness %struct_type.base.0ff [concrete]
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %Self: %X.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %ptr.d06: type = ptr_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.4c5: type = pattern_type %ptr.d06 [symbolic]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %F.type.594: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.b69: %F.type.594 = struct_value () [concrete]
// CHECK:STDOUT:   %X.assoc_type: type = assoc_entity_type @X [concrete]
// CHECK:STDOUT:   %assoc0: %X.assoc_type = assoc_entity element0, @X.%F.decl [concrete]
// CHECK:STDOUT:   %X.impl_witness: <witness> = impl_witness file.%X.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %ptr.6db [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %F.type.f1b0b1.1: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.5161e9.1: %F.type.f1b0b1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %X.facet: %X.type = facet_value %B, (%X.impl_witness) [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %F.type.f1b0b1.2: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.5161e9.2: %F.type.f1b0b1.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %X.decl: type = interface_decl @X [concrete = constants.%X.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.impl_witness_table = impl_witness_table (@impl.%F.decl.loc13_39.2), @impl [concrete]
// CHECK:STDOUT:   %X.impl_witness: <witness> = impl_witness %X.impl_witness_table [concrete = constants.%X.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @X {
// CHECK:STDOUT:   %Self: %X.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.594 = fn_decl @F.1 [concrete = constants.%F.b69] {
// CHECK:STDOUT:     %self.patt: @F.1.%pattern_type (%pattern_type.4c5) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @F.1.%pattern_type (%pattern_type.4c5) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc9_8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     %other.patt: @F.1.%pattern_type (%pattern_type.4c5) = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: @F.1.%pattern_type (%pattern_type.4c5) = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @F.1.%pattern_type (%pattern_type.4c5) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.1.%pattern_type (%pattern_type.4c5) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc9_43: %X.type = name_ref Self, @X.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc9_47: type = facet_access_type %Self.ref.loc9_43 [symbolic = %Self.as_type.loc9_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc9_47: type = converted %Self.ref.loc9_43, %Self.as_type.loc9_47 [symbolic = %Self.as_type.loc9_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %ptr.loc9_47: type = ptr_type %.loc9_47 [symbolic = %ptr.loc9_23.1 (constants.%ptr.d06)]
// CHECK:STDOUT:     %self.param: @F.1.%ptr.loc9_23.1 (%ptr.d06) = value_param call_param0
// CHECK:STDOUT:     %.loc9_23.1: type = splice_block %ptr.loc9_23.2 [symbolic = %ptr.loc9_23.1 (constants.%ptr.d06)] {
// CHECK:STDOUT:       %Self.ref.loc9_19: %X.type = name_ref Self, @X.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc9_23.2: type = facet_access_type %Self.ref.loc9_19 [symbolic = %Self.as_type.loc9_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc9_23.2: type = converted %Self.ref.loc9_19, %Self.as_type.loc9_23.2 [symbolic = %Self.as_type.loc9_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %ptr.loc9_23.2: type = ptr_type %.loc9_23.2 [symbolic = %ptr.loc9_23.1 (constants.%ptr.d06)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%ptr.loc9_23.1 (%ptr.d06) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @F.1.%ptr.loc9_23.1 (%ptr.d06) = value_param call_param1
// CHECK:STDOUT:     %.loc9_37.1: type = splice_block %ptr.loc9_37 [symbolic = %ptr.loc9_23.1 (constants.%ptr.d06)] {
// CHECK:STDOUT:       %Self.ref.loc9_33: %X.type = name_ref Self, @X.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc9_37: type = facet_access_type %Self.ref.loc9_33 [symbolic = %Self.as_type.loc9_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc9_37.2: type = converted %Self.ref.loc9_33, %Self.as_type.loc9_37 [symbolic = %Self.as_type.loc9_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %ptr.loc9_37: type = ptr_type %.loc9_37.2 [symbolic = %ptr.loc9_23.1 (constants.%ptr.d06)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @F.1.%ptr.loc9_23.1 (%ptr.d06) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @F.1.%ptr.loc9_23.1 (%ptr.d06) = out_param call_param2
// CHECK:STDOUT:     %return: ref @F.1.%ptr.loc9_23.1 (%ptr.d06) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %X.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %F.decl.loc13_39.1: %F.type.f1b0b1.1 = fn_decl @F.2 [concrete = constants.%F.5161e9.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.5f8 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5f8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc13_8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.5f8 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.5f8 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.44a = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.44a = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc13_38: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     %self.param: %ptr.6db = value_param call_param0
// CHECK:STDOUT:     %.loc13_20: type = splice_block %ptr.loc13_20 [concrete = constants.%ptr.6db] {
// CHECK:STDOUT:       %A.ref.loc13_19: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc13_20: type = ptr_type %A.ref.loc13_19 [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.6db = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %ptr.6db = value_param call_param1
// CHECK:STDOUT:     %.loc13_31: type = splice_block %ptr.loc13_31 [concrete = constants.%ptr.6db] {
// CHECK:STDOUT:       %A.ref.loc13_30: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc13_31: type = ptr_type %A.ref.loc13_30 [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: %ptr.6db = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %ptr.019 = out_param call_param2
// CHECK:STDOUT:     %return: ref %ptr.019 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc13_39.2: %F.type.f1b0b1.2 = fn_decl @F.3 [concrete = constants.%F.5161e9.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.960 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.960 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc9_8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.960 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.960 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.960 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.960 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e79 = value_param call_param0
// CHECK:STDOUT:     %self: %ptr.e79 = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %ptr.e79 = value_param call_param1
// CHECK:STDOUT:     %other: %ptr.e79 = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %ptr.e79 = out_param call_param2
// CHECK:STDOUT:     %return: ref %ptr.e79 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %F.decl.loc13_39.1
// CHECK:STDOUT:   witness = file.%X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc5: %B.elem = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %A} [concrete = constants.%struct_type.base.953]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.020]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .base = %.loc5
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc6: %C.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %B} [concrete = constants.%struct_type.base.0ff]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.98e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc6
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@X.%Self: %X.type) {
// CHECK:STDOUT:   %Self: %X.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc9_23.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %ptr.loc9_23.1: type = ptr_type %Self.as_type.loc9_23.1 [symbolic = %ptr.loc9_23.1 (constants.%ptr.d06)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc9_23.1 [symbolic = %pattern_type (constants.%pattern_type.4c5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @F.1.%ptr.loc9_23.1 (%ptr.d06), %other.param: @F.1.%ptr.loc9_23.1 (%ptr.d06)) -> @F.1.%ptr.loc9_23.1 (%ptr.d06);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%self.param: %ptr.6db, %other.param: %ptr.6db) -> %ptr.019;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3(%self.param: %ptr.e79, %other.param: %ptr.e79) -> %ptr.e79 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %ptr.e79 = name_ref self, %self.param
// CHECK:STDOUT:   %.loc9_17.1: ref %B = deref %self.ref
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc9_17.1, @impl.%F.decl.loc13_39.1
// CHECK:STDOUT:   %other.ref: %ptr.e79 = name_ref other, %other.param
// CHECK:STDOUT:   %addr.loc9_17.1: %ptr.e79 = addr_of %.loc9_17.1
// CHECK:STDOUT:   %.loc9_17.2: ref %B = deref %addr.loc9_17.1
// CHECK:STDOUT:   %.loc9_17.3: ref %A = class_element_access %.loc9_17.2, element0
// CHECK:STDOUT:   %addr.loc9_17.2: %ptr.6db = addr_of %.loc9_17.3
// CHECK:STDOUT:   %.loc9_17.4: %ptr.6db = converted %addr.loc9_17.1, %addr.loc9_17.2
// CHECK:STDOUT:   %.loc9_31.1: ref %B = deref %other.ref
// CHECK:STDOUT:   %.loc9_31.2: ref %A = class_element_access %.loc9_31.1, element0
// CHECK:STDOUT:   %addr.loc9_31: %ptr.6db = addr_of %.loc9_31.2
// CHECK:STDOUT:   %.loc9_31.3: %ptr.6db = converted %other.ref, %addr.loc9_31
// CHECK:STDOUT:   %F.call: init %ptr.019 = call %F.bound(%.loc9_17.4, %.loc9_31.3)
// CHECK:STDOUT:   %.loc13_39.1: %ptr.019 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc13_39.2: %ptr.019 = converted %F.call, %.loc13_39.1
// CHECK:STDOUT:   %.loc13_39.3: ref %C = deref %.loc13_39.2
// CHECK:STDOUT:   %.loc13_39.4: ref %B = class_element_access %.loc13_39.3, element0
// CHECK:STDOUT:   %addr.loc13: %ptr.e79 = addr_of %.loc13_39.4
// CHECK:STDOUT:   %.loc13_39.5: %ptr.e79 = converted %F.call, %addr.loc13
// CHECK:STDOUT:   return %.loc13_39.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc9_23.1 => constants.%Self.as_type
// CHECK:STDOUT:   %ptr.loc9_23.1 => constants.%ptr.d06
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4c5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%X.facet) {
// CHECK:STDOUT:   %Self => constants.%X.facet
// CHECK:STDOUT:   %Self.as_type.loc9_23.1 => constants.%B
// CHECK:STDOUT:   %ptr.loc9_23.1 => constants.%ptr.e79
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.960
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_value_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %struct_type.base.953: type = struct_type {.base: %A} [concrete]
// CHECK:STDOUT:   %complete_type.020: <witness> = complete_type_witness %struct_type.base.953 [concrete]
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %Self: %X.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.331: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %F.type.594: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.b69: %F.type.594 = struct_value () [concrete]
// CHECK:STDOUT:   %X.assoc_type: type = assoc_entity_type @X [concrete]
// CHECK:STDOUT:   %assoc0: %X.assoc_type = assoc_entity element0, @X.%F.decl [concrete]
// CHECK:STDOUT:   %X.impl_witness: <witness> = impl_witness file.%X.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %F.type.f1b0b1.1: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.5161e9.1: %F.type.f1b0b1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %X.facet: %X.type = facet_value %B, (%X.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %F.type.f1b0b1.2: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.5161e9.2: %F.type.f1b0b1.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %X.decl: type = interface_decl @X [concrete = constants.%X.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.impl_witness_table = impl_witness_table (@impl.%F.decl.loc12_26.2), @impl [concrete]
// CHECK:STDOUT:   %X.impl_witness: <witness> = impl_witness %X.impl_witness_table [concrete = constants.%X.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @X {
// CHECK:STDOUT:   %Self: %X.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.594 = fn_decl @F.1 [concrete = constants.%F.b69] {
// CHECK:STDOUT:     %self.patt: @F.1.%pattern_type (%pattern_type.331) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @F.1.%pattern_type (%pattern_type.331) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: @F.1.%pattern_type (%pattern_type.331) = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: @F.1.%pattern_type (%pattern_type.331) = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc8_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_14.1: type = splice_block %.loc8_14.2 [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc8_14: %X.type = name_ref Self, @X.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc8_14.2: type = facet_access_type %Self.ref.loc8_14 [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc8_14.2: type = converted %Self.ref.loc8_14, %Self.as_type.loc8_14.2 [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc8_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @F.1.%Self.as_type.loc8_14.1 (%Self.as_type) = value_param call_param1
// CHECK:STDOUT:     %.loc8_27.1: type = splice_block %.loc8_27.2 [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref.loc8_27: %X.type = name_ref Self, @X.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc8_27: type = facet_access_type %Self.ref.loc8_27 [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc8_27.2: type = converted %Self.ref.loc8_27, %Self.as_type.loc8_27 [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @F.1.%Self.as_type.loc8_14.1 (%Self.as_type) = bind_name other, %other.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %X.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %F.decl.loc12_26.1: %F.type.f1b0b1.1 = fn_decl @F.2 [concrete = constants.%F.5161e9.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c10 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c10 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.c10 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.c10 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A = value_param call_param0
// CHECK:STDOUT:     %A.ref.loc12_14: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %self: %A = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %A = value_param call_param1
// CHECK:STDOUT:     %A.ref.loc12_24: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %other: %A = bind_name other, %other.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc12_26.2: %F.type.f1b0b1.2 = fn_decl @F.3 [concrete = constants.%F.5161e9.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.049 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.049 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.049 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.049 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %B = value_param call_param0
// CHECK:STDOUT:     %self: %B = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %B = value_param call_param1
// CHECK:STDOUT:     %other: %B = bind_name other, %other.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .F = %F.decl.loc12_26.1
// CHECK:STDOUT:   witness = file.%X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc5: %B.elem = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %A} [concrete = constants.%struct_type.base.953]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.020]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .base = %.loc5
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@X.%Self: %X.type) {
// CHECK:STDOUT:   %Self: %X.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc8_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc8_14.1 [symbolic = %pattern_type (constants.%pattern_type.331)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @F.1.%Self.as_type.loc8_14.1 (%Self.as_type), %other.param: @F.1.%Self.as_type.loc8_14.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%self.param: %A, %other.param: %A);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3(%self.param: %B, %other.param: %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %B = name_ref self, %self.param
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %self.ref, @impl.%F.decl.loc12_26.1
// CHECK:STDOUT:   %other.ref: %B = name_ref other, %other.param
// CHECK:STDOUT:   %.loc8_12.1: ref %A = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc8_12.2: ref %A = converted %self.ref, %.loc8_12.1
// CHECK:STDOUT:   %.loc8_12.3: %A = bind_value %.loc8_12.2
// CHECK:STDOUT:   %.loc8_25.1: ref %A = class_element_access %other.ref, element0
// CHECK:STDOUT:   %.loc8_25.2: ref %A = converted %other.ref, %.loc8_25.1
// CHECK:STDOUT:   %.loc8_25.3: %A = bind_value %.loc8_25.2
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.bound(%.loc8_12.3, %.loc8_25.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc8_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.331
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%X.facet) {
// CHECK:STDOUT:   %Self => constants.%X.facet
// CHECK:STDOUT:   %Self.as_type.loc8_14.1 => constants.%B
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.049
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inheritance_value_conversion_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %struct_type.base.953: type = struct_type {.base: %A} [concrete]
// CHECK:STDOUT:   %complete_type.020: <witness> = complete_type_witness %struct_type.base.953 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B [concrete]
// CHECK:STDOUT:   %struct_type.base.0ff: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.98e: <witness> = complete_type_witness %struct_type.base.0ff [concrete]
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %Self: %X.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %ptr.d06: type = ptr_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.4c5: type = pattern_type %ptr.d06 [symbolic]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %F.type.594: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.b69: %F.type.594 = struct_value () [concrete]
// CHECK:STDOUT:   %X.assoc_type: type = assoc_entity_type @X [concrete]
// CHECK:STDOUT:   %assoc0: %X.assoc_type = assoc_entity element0, @X.%F.decl [concrete]
// CHECK:STDOUT:   %X.impl_witness: <witness> = impl_witness file.%X.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %ptr.6db [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %F.type.f1b0b1.1: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.5161e9.1: %F.type.f1b0b1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %X.facet: %X.type = facet_value %B, (%X.impl_witness) [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %F.type.f1b0b1.2: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.5161e9.2: %F.type.f1b0b1.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %X.decl: type = interface_decl @X [concrete = constants.%X.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.impl_witness_table = impl_witness_table (@impl.%F.decl.loc13_39.2), @impl [concrete]
// CHECK:STDOUT:   %X.impl_witness: <witness> = impl_witness %X.impl_witness_table [concrete = constants.%X.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @X {
// CHECK:STDOUT:   %Self: %X.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.594 = fn_decl @F.1 [concrete = constants.%F.b69] {
// CHECK:STDOUT:     %self.patt: @F.1.%pattern_type (%pattern_type.4c5) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @F.1.%pattern_type (%pattern_type.4c5) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc9_8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     %other.patt: @F.1.%pattern_type (%pattern_type.4c5) = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: @F.1.%pattern_type (%pattern_type.4c5) = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @F.1.%pattern_type (%pattern_type.4c5) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.1.%pattern_type (%pattern_type.4c5) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc9_43: %X.type = name_ref Self, @X.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc9_47: type = facet_access_type %Self.ref.loc9_43 [symbolic = %Self.as_type.loc9_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc9_47: type = converted %Self.ref.loc9_43, %Self.as_type.loc9_47 [symbolic = %Self.as_type.loc9_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %ptr.loc9_47: type = ptr_type %.loc9_47 [symbolic = %ptr.loc9_23.1 (constants.%ptr.d06)]
// CHECK:STDOUT:     %self.param: @F.1.%ptr.loc9_23.1 (%ptr.d06) = value_param call_param0
// CHECK:STDOUT:     %.loc9_23.1: type = splice_block %ptr.loc9_23.2 [symbolic = %ptr.loc9_23.1 (constants.%ptr.d06)] {
// CHECK:STDOUT:       %Self.ref.loc9_19: %X.type = name_ref Self, @X.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc9_23.2: type = facet_access_type %Self.ref.loc9_19 [symbolic = %Self.as_type.loc9_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc9_23.2: type = converted %Self.ref.loc9_19, %Self.as_type.loc9_23.2 [symbolic = %Self.as_type.loc9_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %ptr.loc9_23.2: type = ptr_type %.loc9_23.2 [symbolic = %ptr.loc9_23.1 (constants.%ptr.d06)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%ptr.loc9_23.1 (%ptr.d06) = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: @F.1.%ptr.loc9_23.1 (%ptr.d06) = value_param call_param1
// CHECK:STDOUT:     %.loc9_37.1: type = splice_block %ptr.loc9_37 [symbolic = %ptr.loc9_23.1 (constants.%ptr.d06)] {
// CHECK:STDOUT:       %Self.ref.loc9_33: %X.type = name_ref Self, @X.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc9_37: type = facet_access_type %Self.ref.loc9_33 [symbolic = %Self.as_type.loc9_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc9_37.2: type = converted %Self.ref.loc9_33, %Self.as_type.loc9_37 [symbolic = %Self.as_type.loc9_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %ptr.loc9_37: type = ptr_type %.loc9_37.2 [symbolic = %ptr.loc9_23.1 (constants.%ptr.d06)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: @F.1.%ptr.loc9_23.1 (%ptr.d06) = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref @F.1.%ptr.loc9_23.1 (%ptr.d06) = out_param call_param2
// CHECK:STDOUT:     %return: ref @F.1.%ptr.loc9_23.1 (%ptr.d06) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %X.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %B.ref as %X.ref {
// CHECK:STDOUT:   %F.decl.loc13_39.1: %F.type.f1b0b1.1 = fn_decl @F.2 [concrete = constants.%F.5161e9.1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.5f8 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5f8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc13_8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.5f8 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.5f8 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.44a = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.44a = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc13_38: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     %self.param: %ptr.6db = value_param call_param0
// CHECK:STDOUT:     %.loc13_20: type = splice_block %ptr.loc13_20 [concrete = constants.%ptr.6db] {
// CHECK:STDOUT:       %A.ref.loc13_19: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc13_20: type = ptr_type %A.ref.loc13_19 [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.6db = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %ptr.6db = value_param call_param1
// CHECK:STDOUT:     %.loc13_31: type = splice_block %ptr.loc13_31 [concrete = constants.%ptr.6db] {
// CHECK:STDOUT:       %A.ref.loc13_30: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc13_31: type = ptr_type %A.ref.loc13_30 [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %other: %ptr.6db = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %ptr.019 = out_param call_param2
// CHECK:STDOUT:     %return: ref %ptr.019 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc13_39.2: %F.type.f1b0b1.2 = fn_decl @F.3 [concrete = constants.%F.5161e9.2] {
// CHECK:STDOUT:     %self.patt: %pattern_type.960 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.960 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc9_8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.960 = binding_pattern other [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.960 = value_param_pattern %other.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.960 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.960 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e79 = value_param call_param0
// CHECK:STDOUT:     %self: %ptr.e79 = bind_name self, %self.param
// CHECK:STDOUT:     %other.param: %ptr.e79 = value_param call_param1
// CHECK:STDOUT:     %other: %ptr.e79 = bind_name other, %other.param
// CHECK:STDOUT:     %return.param: ref %ptr.e79 = out_param call_param2
// CHECK:STDOUT:     %return: ref %ptr.e79 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %F.decl.loc13_39.1
// CHECK:STDOUT:   witness = file.%X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc5: %B.elem = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %A} [concrete = constants.%struct_type.base.953]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.020]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .base = %.loc5
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc6: %C.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %B} [concrete = constants.%struct_type.base.0ff]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.98e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc6
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@X.%Self: %X.type) {
// CHECK:STDOUT:   %Self: %X.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc9_23.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_23.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %ptr.loc9_23.1: type = ptr_type %Self.as_type.loc9_23.1 [symbolic = %ptr.loc9_23.1 (constants.%ptr.d06)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc9_23.1 [symbolic = %pattern_type (constants.%pattern_type.4c5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @F.1.%ptr.loc9_23.1 (%ptr.d06), %other.param: @F.1.%ptr.loc9_23.1 (%ptr.d06)) -> @F.1.%ptr.loc9_23.1 (%ptr.d06);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%self.param: %ptr.6db, %other.param: %ptr.6db) -> %ptr.019;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3(%self.param: %ptr.e79, %other.param: %ptr.e79) -> %ptr.e79 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %ptr.e79 = name_ref self, %self.param
// CHECK:STDOUT:   %.loc9_17.1: ref %B = deref %self.ref
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc9_17.1, @impl.%F.decl.loc13_39.1
// CHECK:STDOUT:   %other.ref: %ptr.e79 = name_ref other, %other.param
// CHECK:STDOUT:   %addr.loc9_17.1: %ptr.e79 = addr_of %.loc9_17.1
// CHECK:STDOUT:   %.loc9_17.2: ref %B = deref %addr.loc9_17.1
// CHECK:STDOUT:   %.loc9_17.3: ref %A = class_element_access %.loc9_17.2, element0
// CHECK:STDOUT:   %addr.loc9_17.2: %ptr.6db = addr_of %.loc9_17.3
// CHECK:STDOUT:   %.loc9_17.4: %ptr.6db = converted %addr.loc9_17.1, %addr.loc9_17.2
// CHECK:STDOUT:   %.loc9_31.1: ref %B = deref %other.ref
// CHECK:STDOUT:   %.loc9_31.2: ref %A = class_element_access %.loc9_31.1, element0
// CHECK:STDOUT:   %addr.loc9_31: %ptr.6db = addr_of %.loc9_31.2
// CHECK:STDOUT:   %.loc9_31.3: %ptr.6db = converted %other.ref, %addr.loc9_31
// CHECK:STDOUT:   %F.call: init %ptr.019 = call %F.bound(%.loc9_17.4, %.loc9_31.3)
// CHECK:STDOUT:   %.loc13_39.1: %ptr.019 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc13_39.2: %ptr.019 = converted %F.call, %.loc13_39.1
// CHECK:STDOUT:   %.loc13_39.3: ref %C = deref %.loc13_39.2
// CHECK:STDOUT:   %.loc13_39.4: ref %B = class_element_access %.loc13_39.3, element0
// CHECK:STDOUT:   %addr.loc13: %ptr.e79 = addr_of %.loc13_39.4
// CHECK:STDOUT:   %.loc13_39.5: %ptr.e79 = converted %F.call, %addr.loc13
// CHECK:STDOUT:   return %.loc13_39.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc9_23.1 => constants.%Self.as_type
// CHECK:STDOUT:   %ptr.loc9_23.1 => constants.%ptr.d06
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4c5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%X.facet) {
// CHECK:STDOUT:   %Self => constants.%X.facet
// CHECK:STDOUT:   %Self.as_type.loc9_23.1 => constants.%B
// CHECK:STDOUT:   %ptr.loc9_23.1 => constants.%ptr.e79
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.960
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inheritance_value_conversion_copy_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %struct_type.base.953: type = struct_type {.base: %A} [concrete]
// CHECK:STDOUT:   %complete_type.020: <witness> = complete_type_witness %struct_type.base.953 [concrete]
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %Self: %X.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.331: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %F.type.594: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.b69: %F.type.594 = struct_value () [concrete]
// CHECK:STDOUT:   %X.assoc_type: type = assoc_entity_type @X [concrete]
// CHECK:STDOUT:   %assoc0: %X.assoc_type = assoc_entity element0, @X.%F.decl [concrete]
// CHECK:STDOUT:   %X.impl_witness: <witness> = impl_witness file.%X.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %F.type.b24d6f.1: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.77e9d5.1: %F.type.b24d6f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %X.facet: %X.type = facet_value %A, (%X.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %F.type.b24d6f.2: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.77e9d5.2: %F.type.b24d6f.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %X.decl: type = interface_decl @X [concrete = constants.%X.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.impl_witness_table = impl_witness_table (@impl.%F.decl.loc19_14.2), @impl [concrete]
// CHECK:STDOUT:   %X.impl_witness: <witness> = impl_witness %X.impl_witness_table [concrete = constants.%X.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @X {
// CHECK:STDOUT:   %Self: %X.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.594 = fn_decl @F.1 [concrete = constants.%F.b69] {
// CHECK:STDOUT:     %return.patt: @F.1.%pattern_type (%pattern_type.331) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.1.%pattern_type (%pattern_type.331) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: %X.type = name_ref Self, @X.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc8_13.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_13.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc8: type = converted %Self.ref, %Self.as_type.loc8_13.2 [symbolic = %Self.as_type.loc8_13.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %return.param: ref @F.1.%Self.as_type.loc8_13.1 (%Self.as_type) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.1.%Self.as_type.loc8_13.1 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %X.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A.ref as %X.ref {
// CHECK:STDOUT:   %F.decl.loc19_14.1: %F.type.b24d6f.1 = fn_decl @F.2 [concrete = constants.%F.77e9d5.1] {
// CHECK:STDOUT:     %return.patt: %pattern_type.049 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.049 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc19_14.2: %F.type.b24d6f.2 = fn_decl @F.3 [concrete = constants.%F.77e9d5.2] {
// CHECK:STDOUT:     %return.patt: %pattern_type.c10 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c10 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref %A = out_param call_param0
// CHECK:STDOUT:     %return: ref %A = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .F = %F.decl.loc19_14.1
// CHECK:STDOUT:   witness = file.%X.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc5: %B.elem = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %A} [concrete = constants.%struct_type.base.953]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.020]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .base = %.loc5
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@X.%Self: %X.type) {
// CHECK:STDOUT:   %Self: %X.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc8_13.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_13.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc8_13.1 [symbolic = %pattern_type (constants.%pattern_type.331)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @F.1.%Self.as_type.loc8_13.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %return.param: %B;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3() -> %return.param: %A {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_14.1: ref %B = temporary_storage
// CHECK:STDOUT:   %F.call: init %B = call @impl.%F.decl.loc19_14.1() to %.loc19_14.1
// CHECK:STDOUT:   %.loc19_14.2: ref %B = temporary %.loc19_14.1, %F.call
// CHECK:STDOUT:   %.loc19_14.3: ref %A = class_element_access %.loc19_14.2, element0
// CHECK:STDOUT:   %.loc19_14.4: ref %A = converted %F.call, %.loc19_14.3
// CHECK:STDOUT:   %.loc19_14.5: %A = bind_value %.loc19_14.4
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc8_13.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.331
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%X.facet) {
// CHECK:STDOUT:   %Self => constants.%X.facet
// CHECK:STDOUT:   %Self.as_type.loc8_13.1 => constants.%A
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_param_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %F.type.2ae1ef.1: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.2a7aab.1: %F.type.2ae1ef.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %A, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %F.type.2ae1ef.2: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.2a7aab.2: %F.type.2ae1ef.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@impl.%F.decl.loc22_13.2), @impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.1 [concrete = constants.%F.bc6] {
// CHECK:STDOUT:     %a.patt: @F.1.%pattern_type (%pattern_type.6de) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.1.%pattern_type (%pattern_type.6de) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: @F.1.%Self.as_type.loc8_11.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc8_11.1: type = splice_block %.loc8_11.2 [symbolic = %Self.as_type.loc8_11.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc8_11.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_11.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc8_11.2: type = converted %Self.ref, %Self.as_type.loc8_11.2 [symbolic = %Self.as_type.loc8_11.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.1.%Self.as_type.loc8_11.1 (%Self.as_type) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A.ref as %I.ref {
// CHECK:STDOUT:   %F.decl.loc22_13.1: %F.type.2ae1ef.1 = fn_decl @F.2 [concrete = constants.%F.2a7aab.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type.049 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.049 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %B = value_param call_param0
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %a: %B = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc22_13.2: %F.type.2ae1ef.2 = fn_decl @F.3 [concrete = constants.%F.2a7aab.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.c10 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %A = value_param call_param0
// CHECK:STDOUT:     %a: %A = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .F = %F.decl.loc22_13.1
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc8_11.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_11.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc8_11.1 [symbolic = %pattern_type (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.1.%Self.as_type.loc8_11.1 (%Self.as_type));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%a.param: %B);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3(%a.param: %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %A = name_ref a, %a.param
// CHECK:STDOUT:   %.loc8: %B = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call @impl.%F.decl.loc22_13.1(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc8_11.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc8_11.1 => constants.%A
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_return_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %F.type.2ae1ef.1: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.2a7aab.1: %F.type.2ae1ef.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %A, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %F.type.2ae1ef.2: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.2a7aab.2: %F.type.2ae1ef.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@impl.%F.decl.loc19_14.2), @impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.1 [concrete = constants.%F.bc6] {
// CHECK:STDOUT:     %return.patt: @F.1.%pattern_type (%pattern_type.6de) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.1.%pattern_type (%pattern_type.6de) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc5_13.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_13.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc5: type = converted %Self.ref, %Self.as_type.loc5_13.2 [symbolic = %Self.as_type.loc5_13.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %return.param: ref @F.1.%Self.as_type.loc5_13.1 (%Self.as_type) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.1.%Self.as_type.loc5_13.1 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %A.ref as %I.ref {
// CHECK:STDOUT:   %F.decl.loc19_14.1: %F.type.2ae1ef.1 = fn_decl @F.2 [concrete = constants.%F.2a7aab.1] {
// CHECK:STDOUT:     %return.patt: %pattern_type.049 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.049 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %return.param: ref %B = out_param call_param0
// CHECK:STDOUT:     %return: ref %B = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc19_14.2: %F.type.2ae1ef.2 = fn_decl @F.3 [concrete = constants.%F.2a7aab.2] {
// CHECK:STDOUT:     %return.patt: %pattern_type.c10 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c10 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref %A = out_param call_param0
// CHECK:STDOUT:     %return: ref %A = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .F = %F.decl.loc19_14.1
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_13.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_13.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_13.1 [symbolic = %pattern_type (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @F.1.%Self.as_type.loc5_13.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %return.param: %B;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3() -> %return.param: %A {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_14.1: ref %B = temporary_storage
// CHECK:STDOUT:   %F.call: init %B = call @impl.%F.decl.loc19_14.1() to %.loc19_14.1
// CHECK:STDOUT:   %.loc19_14.2: %A = converted %F.call, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_13.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc5_13.1 => constants.%A
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- return_empty_tuple_mismatch_allowed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type.6de: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %HasReturn.type.bf9: type = fn_type @HasReturn.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasReturn.dcf: %HasReturn.type.bf9 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%HasReturn.decl [concrete]
// CHECK:STDOUT:   %NoReturn.type.7ff: type = fn_type @NoReturn.1 [concrete]
// CHECK:STDOUT:   %NoReturn.7fd: %NoReturn.type.7ff = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.%NoReturn.decl [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %EmptyTupleReturn.type.1cb: type = fn_type @EmptyTupleReturn.1 [concrete]
// CHECK:STDOUT:   %EmptyTupleReturn.c56: %EmptyTupleReturn.type.1cb = struct_value () [concrete]
// CHECK:STDOUT:   %assoc2: %I.assoc_type = assoc_entity element2, @I.%EmptyTupleReturn.decl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %HasReturn.type.c73908.1: type = fn_type @HasReturn.2 [concrete]
// CHECK:STDOUT:   %HasReturn.424378.1: %HasReturn.type.c73908.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NoReturn.type.948: type = fn_type @NoReturn.2 [concrete]
// CHECK:STDOUT:   %NoReturn.9fa: %NoReturn.type.948 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyTupleReturn.type.9ea74f.1: type = fn_type @EmptyTupleReturn.2 [concrete]
// CHECK:STDOUT:   %EmptyTupleReturn.282179.1: %EmptyTupleReturn.type.9ea74f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_tuple.type, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %HasReturn.type.c73908.2: type = fn_type @HasReturn.3 [concrete]
// CHECK:STDOUT:   %HasReturn.424378.2: %HasReturn.type.c73908.2 = struct_value () [concrete]
// CHECK:STDOUT:   %EmptyTupleReturn.type.9ea74f.2: type = fn_type @EmptyTupleReturn.3 [concrete]
// CHECK:STDOUT:   %EmptyTupleReturn.282179.2: %EmptyTupleReturn.type.9ea74f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc10_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_7.2: type = converted %.loc10_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@impl.%HasReturn.decl.loc12_17.2, @impl.%NoReturn.decl, @impl.%EmptyTupleReturn.decl.loc18_24.2), @impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %HasReturn.decl: %HasReturn.type.bf9 = fn_decl @HasReturn.1 [concrete = constants.%HasReturn.dcf] {
// CHECK:STDOUT:     %return.patt: @HasReturn.1.%pattern_type (%pattern_type.6de) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @HasReturn.1.%pattern_type (%pattern_type.6de) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %Self.as_type.loc5_21.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_21.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %.loc5: type = converted %Self.ref, %Self.as_type.loc5_21.2 [symbolic = %Self.as_type.loc5_21.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %return.param: ref @HasReturn.1.%Self.as_type.loc5_21.1 (%Self.as_type) = out_param call_param0
// CHECK:STDOUT:     %return: ref @HasReturn.1.%Self.as_type.loc5_21.1 (%Self.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %HasReturn.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %NoReturn.decl: %NoReturn.type.7ff = fn_decl @NoReturn.1 [concrete = constants.%NoReturn.7fd] {} {}
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %NoReturn.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:   %EmptyTupleReturn.decl: %EmptyTupleReturn.type.1cb = fn_decl @EmptyTupleReturn.1 [concrete = constants.%EmptyTupleReturn.c56] {
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_29.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_29.2: type = converted %.loc7_29.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc2: %I.assoc_type = assoc_entity element2, %EmptyTupleReturn.decl [concrete = constants.%assoc2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .HasReturn = %assoc0
// CHECK:STDOUT:   .NoReturn = %assoc1
// CHECK:STDOUT:   .EmptyTupleReturn = %assoc2
// CHECK:STDOUT:   witness = (%HasReturn.decl, %NoReturn.decl, %EmptyTupleReturn.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc10_7.2 as %I.ref {
// CHECK:STDOUT:   %HasReturn.decl.loc12_17.1: %HasReturn.type.c73908.1 = fn_decl @HasReturn.2 [concrete = constants.%HasReturn.424378.1] {} {}
// CHECK:STDOUT:   %NoReturn.decl: %NoReturn.type.948 = fn_decl @NoReturn.2 [concrete = constants.%NoReturn.9fa] {} {}
// CHECK:STDOUT:   %EmptyTupleReturn.decl.loc18_24.1: %EmptyTupleReturn.type.9ea74f.1 = fn_decl @EmptyTupleReturn.2 [concrete = constants.%EmptyTupleReturn.282179.1] {} {}
// CHECK:STDOUT:   %HasReturn.decl.loc12_17.2: %HasReturn.type.c73908.2 = fn_decl @HasReturn.3 [concrete = constants.%HasReturn.424378.2] {
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %EmptyTupleReturn.decl.loc18_24.2: %EmptyTupleReturn.type.9ea74f.2 = fn_decl @EmptyTupleReturn.3 [concrete = constants.%EmptyTupleReturn.282179.2] {
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .HasReturn = %HasReturn.decl.loc12_17.1
// CHECK:STDOUT:   .NoReturn = %NoReturn.decl
// CHECK:STDOUT:   .EmptyTupleReturn = %EmptyTupleReturn.decl.loc18_24.1
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasReturn.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_21.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_21.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc5_21.1 [symbolic = %pattern_type (constants.%pattern_type.6de)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @HasReturn.1.%Self.as_type.loc5_21.1 (%Self.as_type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NoReturn.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EmptyTupleReturn.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn() -> %empty_tuple.type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @HasReturn.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoReturn.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @EmptyTupleReturn.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @HasReturn.3() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %HasReturn.call: init %empty_tuple.type = call @impl.%HasReturn.decl.loc12_17.1()
// CHECK:STDOUT:   %.loc12_17.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc12_17.2: ref %empty_tuple.type = temporary %.loc12_17.1, %HasReturn.call
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_17.3: %empty_tuple.type = converted %HasReturn.call, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc12_17.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @EmptyTupleReturn.3() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %EmptyTupleReturn.call: init %empty_tuple.type = call @impl.%EmptyTupleReturn.decl.loc18_24.1()
// CHECK:STDOUT:   %.loc18_24.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc18_24.2: ref %empty_tuple.type = temporary %.loc18_24.1, %EmptyTupleReturn.call
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_24.3: %empty_tuple.type = converted %EmptyTupleReturn.call, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc18_24.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasReturn.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_21.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NoReturn.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EmptyTupleReturn.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasReturn.1(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc5_21.1 => constants.%empty_tuple.type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cb1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NoReturn.1(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EmptyTupleReturn.1(constants.%I.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_return_empty_tuple_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %NoReturn.type.7ff: type = fn_type @NoReturn.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NoReturn.7fd: %NoReturn.type.7ff = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%NoReturn.decl [concrete]
// CHECK:STDOUT:   %I.impl_witness.c4e: <witness> = impl_witness file.%I.impl_witness_table.loc8 [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %NoReturn.type.948: type = fn_type @NoReturn.2 [concrete]
// CHECK:STDOUT:   %NoReturn.9fa: %NoReturn.type.948 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.372: %I.type = facet_value %empty_tuple.type, (%I.impl_witness.c4e) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.impl_witness.464: <witness> = impl_witness file.%I.impl_witness_table.loc22 [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %NoReturn.type.930: type = fn_type @NoReturn.3 [concrete]
// CHECK:STDOUT:   %NoReturn.ac0: %NoReturn.type.930 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet.be9: %I.type = facet_value %C, (%I.impl_witness.464) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.4ab [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc8 = impl_witness_table (<error>), @impl.4ab [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc8: <witness> = impl_witness %I.impl_witness_table.loc8 [concrete = constants.%I.impl_witness.c4e]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.770 [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table.loc22 = impl_witness_table (<error>), @impl.770 [concrete]
// CHECK:STDOUT:   %I.impl_witness.loc22: <witness> = impl_witness %I.impl_witness_table.loc22 [concrete = constants.%I.impl_witness.464]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %NoReturn.decl: %NoReturn.type.7ff = fn_decl @NoReturn.1 [concrete = constants.%NoReturn.7fd] {} {}
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, %NoReturn.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .NoReturn = %assoc0
// CHECK:STDOUT:   witness = (%NoReturn.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4ab: %.loc8_7.2 as %I.ref {
// CHECK:STDOUT:   %NoReturn.decl: %NoReturn.type.948 = fn_decl @NoReturn.2 [concrete = constants.%NoReturn.9fa] {
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17_21.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc17_21.2: type = converted %.loc17_21.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .NoReturn = %NoReturn.decl
// CHECK:STDOUT:   witness = file.%I.impl_witness.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.770: %C.ref as %I.ref {
// CHECK:STDOUT:   %NoReturn.decl: %NoReturn.type.930 = fn_decl @NoReturn.3 [concrete = constants.%NoReturn.ac0] {
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .NoReturn = %NoReturn.decl
// CHECK:STDOUT:   witness = file.%I.impl_witness.loc22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @NoReturn.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoReturn.2() -> %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoReturn.3() -> %C;
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NoReturn.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NoReturn.1(constants.%I.facet.372) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NoReturn.1(constants.%I.facet.be9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_param_type_incomplete.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.209: %I.type.325 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type.2ae: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.bb2: %F.type.2ae = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.1e5: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.8f0: %I.assoc_type.1e5 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type.a79: type = facet_type <@I, @I(%B)> [concrete]
// CHECK:STDOUT:   %Self.fe1: %I.type.a79 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.7f2: type = fn_type @F.1, @I(%B) [concrete]
// CHECK:STDOUT:   %F.e0a: %F.type.7f2 = struct_value () [concrete]
// CHECK:STDOUT:   %I.assoc_type.5ff: type = assoc_entity_type @I, @I(%B) [concrete]
// CHECK:STDOUT:   %assoc0.470: %I.assoc_type.5ff = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.type.17e183.1: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.4f1533.1: %F.type.17e183.1 = struct_value () [concrete]
// CHECK:STDOUT:   %I.facet: %I.type.a79 = facet_value %C, (%I.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %F.type.17e183.2: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.4f1533.2: %F.type.17e183.2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_13.2)> [symbolic = %I.type (constants.%I.type.325)]
// CHECK:STDOUT:   %Self.2: @I.%I.type (%I.type.325) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.209)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T.loc4_13.2) [symbolic = %F.type (constants.%F.type.2ae)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.2ae) = struct_value () [symbolic = %F (constants.%F.bb2)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T.loc4_13.2) [symbolic = %I.assoc_type (constants.%I.assoc_type.1e5)]
// CHECK:STDOUT:   %assoc0.loc8_13.2: @I.%I.assoc_type (%I.assoc_type.1e5) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc8_13.2 (constants.%assoc0.8f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.325) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.209)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type.2ae) = fn_decl @F.1 [symbolic = @I.%F (constants.%F.bb2)] {
// CHECK:STDOUT:       %a.patt: @F.1.%pattern_type (%pattern_type.7dc) = binding_pattern a [concrete]
// CHECK:STDOUT:       %a.param_patt: @F.1.%pattern_type (%pattern_type.7dc) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %a.param: @F.1.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref: type = name_ref T, @I.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %a: @F.1.%T (%T) = bind_name a, %a.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc8_13.1: @I.%I.assoc_type (%I.assoc_type.1e5) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc8_13.2 (constants.%assoc0.8f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc8_13.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %Self.ref as %I.type {
// CHECK:STDOUT:   %F.decl.loc32_15.1: %F.type.17e183.1 = fn_decl @F.2 [concrete = constants.%F.4f1533.1] {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.c48 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc32_15.2: %F.type.17e183.2 = fn_decl @F.3 [concrete = constants.%F.4f1533.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type.049 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.049 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %B = value_param call_param0
// CHECK:STDOUT:     %a: %B = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %F.decl.loc32_15.1
// CHECK:STDOUT:   witness = @C.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B;
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%B)> [concrete = constants.%I.type.a79]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (@impl.%F.decl.loc32_15.2), @impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .I = <poisoned>
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%T.loc4_13.1: type, @I.%Self.1: @I.%I.type (%I.type.325)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.1.%T (%T));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%c.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3(%a.param: %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %B = name_ref a, %a.param
// CHECK:STDOUT:   %.loc8: %C = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call @impl.%F.decl.loc32_15.1(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self.209) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%B) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.a79
// CHECK:STDOUT:   %Self.2 => constants.%Self.fe1
// CHECK:STDOUT:   %F.type => constants.%F.type.7f2
// CHECK:STDOUT:   %F => constants.%F.e0a
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.5ff
// CHECK:STDOUT:   %assoc0.loc8_13.2 => constants.%assoc0.470
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%B, constants.%I.facet) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.049
// CHECK:STDOUT: }
// CHECK:STDOUT:
