// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/complete_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/complete_type.carbon

// --- fail_incomplete_in_class.carbon

library "[[@TEST_NAME]]";

class B;

class A(T:! type) {
  // CHECK:STDERR: fail_incomplete_in_class.carbon:[[@LINE+6]]:10: error: `T` evaluates to incomplete type `B` [IncompleteTypeInMonomorphization]
  // CHECK:STDERR:   var v: T;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_incomplete_in_class.carbon:[[@LINE-6]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class B;
  // CHECK:STDERR: ^~~~~~~~
  var v: T;
}

// CHECK:STDERR: fail_incomplete_in_class.carbon:[[@LINE+11]]:6: note: in `A(B)` used here [ResolvingSpecificHere]
// CHECK:STDERR: fn F(x: A(B)) {}
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_incomplete_in_class.carbon:[[@LINE+7]]:6: error: parameter has incomplete type `A(B)` in function definition [IncompleteTypeInFunctionParam]
// CHECK:STDERR: fn F(x: A(B)) {}
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR: fail_incomplete_in_class.carbon:[[@LINE-19]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class B;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
fn F(x: A(B)) {}

class B {}

// --- incomplete_in_function.carbon

library "[[@TEST_NAME]]";

class B;

fn F(T:! type) {
  var v: T;
}

// F(B) isn't resolved until the end of the file.
fn G() { F(B); }

class B {}

// --- fail_incomplete_in_function_at_eof.carbon

library "[[@TEST_NAME]]";

class B;

fn F(T:! type) {
  // CHECK:STDERR: fail_incomplete_in_function_at_eof.carbon:[[@LINE+6]]:10: error: `T` evaluates to incomplete type `B` [IncompleteTypeInMonomorphization]
  // CHECK:STDERR:   var v: T;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_incomplete_in_function_at_eof.carbon:[[@LINE-6]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class B;
  // CHECK:STDERR: ^~~~~~~~
  var v: T;
}

// CHECK:STDERR: fail_incomplete_in_function_at_eof.carbon:[[@LINE+4]]:10: note: in `F(B)` used here [ResolvingSpecificHere]
// CHECK:STDERR: fn G() { F(B); }
// CHECK:STDERR:          ^
// CHECK:STDERR:
fn G() { F(B); }

// CHECK:STDOUT: --- fail_incomplete_in_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.130: type = class_type @A, @A(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %A.elem.1ce: type = unbound_element_type %A.130, %T [symbolic]
// CHECK:STDOUT:   %struct_type.v.ff1: type = struct_type {.v: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.460: <witness> = complete_type_witness %struct_type.v.ff1 [symbolic]
// CHECK:STDOUT:   %A.1d4: type = class_type @A, @A(%B) [concrete]
// CHECK:STDOUT:   %pattern_type.1b7: type = pattern_type %A.1d4 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.elem.4ce: type = unbound_element_type %A.1d4, %B [concrete]
// CHECK:STDOUT:   %struct_type.v.d45: type = struct_type {.v: %B} [concrete]
// CHECK:STDOUT:   %complete_type.1df: <witness> = complete_type_witness %struct_type.v.d45 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .B = %B.decl.loc4
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl.loc4: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc6_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %pattern_type.1b7 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.1b7 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %A.1d4 = value_param call_param0
// CHECK:STDOUT:     %.loc27: type = splice_block %A [concrete = constants.%A.1d4] {
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl.loc4 [concrete = constants.%B]
// CHECK:STDOUT:       %A: type = class_type @A, @A(constants.%B) [concrete = constants.%A.1d4]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %A.1d4 = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc29: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%T.loc6_9.2: type) {
// CHECK:STDOUT:   %T.loc6_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc6_9.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%T.loc6_9.1) [symbolic = %A (constants.%A.130)]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %T.loc6_9.1 [symbolic = %A.elem (constants.%A.elem.1ce)]
// CHECK:STDOUT:   %struct_type.v.loc14_1.2: type = struct_type {.v: @A.%T.loc6_9.1 (%T)} [symbolic = %struct_type.v.loc14_1.2 (constants.%struct_type.v.ff1)]
// CHECK:STDOUT:   %complete_type.loc14_1.2: <witness> = complete_type_witness %struct_type.v.loc14_1.2 [symbolic = %complete_type.loc14_1.2 (constants.%complete_type.460)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_9.2 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc13: @A.%A.elem (%A.elem.1ce) = field_decl v, element0 [concrete]
// CHECK:STDOUT:     %struct_type.v.loc14_1.1: type = struct_type {.v: %T} [symbolic = %struct_type.v.loc14_1.2 (constants.%struct_type.v.ff1)]
// CHECK:STDOUT:     %complete_type.loc14_1.1: <witness> = complete_type_witness %struct_type.v.loc14_1.1 [symbolic = %complete_type.loc14_1.2 (constants.%complete_type.460)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc14_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.130
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .v = %.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %A.1d4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.loc6_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%B) {
// CHECK:STDOUT:   %T.loc6_9.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %A => constants.%A.1d4
// CHECK:STDOUT:   %A.elem => constants.%A.elem.4ce
// CHECK:STDOUT:   %struct_type.v.loc14_1.2 => constants.%struct_type.v.d45
// CHECK:STDOUT:   %complete_type.loc14_1.2 => constants.%complete_type.1df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- incomplete_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.713: %Destroy.type = facet_value %T, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.a63: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.713 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.a63 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Op.1(%Destroy.facet.713) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%B) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.39d: <witness> = impl_witness imports.%Destroy.impl_witness_table, @Destroy.impl(%B) [concrete]
// CHECK:STDOUT:   %Op.type.1a6: type = fn_type @Op.2, @Destroy.impl(%B) [concrete]
// CHECK:STDOUT:   %Op.3c2: %Op.type.1a6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %complete_type.3bf: <witness> = complete_type_witness %ptr.e79 [concrete]
// CHECK:STDOUT:   %Destroy.facet.331: %Destroy.type = facet_value %B, (%Destroy.impl_witness.39d) [concrete]
// CHECK:STDOUT:   %.c16: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.331 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.3c2, @Op.2(%B) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @Destroy.impl.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @Destroy.impl.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @Destroy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .B = %B.decl.loc4
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl.loc4: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %B.decl.loc13: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.2: type) {
// CHECK:STDOUT:   %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_10: <witness> = require_complete_type %T.loc6_6.1 [symbolic = %require_complete.loc7_10 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc6_6.1 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc6_6.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %T.loc6_6.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.713)]
// CHECK:STDOUT:   %.loc7_3.2: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet [symbolic = %.loc7_3.2 (constants.%.a63)]
// CHECK:STDOUT:   %impl.elem0.loc7_3.2: @F.%.loc7_3.2 (%.a63) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2: <specific function> = specific_impl_function %impl.elem0.loc7_3.2, @Op.1(%Destroy.facet) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.loc6_6.1 [symbolic = %ptr (constants.%ptr.79f)]
// CHECK:STDOUT:   %require_complete.loc7_3: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc7_3 (constants.%require_complete.6e5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type (%pattern_type.7dc) = binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type (%pattern_type.7dc) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%T.loc6_6.1 (%T) = var %v.var_patt
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %v: ref @F.%T.loc6_6.1 (%T) = bind_name v, %v.var
// CHECK:STDOUT:     %impl.elem0.loc7_3.1: @F.%.loc7_3.2 (%.a63) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc7_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_3.1: <specific function> = specific_impl_function %impl.elem0.loc7_3.1, @Op.1(constants.%Destroy.facet.713) [symbolic = %specific_impl_fn.loc7_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc7_3.1
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.79f) = addr_of %v.var
// CHECK:STDOUT:     %.loc7_3.1: init %empty_tuple.type = call %bound_method.loc7_3.2(%addr)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl.loc4 [concrete = constants.%B]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%B) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%B) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_10 => constants.%complete_type.357
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.049
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.39d
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.331
// CHECK:STDOUT:   %.loc7_3.2 => constants.%.c16
// CHECK:STDOUT:   %impl.elem0.loc7_3.2 => constants.%Op.3c2
// CHECK:STDOUT:   %specific_impl_fn.loc7_3.2 => constants.%Op.specific_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.e79
// CHECK:STDOUT:   %require_complete.loc7_3 => constants.%complete_type.3bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_incomplete_in_function_at_eof.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.713: %Destroy.type = facet_value %T, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.a63: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.713 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.a63 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Op.1(%Destroy.facet.713) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%B) [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.39d: <witness> = impl_witness imports.%Destroy.impl_witness_table, @Destroy.impl(%B) [concrete]
// CHECK:STDOUT:   %Op.type.1a6: type = fn_type @Op.2, @Destroy.impl(%B) [concrete]
// CHECK:STDOUT:   %Op.3c2: %Op.type.1a6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %ptr.e79 [concrete]
// CHECK:STDOUT:   %Destroy.facet.331: %Destroy.type = facet_value %B, (%Destroy.impl_witness.39d) [concrete]
// CHECK:STDOUT:   %.c16: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.331 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.3c2, @Op.2(%B) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @Destroy.impl.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @Destroy.impl.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @Destroy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B;
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.2: type) {
// CHECK:STDOUT:   %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_10: <witness> = require_complete_type %T.loc6_6.1 [symbolic = %require_complete.loc13_10 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc6_6.1 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc6_6.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %T.loc6_6.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.713)]
// CHECK:STDOUT:   %.loc13_3.2: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet [symbolic = %.loc13_3.2 (constants.%.a63)]
// CHECK:STDOUT:   %impl.elem0.loc13_3.2: @F.%.loc13_3.2 (%.a63) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc13_3.2: <specific function> = specific_impl_function %impl.elem0.loc13_3.2, @Op.1(%Destroy.facet) [symbolic = %specific_impl_fn.loc13_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.loc6_6.1 [symbolic = %ptr (constants.%ptr.79f)]
// CHECK:STDOUT:   %require_complete.loc13_3: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc13_3 (constants.%require_complete.6e5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type (%pattern_type.7dc) = binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type (%pattern_type.7dc) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%T.loc6_6.1 (%T) = var %v.var_patt
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %v: ref @F.%T.loc6_6.1 (%T) = bind_name v, %v.var
// CHECK:STDOUT:     %impl.elem0.loc13_3.1: @F.%.loc13_3.2 (%.a63) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc13_3.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc13_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc13_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc13_3.1: <specific function> = specific_impl_function %impl.elem0.loc13_3.1, @Op.1(constants.%Destroy.facet.713) [symbolic = %specific_impl_fn.loc13_3.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc13_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc13_3.1
// CHECK:STDOUT:     %addr: @F.%ptr (%ptr.79f) = addr_of %v.var
// CHECK:STDOUT:     %.loc13_3.1: init %empty_tuple.type = call %bound_method.loc13_3.2(%addr)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%B) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%B) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_10 => <error>
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.049
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%Destroy.impl_witness.39d
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.331
// CHECK:STDOUT:   %.loc13_3.2 => constants.%.c16
// CHECK:STDOUT:   %impl.elem0.loc13_3.2 => constants.%Op.3c2
// CHECK:STDOUT:   %specific_impl_fn.loc13_3.2 => constants.%Op.specific_fn
// CHECK:STDOUT:   %ptr => constants.%ptr.e79
// CHECK:STDOUT:   %require_complete.loc13_3 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
