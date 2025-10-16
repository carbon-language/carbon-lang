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
  var v: T*;
  *v;
}

// F(B) isn't resolved until the end of the file.
fn G() { F(B); }

class B {}

// --- fail_incomplete_in_function_at_eof.carbon

library "[[@TEST_NAME]]";

class B;

fn F(T:! type) {
  var v: T*;
  // CHECK:STDERR: fail_incomplete_in_function_at_eof.carbon:[[@LINE+6]]:3: error: `T` evaluates to incomplete type `B` [IncompleteTypeInMonomorphization]
  // CHECK:STDERR:   *v;
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_incomplete_in_function_at_eof.carbon:[[@LINE-7]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class B;
  // CHECK:STDERR: ^~~~~~~~
  *v;
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
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
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %pattern_type.1b7 = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.1b7 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %A.1d4 = value_param call_param0
// CHECK:STDOUT:     %.loc27: type = splice_block %A [concrete = constants.%A.1d4] {
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl.loc4 [concrete = constants.%B]
// CHECK:STDOUT:       %A: type = class_type @A, @A(constants.%B) [concrete = constants.%A.1d4]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %A.1d4 = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc29: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: @A.%T.loc6_9.1 (%T)} [symbolic = %struct_type.v (constants.%struct_type.v.ff1)]
// CHECK:STDOUT:   %complete_type.loc14_1.2: <witness> = complete_type_witness %struct_type.v [symbolic = %complete_type.loc14_1.2 (constants.%complete_type.460)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc6_9.2 [symbolic = %T.loc6_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc13: @A.%A.elem (%A.elem.1ce) = field_decl v, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc14_1.1: <witness> = complete_type_witness constants.%struct_type.v.ff1 [symbolic = %complete_type.loc14_1.2 (constants.%complete_type.460)]
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
// CHECK:STDOUT:   %struct_type.v => constants.%struct_type.v.d45
// CHECK:STDOUT:   %complete_type.loc14_1.2 => constants.%complete_type.1df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- incomplete_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %DestroyT: %type_where = bind_symbolic_name DestroyT, 0 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b5d: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%DestroyT) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.544: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value.57a: %type_where = facet_value %ptr.79f, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.5e2: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.57a) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.ca5: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.57a) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.7c0: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.ca5 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.a13: type = ptr_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %require_complete.132: <witness> = require_complete_type %ptr.a13 [symbolic]
// CHECK:STDOUT:   %Destroy.facet.136: %Destroy.type = facet_value %ptr.79f, (%Destroy.impl_witness.5e2) [symbolic]
// CHECK:STDOUT:   %.0f0: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.136 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.195: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.7c0, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.57a) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%B) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %complete_type.3bf: <witness> = complete_type_witness %ptr.e79 [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %facet_value.888: %type_where = facet_value %ptr.e79, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.ee7: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.888) [concrete]
// CHECK:STDOUT:   %Destroy.facet.4ee: %Destroy.type = facet_value %ptr.e79, (%Destroy.impl_witness.ee7) [concrete]
// CHECK:STDOUT:   %.9fa: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.4ee [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.ba6: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.888) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.3f1: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.ba6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.3fb: type = ptr_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.8ba: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.3f1, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.888) [concrete]
// CHECK:STDOUT:   %complete_type.c9f: <witness> = complete_type_witness %ptr.3fb [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.950: @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.b5d) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.544)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.950), @DestroyT.binding.as_type.as.Destroy.impl [concrete]
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
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %B.decl.loc14: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %ptr.loc7_11.2: type = ptr_type %T.loc6_6.1 [symbolic = %ptr.loc7_11.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %require_complete.loc7_11: <witness> = require_complete_type %ptr.loc7_11.2 [symbolic = %require_complete.loc7_11 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc7_11.2 [symbolic = %pattern_type (constants.%pattern_type.afe)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %T.loc6_6.1 [symbolic = %require_complete.loc8 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %ptr.loc7_11.2, () [symbolic = %facet_value (constants.%facet_value.57a)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.5e2)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr.loc7_11.2, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.136)]
// CHECK:STDOUT:   %.loc7_3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc7_3 (constants.%.0f0)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.ca5)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op: @F.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.ca5) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.7c0)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.195)]
// CHECK:STDOUT:   %ptr.loc7_3: type = ptr_type %ptr.loc7_11.2 [symbolic = %ptr.loc7_3 (constants.%ptr.a13)]
// CHECK:STDOUT:   %require_complete.loc7_3: <witness> = require_complete_type %ptr.loc7_3 [symbolic = %require_complete.loc7_3 (constants.%require_complete.132)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type (%pattern_type.afe) = ref_binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type (%pattern_type.afe) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%ptr.loc7_11.2 (%ptr.79f) = var %v.var_patt
// CHECK:STDOUT:     %.loc7_11: type = splice_block %ptr.loc7_11.1 [symbolic = %ptr.loc7_11.2 (constants.%ptr.79f)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc7_11.1: type = ptr_type %T.ref [symbolic = %ptr.loc7_11.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%ptr.loc7_11.2 (%ptr.79f) = ref_binding v, %v.var
// CHECK:STDOUT:     %v.ref: ref @F.%ptr.loc7_11.2 (%ptr.79f) = name_ref v, %v
// CHECK:STDOUT:     %.loc8_4: @F.%ptr.loc7_11.2 (%ptr.79f) = bind_value %v.ref
// CHECK:STDOUT:     %.loc8_3: ref @F.%T.loc6_6.1 (%T) = deref %.loc8_4
// CHECK:STDOUT:     %impl.elem0: @F.%.loc7_3 (%.0f0) = impl_witness_access constants.%Destroy.impl_witness.5e2, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.7c0)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %v.var, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.57a) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.195)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %v.var, %specific_fn
// CHECK:STDOUT:     %addr: @F.%ptr.loc7_3 (%ptr.a13) = addr_of %v.var
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc7_3.2(%addr)
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
// CHECK:STDOUT:   %ptr.loc7_11.2 => constants.%ptr.e79
// CHECK:STDOUT:   %require_complete.loc7_11 => constants.%complete_type.3bf
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.960
// CHECK:STDOUT:   %require_complete.loc8 => constants.%complete_type.357
// CHECK:STDOUT:   %facet_value => constants.%facet_value.888
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.ee7
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.4ee
// CHECK:STDOUT:   %.loc7_3 => constants.%.9fa
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.ba6
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.3f1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.8ba
// CHECK:STDOUT:   %ptr.loc7_3 => constants.%ptr.3fb
// CHECK:STDOUT:   %require_complete.loc7_3 => constants.%complete_type.c9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_incomplete_in_function_at_eof.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %DestroyT: %type_where = bind_symbolic_name DestroyT, 0 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b5d: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%DestroyT) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.544: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value.57a: %type_where = facet_value %ptr.79f, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.5e2: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.57a) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.ca5: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.57a) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.7c0: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.ca5 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.a13: type = ptr_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %require_complete.132: <witness> = require_complete_type %ptr.a13 [symbolic]
// CHECK:STDOUT:   %Destroy.facet.136: %Destroy.type = facet_value %ptr.79f, (%Destroy.impl_witness.5e2) [symbolic]
// CHECK:STDOUT:   %.0f0: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.136 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.195: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.7c0, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.57a) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%B) [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %complete_type.3bf: <witness> = complete_type_witness %ptr.e79 [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %facet_value.888: %type_where = facet_value %ptr.e79, () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.ee7: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.888) [concrete]
// CHECK:STDOUT:   %Destroy.facet.4ee: %Destroy.type = facet_value %ptr.e79, (%Destroy.impl_witness.ee7) [concrete]
// CHECK:STDOUT:   %.9fa: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.4ee [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.ba6: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.888) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.3f1: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.ba6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.3fb: type = ptr_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.8ba: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.3f1, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.888) [concrete]
// CHECK:STDOUT:   %complete_type.c9f: <witness> = complete_type_witness %ptr.3fb [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.950: @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.b5d) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.544)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.950), @DestroyT.binding.as_type.as.Destroy.impl [concrete]
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
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
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
// CHECK:STDOUT:   %ptr.loc7_11.2: type = ptr_type %T.loc6_6.1 [symbolic = %ptr.loc7_11.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %require_complete.loc7_11: <witness> = require_complete_type %ptr.loc7_11.2 [symbolic = %require_complete.loc7_11 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc7_11.2 [symbolic = %pattern_type (constants.%pattern_type.afe)]
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %T.loc6_6.1 [symbolic = %require_complete.loc14 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %ptr.loc7_11.2, () [symbolic = %facet_value (constants.%facet_value.57a)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.5e2)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr.loc7_11.2, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.136)]
// CHECK:STDOUT:   %.loc7_3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc7_3 (constants.%.0f0)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.ca5)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op: @F.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.ca5) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.7c0)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.195)]
// CHECK:STDOUT:   %ptr.loc7_3: type = ptr_type %ptr.loc7_11.2 [symbolic = %ptr.loc7_3 (constants.%ptr.a13)]
// CHECK:STDOUT:   %require_complete.loc7_3: <witness> = require_complete_type %ptr.loc7_3 [symbolic = %require_complete.loc7_3 (constants.%require_complete.132)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @F.%pattern_type (%pattern_type.afe) = ref_binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @F.%pattern_type (%pattern_type.afe) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @F.%ptr.loc7_11.2 (%ptr.79f) = var %v.var_patt
// CHECK:STDOUT:     %.loc7_11: type = splice_block %ptr.loc7_11.1 [symbolic = %ptr.loc7_11.2 (constants.%ptr.79f)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc7_11.1: type = ptr_type %T.ref [symbolic = %ptr.loc7_11.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%ptr.loc7_11.2 (%ptr.79f) = ref_binding v, %v.var
// CHECK:STDOUT:     %v.ref: ref @F.%ptr.loc7_11.2 (%ptr.79f) = name_ref v, %v
// CHECK:STDOUT:     %.loc14_4: @F.%ptr.loc7_11.2 (%ptr.79f) = bind_value %v.ref
// CHECK:STDOUT:     %.loc14_3: ref @F.%T.loc6_6.1 (%T) = deref %.loc14_4
// CHECK:STDOUT:     %impl.elem0: @F.%.loc7_3 (%.0f0) = impl_witness_access constants.%Destroy.impl_witness.5e2, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.7c0)]
// CHECK:STDOUT:     %bound_method.loc7_3.1: <bound method> = bound_method %v.var, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.57a) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.195)]
// CHECK:STDOUT:     %bound_method.loc7_3.2: <bound method> = bound_method %v.var, %specific_fn
// CHECK:STDOUT:     %addr: @F.%ptr.loc7_3 (%ptr.a13) = addr_of %v.var
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc7_3.2(%addr)
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
// CHECK:STDOUT:   %ptr.loc7_11.2 => constants.%ptr.e79
// CHECK:STDOUT:   %require_complete.loc7_11 => constants.%complete_type.3bf
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.960
// CHECK:STDOUT:   %require_complete.loc14 => <error>
// CHECK:STDOUT:   %facet_value => constants.%facet_value.888
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.ee7
// CHECK:STDOUT:   %Destroy.facet => constants.%Destroy.facet.4ee
// CHECK:STDOUT:   %.loc7_3 => constants.%.9fa
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.ba6
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.3f1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn => constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.8ba
// CHECK:STDOUT:   %ptr.loc7_3 => constants.%ptr.3fb
// CHECK:STDOUT:   %require_complete.loc7_3 => constants.%complete_type.c9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
