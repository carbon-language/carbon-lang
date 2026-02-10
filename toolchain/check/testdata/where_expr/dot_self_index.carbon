// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/dot_self_index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/dot_self_index.carbon

interface Empty(W:! type) {
  let A:! type;
}

// T has index 0
// .Self has index invalid, not 1
// V has index 1, does not match .Self
//@dump-sem-ir-begin
fn H(T:! type, U: Empty(T) where .A = T*, V:! type) {}
//@dump-sem-ir-end

fn G(U: Empty(i32) where .A = i32*) {
  H(i32, U, bool);
}

// CHECK:STDOUT: --- dot_self_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Empty.type.d36: type = generic_interface_type @Empty [concrete]
// CHECK:STDOUT:   %Empty.generic: %Empty.type.d36 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Empty.type.6f76f1.2: type = facet_type <@Empty, @Empty(%T)> [symbolic]
// CHECK:STDOUT:   %.Self.b4b: %Empty.type.6f76f1.2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %Empty.assoc_type.ce4b5a.2: type = assoc_entity_type @Empty, @Empty(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.beb632.2: %Empty.assoc_type.ce4b5a.2 = assoc_entity element0, @Empty.%A [symbolic]
// CHECK:STDOUT:   %require_complete.8eb: <witness> = require_complete_type %Empty.type.6f76f1.2 [symbolic]
// CHECK:STDOUT:   %.Self.binding.as_type.b61: type = symbolic_binding_type .Self, %.Self.b4b [symbolic]
// CHECK:STDOUT:   %Empty.lookup_impl_witness.177: <witness> = lookup_impl_witness %.Self.b4b, @Empty, @Empty(%T) [symbolic]
// CHECK:STDOUT:   %impl.elem0.de5: type = impl_witness_access %Empty.lookup_impl_witness.177, element0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %Empty_where.type.2dc: type = facet_type <@Empty, @Empty(%T) where %impl.elem0.de5 = %ptr.e8f> [symbolic]
// CHECK:STDOUT:   %pattern_type.bb3: type = pattern_type %Empty_where.type.2dc [symbolic]
// CHECK:STDOUT:   %V: type = symbolic_binding V, 1 [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6bc: <witness> = require_complete_type %Empty_where.type.2dc [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Empty.type.4fb: type = facet_type <@Empty, @Empty(%i32)> [concrete]
// CHECK:STDOUT:   %.Self.ec6: %Empty.type.4fb = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Empty.assoc_type.623: type = assoc_entity_type @Empty, @Empty(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.edc: %Empty.assoc_type.623 = assoc_entity element0, @Empty.%A [concrete]
// CHECK:STDOUT:   %.Self.binding.as_type.63c: type = symbolic_binding_type .Self, %.Self.ec6 [symbolic_self]
// CHECK:STDOUT:   %Empty.lookup_impl_witness.924: <witness> = lookup_impl_witness %.Self.ec6, @Empty, @Empty(%i32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.238: type = impl_witness_access %Empty.lookup_impl_witness.924, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Empty_where.type.729: type = facet_type <@Empty, @Empty(%i32) where %impl.elem0.238 = %ptr.235> [concrete]
// CHECK:STDOUT:   %pattern_type.ee5: type = pattern_type %Empty_where.type.729 [concrete]
// CHECK:STDOUT:   %complete_type.ba8: <witness> = complete_type_witness %Empty.type.4fb [concrete]
// CHECK:STDOUT:   %complete_type.863: <witness> = complete_type_witness %Empty_where.type.729 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @H.%pattern_type (%pattern_type.bb3) = value_binding_pattern U [concrete]
// CHECK:STDOUT:     %U.param_patt: @H.%pattern_type (%pattern_type.bb3) = value_param_pattern %U.patt, call_param0 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc21_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc21_6.1 (constants.%T)]
// CHECK:STDOUT:     %U.param: @H.%Empty_where.type (%Empty_where.type.2dc) = value_param call_param0
// CHECK:STDOUT:     %.loc21_28.1: type = splice_block %.loc21_28.2 [symbolic = %Empty_where.type (constants.%Empty_where.type.2dc)] {
// CHECK:STDOUT:       %Empty.ref: %Empty.type.d36 = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.generic]
// CHECK:STDOUT:       %T.ref.loc21_25: type = name_ref T, %T.loc21_6.2 [symbolic = %T.loc21_6.1 (constants.%T)]
// CHECK:STDOUT:       %Empty.type.loc21_26.2: type = facet_type <@Empty, @Empty(constants.%T)> [symbolic = %Empty.type.loc21_26.1 (constants.%Empty.type.6f76f1.2)]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: @H.%Empty.type.loc21_26.1 (%Empty.type.6f76f1.2) = name_ref .Self, %.Self.4 [symbolic = %.Self.1 (constants.%.Self.b4b)]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.b61)]
// CHECK:STDOUT:       %.loc21_34.1: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.b61)]
// CHECK:STDOUT:       %.loc21_34.2: @H.%Empty.assoc_type (%Empty.assoc_type.ce4b5a.2) = specific_constant @A.%assoc0, @Empty(constants.%T, constants.%.Self.b4b) [symbolic = %assoc0 (constants.%assoc0.beb632.2)]
// CHECK:STDOUT:       %A.ref: @H.%Empty.assoc_type (%Empty.assoc_type.ce4b5a.2) = name_ref A, %.loc21_34.2 [symbolic = %assoc0 (constants.%assoc0.beb632.2)]
// CHECK:STDOUT:       %impl.elem0.loc21_34.2: type = impl_witness_access constants.%Empty.lookup_impl_witness.177, element0 [symbolic = %impl.elem0.loc21_34.1 (constants.%impl.elem0.de5)]
// CHECK:STDOUT:       %T.ref.loc21_39: type = name_ref T, %T.loc21_6.2 [symbolic = %T.loc21_6.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc21_40.2: type = ptr_type %T.ref.loc21_39 [symbolic = %ptr.loc21_40.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:       %.loc21_28.2: type = where_expr %.Self.4 [symbolic = %Empty_where.type (constants.%Empty_where.type.2dc)] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%Empty.type.6f76f1.2
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc21_34.2, %ptr.loc21_40.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U: @H.%Empty_where.type (%Empty_where.type.2dc) = value_binding U, %U.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %V.loc21_43.2: type = symbolic_binding V, 1 [symbolic = %V.loc21_43.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%T.loc21_6.2: type, %V.loc21_43.2: type) {
// CHECK:STDOUT:   %T.loc21_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc21_6.1 (constants.%T)]
// CHECK:STDOUT:   %Empty.type.loc21_26.1: type = facet_type <@Empty, @Empty(%T.loc21_6.1)> [symbolic = %Empty.type.loc21_26.1 (constants.%Empty.type.6f76f1.2)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc21_34: <witness> = require_complete_type %Empty.type.loc21_26.1 [symbolic = %require_complete.loc21_34 (constants.%require_complete.8eb)]
// CHECK:STDOUT:   %.Self.binding.as_type: type = symbolic_binding_type .Self, %.Self.1 [symbolic = %.Self.binding.as_type (constants.%.Self.binding.as_type.b61)]
// CHECK:STDOUT:   %Empty.assoc_type: type = assoc_entity_type @Empty, @Empty(%T.loc21_6.1) [symbolic = %Empty.assoc_type (constants.%Empty.assoc_type.ce4b5a.2)]
// CHECK:STDOUT:   %assoc0: @H.%Empty.assoc_type (%Empty.assoc_type.ce4b5a.2) = assoc_entity element0, @Empty.%A [symbolic = %assoc0 (constants.%assoc0.beb632.2)]
// CHECK:STDOUT:   %Empty.lookup_impl_witness: <witness> = lookup_impl_witness %.Self.1, @Empty, @Empty(%T.loc21_6.1) [symbolic = %Empty.lookup_impl_witness (constants.%Empty.lookup_impl_witness.177)]
// CHECK:STDOUT:   %impl.elem0.loc21_34.1: type = impl_witness_access %Empty.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc21_34.1 (constants.%impl.elem0.de5)]
// CHECK:STDOUT:   %ptr.loc21_40.1: type = ptr_type %T.loc21_6.1 [symbolic = %ptr.loc21_40.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %Empty_where.type: type = facet_type <@Empty, @Empty(%T.loc21_6.1) where %impl.elem0.loc21_34.1 = %ptr.loc21_40.1> [symbolic = %Empty_where.type (constants.%Empty_where.type.2dc)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Empty_where.type [symbolic = %pattern_type (constants.%pattern_type.bb3)]
// CHECK:STDOUT:   %V.loc21_43.1: type = symbolic_binding V, 1 [symbolic = %V.loc21_43.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc21_17: <witness> = require_complete_type %Empty_where.type [symbolic = %require_complete.loc21_17 (constants.%require_complete.6bc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.param: @H.%Empty_where.type (%Empty_where.type.2dc)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%T, constants.%V) {
// CHECK:STDOUT:   %T.loc21_6.1 => constants.%T
// CHECK:STDOUT:   %Empty.type.loc21_26.1 => constants.%Empty.type.6f76f1.2
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.b4b
// CHECK:STDOUT:   %require_complete.loc21_34 => constants.%require_complete.8eb
// CHECK:STDOUT:   %.Self.binding.as_type => constants.%.Self.binding.as_type.b61
// CHECK:STDOUT:   %Empty.assoc_type => constants.%Empty.assoc_type.ce4b5a.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.beb632.2
// CHECK:STDOUT:   %Empty.lookup_impl_witness => constants.%Empty.lookup_impl_witness.177
// CHECK:STDOUT:   %impl.elem0.loc21_34.1 => constants.%impl.elem0.de5
// CHECK:STDOUT:   %ptr.loc21_40.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %Empty_where.type => constants.%Empty_where.type.2dc
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bb3
// CHECK:STDOUT:   %V.loc21_43.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%i32, bool) {
// CHECK:STDOUT:   %T.loc21_6.1 => constants.%i32
// CHECK:STDOUT:   %Empty.type.loc21_26.1 => constants.%Empty.type.4fb
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.ec6
// CHECK:STDOUT:   %require_complete.loc21_34 => constants.%complete_type.ba8
// CHECK:STDOUT:   %.Self.binding.as_type => constants.%.Self.binding.as_type.63c
// CHECK:STDOUT:   %Empty.assoc_type => constants.%Empty.assoc_type.623
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.edc
// CHECK:STDOUT:   %Empty.lookup_impl_witness => constants.%Empty.lookup_impl_witness.924
// CHECK:STDOUT:   %impl.elem0.loc21_34.1 => constants.%impl.elem0.238
// CHECK:STDOUT:   %ptr.loc21_40.1 => constants.%ptr.235
// CHECK:STDOUT:   %Empty_where.type => constants.%Empty_where.type.729
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ee5
// CHECK:STDOUT:   %V.loc21_43.1 => bool
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc21_17 => constants.%complete_type.863
// CHECK:STDOUT: }
// CHECK:STDOUT:
