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

interface Empty(W: type) {
  let A: type;
}

// T has index 0
// .Self has index invalid, not 1
// V has index 1, does not match .Self
//@dump-sem-ir-begin
fn H(generic T: type, unused U: Empty(T) where .A = T*, unused generic V: type) {}
//@dump-sem-ir-end

fn G(U: Empty(i32) where .A = i32*) {
  H(i32, U, bool);
}

// CHECK:STDOUT: --- dot_self_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.e58: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Empty.type.c52: type = generic_interface_type @Empty [concrete]
// CHECK:STDOUT:   %Empty.generic: %Empty.type.c52 = struct_value () [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Empty.type.f57232.2: type = facet_type <@Empty, @Empty(%T)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.3b0: %Empty.type.f57232.2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %Empty.assoc_type.52a113.2: type = assoc_entity_type @Empty, @Empty(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.280be6.2: %Empty.assoc_type.52a113.2 = assoc_entity element0, @Empty.WithSelf.%A [symbolic]
// CHECK:STDOUT:   %require_complete.4f0: <witness> = require_complete_type %Empty.type.f57232.2 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.dd3: type = facet_access_type %.Self.frozen.3b0 [symbolic]
// CHECK:STDOUT:   %Empty.lookup_impl_witness.cfd: <witness> = lookup_impl_witness %.Self.frozen.3b0, @Empty, @Empty(%T) [symbolic]
// CHECK:STDOUT:   %impl.elem0.14d: type = impl_witness_access %Empty.lookup_impl_witness.cfd, element0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %.Self.4bb: %Empty.type.f57232.2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %Empty.lookup_impl_witness.ea2: <witness> = lookup_impl_witness %.Self.4bb, @Empty, @Empty(%T) [symbolic]
// CHECK:STDOUT:   %impl.elem0.2a8: type = impl_witness_access %Empty.lookup_impl_witness.ea2, element0 [symbolic]
// CHECK:STDOUT:   %Empty_where.type.4e8: type = facet_type <@Empty, @Empty(%T) where %impl.elem0.2a8 = %ptr.e8f> [symbolic]
// CHECK:STDOUT:   %pattern_type.67b: type = pattern_type %Empty_where.type.4e8 [symbolic]
// CHECK:STDOUT:   %U.param_patt.6e9: %pattern_type.67b = value_param_pattern [symbolic]
// CHECK:STDOUT:   %U.patt.a80: %pattern_type.67b = wrapper_binding_pattern U, %U.param_patt.6e9 [symbolic]
// CHECK:STDOUT:   %V.patt: %pattern_type.9a5 = symbolic_binding_pattern V, 1 [symbolic]
// CHECK:STDOUT:   %V: type = symbolic_binding V, 1 [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.980: <witness> = require_complete_type %Empty_where.type.4e8 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Empty.type.8d5: type = facet_type <@Empty, @Empty(%i32)> [concrete]
// CHECK:STDOUT:   %.Self.frozen.90d: %Empty.type.8d5 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Empty.assoc_type.c92: type = assoc_entity_type @Empty, @Empty(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.19c: %Empty.assoc_type.c92 = assoc_entity element0, @Empty.WithSelf.%A [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.636: type = facet_access_type %.Self.frozen.90d [symbolic_self]
// CHECK:STDOUT:   %Empty.lookup_impl_witness.609: <witness> = lookup_impl_witness %.Self.frozen.90d, @Empty, @Empty(%i32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.0f9: type = impl_witness_access %Empty.lookup_impl_witness.609, element0 [symbolic_self]
// CHECK:STDOUT:   %ptr.1fe: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %.Self.401: %Empty.type.8d5 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Empty.lookup_impl_witness.25f: <witness> = lookup_impl_witness %.Self.401, @Empty, @Empty(%i32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.2ec: type = impl_witness_access %Empty.lookup_impl_witness.25f, element0 [symbolic_self]
// CHECK:STDOUT:   %Empty_where.type.57b: type = facet_type <@Empty, @Empty(%i32) where %impl.elem0.2ec = %ptr.1fe> [concrete]
// CHECK:STDOUT:   %pattern_type.292: type = pattern_type %Empty_where.type.57b [concrete]
// CHECK:STDOUT:   %U.param_patt.cae: %pattern_type.292 = value_param_pattern [concrete]
// CHECK:STDOUT:   %complete_type.430: <witness> = complete_type_witness %Empty.type.8d5 [concrete]
// CHECK:STDOUT:   %U.patt.0f3348.2: %pattern_type.292 = wrapper_binding_pattern U, %U.param_patt.cae [concrete]
// CHECK:STDOUT:   %complete_type.229: <witness> = complete_type_witness %Empty_where.type.57b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %T.patt.loc21_15.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.param_patt.loc21_31.1: @H.%pattern_type (%pattern_type.67b) = value_param_pattern [symbolic = %U.param_patt.loc21_31.2 (constants.%U.param_patt.6e9)]
// CHECK:STDOUT:     %U.patt.loc21_31.1: @H.%pattern_type (%pattern_type.67b) = wrapper_binding_pattern U, %U.param_patt.loc21_31.1 [symbolic = %U.patt.loc21_31.2 (constants.%U.patt.a80)]
// CHECK:STDOUT:     %V.patt.loc21_73.1: %pattern_type.9a5 = symbolic_binding_pattern V, 1 [symbolic = %V.patt.loc21_73.2 (constants.%V.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_17.1: type = splice_block %.loc21_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc21_15: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %.loc21_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc21_15.1 (constants.%T)]
// CHECK:STDOUT:     %U.param: @H.%Empty_where.type (%Empty_where.type.4e8) = value_param call_param0
// CHECK:STDOUT:     %.loc21_42.1: type = splice_block %.loc21_42.2 [symbolic = %Empty_where.type (constants.%Empty_where.type.4e8)] {
// CHECK:STDOUT:       %Empty.ref: %Empty.type.c52 = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.generic]
// CHECK:STDOUT:       %T.ref.loc21_39: type = name_ref T, %T.loc21_15.2 [symbolic = %T.loc21_15.1 (constants.%T)]
// CHECK:STDOUT:       %Empty.type.loc21_40.2: type = facet_type <@Empty, @Empty(constants.%T)> [symbolic = %Empty.type.loc21_40.1 (constants.%Empty.type.f57232.2)]
// CHECK:STDOUT:       %.Self.frozen.loc21_42.2: @H.%Empty.type.loc21_40.1 (%Empty.type.f57232.2) = symbolic_binding .Self [symbolic = %.Self.frozen.loc21_42.1 (constants.%.Self.frozen.3b0)]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Empty.type.loc21_40.2 [concrete]
// CHECK:STDOUT:       %.Self.ref: @H.%Empty.type.loc21_40.1 (%Empty.type.f57232.2) = name_ref .Self, %.Self.frozen.loc21_42.2 [symbolic = %.Self.frozen.loc21_42.1 (constants.%.Self.frozen.3b0)]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.dd3)]
// CHECK:STDOUT:       %.loc21_48.1: type = converted %.Self.ref, %.Self.as_type [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.dd3)]
// CHECK:STDOUT:       %.loc21_48.2: @H.%Empty.assoc_type (%Empty.assoc_type.52a113.2) = specific_constant @A.%assoc0, @Empty.WithSelf(constants.%T, constants.%.Self.frozen.3b0) [symbolic = %assoc0 (constants.%assoc0.280be6.2)]
// CHECK:STDOUT:       %A.ref: @H.%Empty.assoc_type (%Empty.assoc_type.52a113.2) = name_ref A, %.loc21_48.2 [symbolic = %assoc0 (constants.%assoc0.280be6.2)]
// CHECK:STDOUT:       %impl.elem0.loc21_48.3: type = impl_witness_access constants.%Empty.lookup_impl_witness.cfd, element0 [symbolic = %impl.elem0.loc21_48.1 (constants.%impl.elem0.14d)]
// CHECK:STDOUT:       %T.ref.loc21_53: type = name_ref T, %T.loc21_15.2 [symbolic = %T.loc21_15.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc21_54.2: type = ptr_type %T.ref.loc21_53 [symbolic = %ptr.loc21_54.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:       %rewrite.loc21_51.1 = requirement_rewrite %impl.elem0.loc21_48.3, %ptr.loc21_54.2 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc21_48.4: type = impl_witness_access constants.%Empty.lookup_impl_witness.ea2, element0 [symbolic = %impl.elem0.loc21_48.2 (constants.%impl.elem0.2a8)]
// CHECK:STDOUT:       %rewrite.loc21_51.2 = requirement_rewrite %impl.elem0.loc21_48.4, %ptr.loc21_54.2 [concrete]
// CHECK:STDOUT:       %.loc21_42.2: type = where_expr [symbolic = %Empty_where.type (constants.%Empty_where.type.4e8)] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %Empty.type.loc21_40.2 [concrete]
// CHECK:STDOUT:         %rewrite.loc21_51.2 = requirement_rewrite %impl.elem0.loc21_48.4, %ptr.loc21_54.2 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U: @H.%Empty_where.type (%Empty_where.type.4e8) = wrapper_binding U, %U.param
// CHECK:STDOUT:     %.loc21_75.1: type = splice_block %.loc21_75.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc21_73: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %.loc21_75.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc21_73.2: type = symbolic_binding V, 1 [symbolic = %V.loc21_73.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%T.loc21_15.2: type, %V.loc21_73.2: type) {
// CHECK:STDOUT:   %T.patt.loc21_15.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc21_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc21_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc21_15.1 (constants.%T)]
// CHECK:STDOUT:   %Empty.type.loc21_40.1: type = facet_type <@Empty, @Empty(%T.loc21_15.1)> [symbolic = %Empty.type.loc21_40.1 (constants.%Empty.type.f57232.2)]
// CHECK:STDOUT:   %.Self.frozen.loc21_42.1: @H.%Empty.type.loc21_40.1 (%Empty.type.f57232.2) = symbolic_binding .Self [symbolic = %.Self.frozen.loc21_42.1 (constants.%.Self.frozen.3b0)]
// CHECK:STDOUT:   %require_complete.loc21_48: <witness> = require_complete_type %Empty.type.loc21_40.1 [symbolic = %require_complete.loc21_48 (constants.%require_complete.4f0)]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.loc21_42.1 [symbolic = %.Self.frozen.as_type (constants.%.Self.frozen.as_type.dd3)]
// CHECK:STDOUT:   %Empty.assoc_type: type = assoc_entity_type @Empty, @Empty(%T.loc21_15.1) [symbolic = %Empty.assoc_type (constants.%Empty.assoc_type.52a113.2)]
// CHECK:STDOUT:   %assoc0: @H.%Empty.assoc_type (%Empty.assoc_type.52a113.2) = assoc_entity element0, @Empty.WithSelf.%A [symbolic = %assoc0 (constants.%assoc0.280be6.2)]
// CHECK:STDOUT:   %Empty.lookup_impl_witness.loc21_48.1: <witness> = lookup_impl_witness %.Self.frozen.loc21_42.1, @Empty, @Empty(%T.loc21_15.1) [symbolic = %Empty.lookup_impl_witness.loc21_48.1 (constants.%Empty.lookup_impl_witness.cfd)]
// CHECK:STDOUT:   %impl.elem0.loc21_48.1: type = impl_witness_access %Empty.lookup_impl_witness.loc21_48.1, element0 [symbolic = %impl.elem0.loc21_48.1 (constants.%impl.elem0.14d)]
// CHECK:STDOUT:   %ptr.loc21_54.1: type = ptr_type %T.loc21_15.1 [symbolic = %ptr.loc21_54.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %.Self: @H.%Empty.type.loc21_40.1 (%Empty.type.f57232.2) = symbolic_binding .Self [symbolic = %.Self (constants.%.Self.4bb)]
// CHECK:STDOUT:   %Empty.lookup_impl_witness.loc21_48.2: <witness> = lookup_impl_witness %.Self, @Empty, @Empty(%T.loc21_15.1) [symbolic = %Empty.lookup_impl_witness.loc21_48.2 (constants.%Empty.lookup_impl_witness.ea2)]
// CHECK:STDOUT:   %impl.elem0.loc21_48.2: type = impl_witness_access %Empty.lookup_impl_witness.loc21_48.2, element0 [symbolic = %impl.elem0.loc21_48.2 (constants.%impl.elem0.2a8)]
// CHECK:STDOUT:   %Empty_where.type: type = facet_type <@Empty, @Empty(%T.loc21_15.1) where %impl.elem0.loc21_48.2 = %ptr.loc21_54.1> [symbolic = %Empty_where.type (constants.%Empty_where.type.4e8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Empty_where.type [symbolic = %pattern_type (constants.%pattern_type.67b)]
// CHECK:STDOUT:   %U.param_patt.loc21_31.2: @H.%pattern_type (%pattern_type.67b) = value_param_pattern [symbolic = %U.param_patt.loc21_31.2 (constants.%U.param_patt.6e9)]
// CHECK:STDOUT:   %U.patt.loc21_31.2: @H.%pattern_type (%pattern_type.67b) = wrapper_binding_pattern U, %U.param_patt.loc21_31.2 [symbolic = %U.patt.loc21_31.2 (constants.%U.patt.a80)]
// CHECK:STDOUT:   %V.patt.loc21_73.2: %pattern_type.9a5 = symbolic_binding_pattern V, 1 [symbolic = %V.patt.loc21_73.2 (constants.%V.patt)]
// CHECK:STDOUT:   %V.loc21_73.1: type = symbolic_binding V, 1 [symbolic = %V.loc21_73.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc21_31: <witness> = require_complete_type %Empty_where.type [symbolic = %require_complete.loc21_31 (constants.%require_complete.980)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.param: @H.%Empty_where.type (%Empty_where.type.4e8)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%T, constants.%V) {
// CHECK:STDOUT:   %T.patt.loc21_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc21_15.1 => constants.%T
// CHECK:STDOUT:   %Empty.type.loc21_40.1 => constants.%Empty.type.f57232.2
// CHECK:STDOUT:   %.Self.frozen.loc21_42.1 => constants.%.Self.frozen.3b0
// CHECK:STDOUT:   %require_complete.loc21_48 => constants.%require_complete.4f0
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.dd3
// CHECK:STDOUT:   %Empty.assoc_type => constants.%Empty.assoc_type.52a113.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.280be6.2
// CHECK:STDOUT:   %Empty.lookup_impl_witness.loc21_48.1 => constants.%Empty.lookup_impl_witness.cfd
// CHECK:STDOUT:   %impl.elem0.loc21_48.1 => constants.%impl.elem0.14d
// CHECK:STDOUT:   %ptr.loc21_54.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %.Self => constants.%.Self.4bb
// CHECK:STDOUT:   %Empty.lookup_impl_witness.loc21_48.2 => constants.%Empty.lookup_impl_witness.ea2
// CHECK:STDOUT:   %impl.elem0.loc21_48.2 => constants.%impl.elem0.2a8
// CHECK:STDOUT:   %Empty_where.type => constants.%Empty_where.type.4e8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.67b
// CHECK:STDOUT:   %U.param_patt.loc21_31.2 => constants.%U.param_patt.6e9
// CHECK:STDOUT:   %U.patt.loc21_31.2 => constants.%U.patt.a80
// CHECK:STDOUT:   %V.patt.loc21_73.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc21_73.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%i32, bool) {
// CHECK:STDOUT:   %T.patt.loc21_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc21_15.1 => constants.%i32
// CHECK:STDOUT:   %Empty.type.loc21_40.1 => constants.%Empty.type.8d5
// CHECK:STDOUT:   %.Self.frozen.loc21_42.1 => constants.%.Self.frozen.90d
// CHECK:STDOUT:   %require_complete.loc21_48 => constants.%complete_type.430
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.636
// CHECK:STDOUT:   %Empty.assoc_type => constants.%Empty.assoc_type.c92
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.19c
// CHECK:STDOUT:   %Empty.lookup_impl_witness.loc21_48.1 => constants.%Empty.lookup_impl_witness.609
// CHECK:STDOUT:   %impl.elem0.loc21_48.1 => constants.%impl.elem0.0f9
// CHECK:STDOUT:   %ptr.loc21_54.1 => constants.%ptr.1fe
// CHECK:STDOUT:   %.Self => constants.%.Self.401
// CHECK:STDOUT:   %Empty.lookup_impl_witness.loc21_48.2 => constants.%Empty.lookup_impl_witness.25f
// CHECK:STDOUT:   %impl.elem0.loc21_48.2 => constants.%impl.elem0.2ec
// CHECK:STDOUT:   %Empty_where.type => constants.%Empty_where.type.57b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.292
// CHECK:STDOUT:   %U.param_patt.loc21_31.2 => constants.%U.param_patt.cae
// CHECK:STDOUT:   %U.patt.loc21_31.2 => constants.%U.patt.0f3348.2
// CHECK:STDOUT:   %V.patt.loc21_73.2 => constants.%V.patt
// CHECK:STDOUT:   %V.loc21_73.1 => bool
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc21_31 => constants.%complete_type.229
// CHECK:STDOUT: }
// CHECK:STDOUT:
