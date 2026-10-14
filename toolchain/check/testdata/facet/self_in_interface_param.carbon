// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/self_in_interface_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/self_in_interface_param.carbon

interface I(T: type) {
  let I1: type;
}

//@dump-sem-ir-begin
fn F(generic T: I(.Self) where .I1 = ()) -> T.I1 {
  return ();
}
//@dump-sem-ir-end

fn G(generic _: I(.Self) where .I1 = ()) {}

// CHECK:STDOUT: --- self_in_interface_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen.e58: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.d52: type = facet_type <@I, @I(%.Self.frozen.e58)> [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.075: %I.type.d52 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %I.assoc_type.d56: type = assoc_entity_type @I, @I(%.Self.frozen.e58) [symbolic_self]
// CHECK:STDOUT:   %assoc0.aed: %I.assoc_type.d56 = assoc_entity element0, @I.WithSelf.%I1 [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.075 [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.b6a: <witness> = lookup_impl_witness %.Self.frozen.075, @I, @I(%.Self.frozen.e58) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.6ec: type = impl_witness_access %I.lookup_impl_witness.b6a, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self.c86: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %I.type.e26: type = facet_type <@I, @I(%.Self.c86)> [symbolic_self]
// CHECK:STDOUT:   %.Self.dc6: %I.type.d52 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.732: <witness> = lookup_impl_witness %.Self.dc6, @I, @I(%.Self.c86) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c33: type = impl_witness_access %I.lookup_impl_witness.732, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type.691: type = facet_type <@I, @I(%.Self.c86) where %impl.elem0.c33 = %empty_tuple.type> [symbolic_self]
// CHECK:STDOUT:   %.Self.105: %I.type.e26 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.59f: <witness> = lookup_impl_witness %.Self.105, @I, @I(%.Self.c86) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.7f7: type = impl_witness_access %I.lookup_impl_witness.59f, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type.4b3: type = facet_type <@I, @I(%.Self.c86) where %impl.elem0.7f7 = %empty_tuple.type> [symbolic_self]
// CHECK:STDOUT:   %pattern_type.ad5: type = pattern_type %I_where.type.4b3 [symbolic_self]
// CHECK:STDOUT:   %T.patt.818: %pattern_type.ad5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.019: %I_where.type.4b3 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.46d: type = assoc_entity_type @I, @I(%.Self.c86) [symbolic_self]
// CHECK:STDOUT:   %assoc0.1a5: %I.assoc_type.46d = assoc_entity element0, @I.WithSelf.%I1 [symbolic_self]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.019 [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness.1f0: <witness> = lookup_impl_witness %T.019, @I, @I(%T.as_type) [symbolic]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc18_15.1: %pattern_type.ad5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_15.2 (constants.%T.patt.818)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %impl.elem0.loc18_46 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I_where.type.4b3 = name_ref T, %T.loc18_15.2 [symbolic = %T.loc18_15.1 (constants.%T.019)]
// CHECK:STDOUT:     %T.as_type.loc18_46.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc18_46.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc18_46.1: type = converted %T.ref, %T.as_type.loc18_46.2 [symbolic = %T.as_type.loc18_46.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc18_46.2: %I.assoc_type.46d = specific_constant @I1.%assoc0, @I.WithSelf(constants.%.Self.c86, constants.%T.019) [symbolic_self = constants.%assoc0.1a5]
// CHECK:STDOUT:     %I1.ref.loc18_46: %I.assoc_type.46d = name_ref I1, %.loc18_46.2 [symbolic_self = constants.%assoc0.1a5]
// CHECK:STDOUT:     %impl.elem0.loc18_46: type = impl_witness_access constants.%I.lookup_impl_witness.1f0, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc18_46.3: Core.Form = init_form %impl.elem0.loc18_46 [concrete = constants.%.262]
// CHECK:STDOUT:     %.loc18_26.1: type = splice_block %.loc18_26.3 [symbolic_self = constants.%I_where.type.4b3] {
// CHECK:STDOUT:       %.Self.frozen.loc18_15: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %I.ref: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.Self.ref.loc18_19: type = name_ref .Self, %.Self.frozen.loc18_15 [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %I.type.loc18_24.1: type = facet_type <@I, @I(constants.%.Self.frozen.e58)> [symbolic_self = constants.%I.type.d52]
// CHECK:STDOUT:       %.Self.frozen.loc18_26: %I.type.d52 = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.075]
// CHECK:STDOUT:       %base_facet_type.loc18_26.1 = requirement_base_facet_type %I.type.loc18_24.1 [concrete]
// CHECK:STDOUT:       %.Self.ref.loc18_32: %I.type.d52 = name_ref .Self, %.Self.frozen.loc18_26 [symbolic_self = constants.%.Self.frozen.075]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref.loc18_32 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc18_32.1: type = converted %.Self.ref.loc18_32, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc18_32.2: %I.assoc_type.d56 = specific_constant @I1.%assoc0, @I.WithSelf(constants.%.Self.frozen.e58, constants.%.Self.frozen.075) [symbolic_self = constants.%assoc0.aed]
// CHECK:STDOUT:       %I1.ref.loc18_32: %I.assoc_type.d56 = name_ref I1, %.loc18_32.2 [symbolic_self = constants.%assoc0.aed]
// CHECK:STDOUT:       %impl.elem0.loc18_32.1: type = impl_witness_access constants.%I.lookup_impl_witness.b6a, element0 [symbolic_self = constants.%impl.elem0.6ec]
// CHECK:STDOUT:       %.loc18_39.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc18_39.2: type = converted %.loc18_39.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %rewrite.loc18_36.1 = requirement_rewrite %impl.elem0.loc18_32.1, %.loc18_39.2 [concrete]
// CHECK:STDOUT:       %I.type.loc18_24.2: type = facet_type <@I, @I(constants.%.Self.c86)> [symbolic_self = constants.%I.type.e26]
// CHECK:STDOUT:       %base_facet_type.loc18_26.2 = requirement_base_facet_type %I.type.loc18_24.2 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc18_32.2: type = impl_witness_access constants.%I.lookup_impl_witness.732, element0 [symbolic_self = constants.%impl.elem0.c33]
// CHECK:STDOUT:       %rewrite.loc18_36.2 = requirement_rewrite %impl.elem0.loc18_32.2, %.loc18_39.2 [concrete]
// CHECK:STDOUT:       %.loc18_26.2: type = where_expr [symbolic_self = constants.%I_where.type.691] {
// CHECK:STDOUT:         %base_facet_type.loc18_26.2 = requirement_base_facet_type %I.type.loc18_24.2 [concrete]
// CHECK:STDOUT:         %rewrite.loc18_36.2 = requirement_rewrite %impl.elem0.loc18_32.2, %.loc18_39.2 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %impl.elem0.loc18_32.3: type = impl_witness_access constants.%I.lookup_impl_witness.59f, element0 [symbolic_self = constants.%impl.elem0.7f7]
// CHECK:STDOUT:       %rewrite.loc18_36.3 = requirement_rewrite %impl.elem0.loc18_32.3, %.loc18_39.2 [concrete]
// CHECK:STDOUT:       %.loc18_26.3: type = where_expr [symbolic_self = constants.%I_where.type.4b3] {
// CHECK:STDOUT:         %base_facet_type.loc18_26.2 = requirement_base_facet_type %I.type.loc18_24.2 [concrete]
// CHECK:STDOUT:         %rewrite.loc18_36.3 = requirement_rewrite %impl.elem0.loc18_32.3, %.loc18_39.2 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_15.2: %I_where.type.4b3 = symbolic_binding T, 0 [symbolic = %T.loc18_15.1 (constants.%T.019)]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc18_15.2: %I_where.type.4b3) {
// CHECK:STDOUT:   %T.patt.loc18_15.2: %pattern_type.ad5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_15.2 (constants.%T.patt.818)]
// CHECK:STDOUT:   %T.loc18_15.1: %I_where.type.4b3 = symbolic_binding T, 0 [symbolic = %T.loc18_15.1 (constants.%T.019)]
// CHECK:STDOUT:   %T.as_type.loc18_46.1: type = facet_access_type %T.loc18_15.1 [symbolic = %T.as_type.loc18_46.1 (constants.%T.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc19_11.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc19_11.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc19_12: init %empty_tuple.type = converted %.loc19_11.1, %.loc19_11.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc19_12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.019) {
// CHECK:STDOUT:   %T.patt.loc18_15.2 => constants.%T.patt.818
// CHECK:STDOUT:   %T.loc18_15.1 => constants.%T.019
// CHECK:STDOUT:   %T.as_type.loc18_46.1 => constants.%T.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
