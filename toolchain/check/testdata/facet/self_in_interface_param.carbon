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

interface I(T:! type) {
  let I1:! type;
}

//@dump-sem-ir-begin
fn F(T:! I(.Self) where .I1 = ()) -> T.I1 {
  return ();
}
//@dump-sem-ir-end

fn G(_:! I(.Self) where .I1 = ()) {}

// CHECK:STDOUT: --- self_in_interface_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.c39: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %I.type.a64: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.a64 = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.as_type.246: type = facet_access_type %.Self.c39 [symbolic_self]
// CHECK:STDOUT:   %I.type.64e: type = facet_type <@I, @I(%.Self.as_type.246)> [symbolic_self]
// CHECK:STDOUT:   %.Self.00b: %I.type.64e = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %I.assoc_type.e08: type = assoc_entity_type @I, @I(%.Self.as_type.246) [symbolic_self]
// CHECK:STDOUT:   %assoc0.636: %I.assoc_type.e08 = assoc_entity element0, @I.WithSelf.%I1 [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.f98: type = facet_access_type %.Self.00b [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.e39: <witness> = lookup_impl_witness %.Self.00b, @I, @I(%.Self.as_type.246) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %I.lookup_impl_witness.e39, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I, @I(%.Self.as_type.246) where %impl.elem0 = %empty_tuple.type> [symbolic_self]
// CHECK:STDOUT:   %pattern_type.c98: type = pattern_type %I_where.type [symbolic_self]
// CHECK:STDOUT:   %T.3a9: %I_where.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.3a9 [symbolic]
// CHECK:STDOUT:   %facet_value: %type = facet_value %T.as_type, () [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness.9e3: <witness> = lookup_impl_witness %T.3a9, @I, @I(%T.as_type) [symbolic]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.c98 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %impl.elem0.loc18_39 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I_where.type = name_ref T, %T.loc18_7.2 [symbolic = %T.loc18_7.1 (constants.%T.3a9)]
// CHECK:STDOUT:     %T.as_type.loc18_39.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc18_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc18_39.1: type = converted %T.ref, %T.as_type.loc18_39.2 [symbolic = %T.as_type.loc18_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc18_39.2: %I.assoc_type.e08 = specific_constant @I1.%assoc0, @I.WithSelf(constants.%.Self.as_type.246, constants.%T.3a9) [symbolic_self = constants.%assoc0.636]
// CHECK:STDOUT:     %I1.ref.loc18_39: %I.assoc_type.e08 = name_ref I1, %.loc18_39.2 [symbolic_self = constants.%assoc0.636]
// CHECK:STDOUT:     %facet_value.loc18_39.2: %type = facet_value constants.%T.as_type, () [symbolic = %facet_value.loc18_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc18_39.3: %type = converted constants.%T.as_type, %facet_value.loc18_39.2 [symbolic = %facet_value.loc18_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %T.as_type.loc18_39.3: type = facet_access_type constants.%T.3a9 [symbolic = %T.as_type.loc18_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %facet_value.loc18_39.3: %type = facet_value %T.as_type.loc18_39.3, () [symbolic = %facet_value.loc18_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc18_39.4: %type = converted constants.%T.3a9, %facet_value.loc18_39.3 [symbolic = %facet_value.loc18_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %T.as_type.loc18_39.4: type = facet_access_type constants.%T.3a9 [symbolic = %T.as_type.loc18_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %facet_value.loc18_39.4: %type = facet_value %T.as_type.loc18_39.4, () [symbolic = %facet_value.loc18_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc18_39.5: %type = converted constants.%T.3a9, %facet_value.loc18_39.4 [symbolic = %facet_value.loc18_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %impl.elem0.loc18_39: type = impl_witness_access constants.%I.lookup_impl_witness.9e3, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc18_39.6: Core.Form = init_form %impl.elem0.loc18_39 [concrete = constants.%.262]
// CHECK:STDOUT:     %.loc18_19.1: type = splice_block %.loc18_19.2 [symbolic_self = constants.%I_where.type] {
// CHECK:STDOUT:       %.Self.loc18_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %I.ref: %I.type.a64 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.Self.ref.loc18_12: %type = name_ref .Self, %.Self.loc18_7 [symbolic_self = constants.%.Self.c39]
// CHECK:STDOUT:       %.Self.as_type.loc18_17: type = facet_access_type %.Self.ref.loc18_12 [symbolic_self = constants.%.Self.as_type.246]
// CHECK:STDOUT:       %.loc18_17: type = converted %.Self.ref.loc18_12, %.Self.as_type.loc18_17 [symbolic_self = constants.%.Self.as_type.246]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%.Self.as_type.246)> [symbolic_self = constants.%I.type.64e]
// CHECK:STDOUT:       %.Self.loc18_19: %I.type.64e = symbolic_binding .Self [symbolic_self = constants.%.Self.00b]
// CHECK:STDOUT:       %.Self.ref.loc18_25: %I.type.64e = name_ref .Self, %.Self.loc18_19 [symbolic_self = constants.%.Self.00b]
// CHECK:STDOUT:       %.Self.as_type.loc18_25: type = facet_access_type %.Self.ref.loc18_25 [symbolic_self = constants.%.Self.as_type.f98]
// CHECK:STDOUT:       %.loc18_25.1: type = converted %.Self.ref.loc18_25, %.Self.as_type.loc18_25 [symbolic_self = constants.%.Self.as_type.f98]
// CHECK:STDOUT:       %.loc18_25.2: %I.assoc_type.e08 = specific_constant @I1.%assoc0, @I.WithSelf(constants.%.Self.as_type.246, constants.%.Self.00b) [symbolic_self = constants.%assoc0.636]
// CHECK:STDOUT:       %I1.ref.loc18_25: %I.assoc_type.e08 = name_ref I1, %.loc18_25.2 [symbolic_self = constants.%assoc0.636]
// CHECK:STDOUT:       %impl.elem0.loc18_25: type = impl_witness_access constants.%I.lookup_impl_witness.e39, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc18_32.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc18_32.2: type = converted %.loc18_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc18_19.2: type = where_expr [symbolic_self = constants.%I_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type %I.type
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc18_25, %.loc18_32.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_7.2: %I_where.type = symbolic_binding T, 0 [symbolic = %T.loc18_7.1 (constants.%T.3a9)]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc18_7.2: %I_where.type) {
// CHECK:STDOUT:   %T.loc18_7.1: %I_where.type = symbolic_binding T, 0 [symbolic = %T.loc18_7.1 (constants.%T.3a9)]
// CHECK:STDOUT:   %T.as_type.loc18_39.1: type = facet_access_type %T.loc18_7.1 [symbolic = %T.as_type.loc18_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %facet_value.loc18_39.1: %type = facet_value %T.as_type.loc18_39.1, () [symbolic = %facet_value.loc18_39.1 (constants.%facet_value)]
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
// CHECK:STDOUT: specific @F(constants.%T.3a9) {
// CHECK:STDOUT:   %T.loc18_7.1 => constants.%T.3a9
// CHECK:STDOUT:   %T.as_type.loc18_39.1 => constants.%T.as_type
// CHECK:STDOUT:   %facet_value.loc18_39.1 => constants.%facet_value
// CHECK:STDOUT: }
// CHECK:STDOUT:
