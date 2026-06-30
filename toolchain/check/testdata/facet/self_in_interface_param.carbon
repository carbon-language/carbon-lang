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
// CHECK:STDOUT:   %.Self.197: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.as_type.bce: type = facet_access_type %.Self.197 [symbolic_self]
// CHECK:STDOUT:   %I.type.0e8: type = facet_type <@I, @I(%.Self.as_type.bce)> [symbolic_self]
// CHECK:STDOUT:   %.Self.edc: %I.type.0e8 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %I.assoc_type.96e: type = assoc_entity_type @I, @I(%.Self.as_type.bce) [symbolic_self]
// CHECK:STDOUT:   %assoc0.cab: %I.assoc_type.96e = assoc_entity element0, @I.WithSelf.%I1 [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.7bb: type = facet_access_type %.Self.edc [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.d6e: <witness> = lookup_impl_witness %.Self.edc, @I, @I(%.Self.as_type.bce) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.14f: type = impl_witness_access %I.lookup_impl_witness.d6e, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self.b30: %I.type.0e8 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.e0b: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type.007: type = facet_access_type %.Self.e0b [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.eea: <witness> = lookup_impl_witness %.Self.b30, @I, @I(%.Self.as_type.007) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.439: type = impl_witness_access %I.lookup_impl_witness.eea, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I.type.a4e: type = facet_type <@I, @I(%.Self.as_type.007)> [symbolic_self]
// CHECK:STDOUT:   %I_where.type.760: type = facet_type <@I, @I(%.Self.as_type.007) where %impl.elem0.439 = %empty_tuple.type> [symbolic_self]
// CHECK:STDOUT:   %.Self.df8: %I.type.a4e = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.0d5: <witness> = lookup_impl_witness %.Self.df8, @I, @I(%.Self.as_type.007) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.21c: type = impl_witness_access %I.lookup_impl_witness.0d5, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type.6d9: type = facet_type <@I, @I(%.Self.as_type.007) where %impl.elem0.21c = %empty_tuple.type> [symbolic_self]
// CHECK:STDOUT:   %pattern_type.4af: type = pattern_type %I_where.type.6d9 [symbolic_self]
// CHECK:STDOUT:   %T.patt.bfa: %pattern_type.4af = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.fb1: %I_where.type.6d9 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type.901: type = assoc_entity_type @I, @I(%.Self.as_type.007) [symbolic_self]
// CHECK:STDOUT:   %assoc0.267: %I.assoc_type.901 = assoc_entity element0, @I.WithSelf.%I1 [symbolic_self]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.fb1 [symbolic]
// CHECK:STDOUT:   %facet_value: %type = facet_value %T.as_type, () [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness.962: <witness> = lookup_impl_witness %T.fb1, @I, @I(%T.as_type) [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc18_7.1: %pattern_type.4af = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_7.2 (constants.%T.patt.bfa)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %impl.elem0.loc18_39 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I_where.type.6d9 = name_ref T, %T.loc18_7.2 [symbolic = %T.loc18_7.1 (constants.%T.fb1)]
// CHECK:STDOUT:     %T.as_type.loc18_39.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc18_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc18_39.1: type = converted %T.ref, %T.as_type.loc18_39.2 [symbolic = %T.as_type.loc18_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc18_39.2: %I.assoc_type.901 = specific_constant @I1.%assoc0, @I.WithSelf(constants.%.Self.as_type.007, constants.%T.fb1) [symbolic_self = constants.%assoc0.267]
// CHECK:STDOUT:     %I1.ref.loc18_39: %I.assoc_type.901 = name_ref I1, %.loc18_39.2 [symbolic_self = constants.%assoc0.267]
// CHECK:STDOUT:     %facet_value.loc18_39.2: %type = facet_value constants.%T.as_type, () [symbolic = %facet_value.loc18_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc18_39.3: %type = converted constants.%T.as_type, %facet_value.loc18_39.2 [symbolic = %facet_value.loc18_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %T.as_type.loc18_39.3: type = facet_access_type constants.%T.fb1 [symbolic = %T.as_type.loc18_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %facet_value.loc18_39.3: %type = facet_value %T.as_type.loc18_39.3, () [symbolic = %facet_value.loc18_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc18_39.4: %type = converted constants.%T.fb1, %facet_value.loc18_39.3 [symbolic = %facet_value.loc18_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %T.as_type.loc18_39.4: type = facet_access_type constants.%T.fb1 [symbolic = %T.as_type.loc18_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %facet_value.loc18_39.4: %type = facet_value %T.as_type.loc18_39.4, () [symbolic = %facet_value.loc18_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc18_39.5: %type = converted constants.%T.fb1, %facet_value.loc18_39.4 [symbolic = %facet_value.loc18_39.1 (constants.%facet_value)]
// CHECK:STDOUT:     %impl.elem0.loc18_39: type = impl_witness_access constants.%I.lookup_impl_witness.962, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc18_39.6: Core.Form = init_form %impl.elem0.loc18_39 [concrete = constants.%.262]
// CHECK:STDOUT:     %.loc18_19.1: type = splice_block <unexpected>.inst{{[0-9A-F]+}}.loc18_19 [symbolic_self = constants.%I_where.type.6d9] {
// CHECK:STDOUT:       %.Self.loc18_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.197]
// CHECK:STDOUT:       %I.ref: %I.type.335 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.Self.ref.loc18_12: %type = name_ref .Self, %.Self.loc18_7 [symbolic_self = constants.%.Self.197]
// CHECK:STDOUT:       %.Self.as_type.loc18_17: type = facet_access_type %.Self.ref.loc18_12 [symbolic_self = constants.%.Self.as_type.bce]
// CHECK:STDOUT:       %.loc18_17: type = converted %.Self.ref.loc18_12, %.Self.as_type.loc18_17 [symbolic_self = constants.%.Self.as_type.bce]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%.Self.as_type.bce)> [symbolic_self = constants.%I.type.0e8]
// CHECK:STDOUT:       %.Self.loc18_19: %I.type.0e8 = symbolic_binding .Self [symbolic_self = constants.%.Self.edc]
// CHECK:STDOUT:       %.Self.ref.loc18_25: %I.type.0e8 = name_ref .Self, %.Self.loc18_19 [symbolic_self = constants.%.Self.edc]
// CHECK:STDOUT:       %.Self.as_type.loc18_25: type = facet_access_type %.Self.ref.loc18_25 [symbolic_self = constants.%.Self.as_type.7bb]
// CHECK:STDOUT:       %.loc18_25.1: type = converted %.Self.ref.loc18_25, %.Self.as_type.loc18_25 [symbolic_self = constants.%.Self.as_type.7bb]
// CHECK:STDOUT:       %.loc18_25.2: %I.assoc_type.96e = specific_constant @I1.%assoc0, @I.WithSelf(constants.%.Self.as_type.bce, constants.%.Self.edc) [symbolic_self = constants.%assoc0.cab]
// CHECK:STDOUT:       %I1.ref.loc18_25: %I.assoc_type.96e = name_ref I1, %.loc18_25.2 [symbolic_self = constants.%assoc0.cab]
// CHECK:STDOUT:       %impl.elem0.loc18_25: type = impl_witness_access constants.%I.lookup_impl_witness.d6e, element0 [symbolic_self = constants.%impl.elem0.14f]
// CHECK:STDOUT:       %.loc18_32.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc18_32.2: type = converted %.loc18_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc18_19.2: type = where_expr [symbolic_self = constants.%I_where.type.760] {
// CHECK:STDOUT:         requirement_base_facet_type <unexpected>.inst{{[0-9A-F]+}}.loc18_17
// CHECK:STDOUT:         requirement_rewrite <unexpected>.inst{{[0-9A-F]+}}.loc18_25, %.loc18_32.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_7.2: %I_where.type.6d9 = symbolic_binding T, 0 [symbolic = %T.loc18_7.1 (constants.%T.fb1)]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc18_7.2: %I_where.type.6d9) {
// CHECK:STDOUT:   %T.patt.loc18_7.2: %pattern_type.4af = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc18_7.2 (constants.%T.patt.bfa)]
// CHECK:STDOUT:   %T.loc18_7.1: %I_where.type.6d9 = symbolic_binding T, 0 [symbolic = %T.loc18_7.1 (constants.%T.fb1)]
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
// CHECK:STDOUT: specific @F(constants.%T.fb1) {
// CHECK:STDOUT:   %T.patt.loc18_7.2 => constants.%T.patt.bfa
// CHECK:STDOUT:   %T.loc18_7.1 => constants.%T.fb1
// CHECK:STDOUT:   %T.as_type.loc18_39.1 => constants.%T.as_type
// CHECK:STDOUT:   %facet_value.loc18_39.1 => constants.%facet_value
// CHECK:STDOUT: }
// CHECK:STDOUT:
