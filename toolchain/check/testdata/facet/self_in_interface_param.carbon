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
// CHECK:STDOUT:   %.Self.644: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.962: type = facet_type <@I, @I(%.Self.644)> [symbolic_self]
// CHECK:STDOUT:   %.Self.50d: %I.type.962 = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %I.assoc_type.180: type = assoc_entity_type @I, @I(%.Self.644) [symbolic_self]
// CHECK:STDOUT:   %assoc0.357: %I.assoc_type.180 = assoc_entity element0, @I.%I1 [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.50d [symbolic_self]
// CHECK:STDOUT:   %I.lookup_impl_witness.aee: <witness> = lookup_impl_witness %.Self.50d, @I, @I(%.Self.644) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %I.lookup_impl_witness.aee, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I, @I(%.Self.644) where %impl.elem0 = %empty_tuple.type> [symbolic_self]
// CHECK:STDOUT:   %T.735: %I_where.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3cf: type = pattern_type %I_where.type [symbolic_self]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.735 [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness.262: <witness> = lookup_impl_witness %T.735, @I, @I(%.Self.644) [symbolic]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3cf = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: %I_where.type = name_ref T, %T.loc18_6.2 [symbolic = %T.loc18_6.1 (constants.%T.735)]
// CHECK:STDOUT:     %.loc18_39.1: %I.assoc_type.180 = specific_constant @I1.%assoc0, @I(constants.%.Self.644) [symbolic_self = constants.%assoc0.357]
// CHECK:STDOUT:     %I1.ref.loc18_39: %I.assoc_type.180 = name_ref I1, %.loc18_39.1 [symbolic_self = constants.%assoc0.357]
// CHECK:STDOUT:     %T.as_type.loc18_39.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc18_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc18_39.2: type = converted %T.ref, %T.as_type.loc18_39.2 [symbolic = %T.as_type.loc18_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc18_39: type = impl_witness_access constants.%I.lookup_impl_witness.262, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc18_19.1: type = splice_block %.loc18_19.2 [symbolic_self = constants.%I_where.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %.Self.ref.loc18_12: type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self.644]
// CHECK:STDOUT:       %I.type: type = facet_type <@I, @I(constants.%.Self.644)> [symbolic_self = constants.%I.type.962]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref.loc18_25: %I.type.962 = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self.50d]
// CHECK:STDOUT:       %.loc18_25.1: %I.assoc_type.180 = specific_constant @I1.%assoc0, @I(constants.%.Self.644) [symbolic_self = constants.%assoc0.357]
// CHECK:STDOUT:       %I1.ref.loc18_25: %I.assoc_type.180 = name_ref I1, %.loc18_25.1 [symbolic_self = constants.%assoc0.357]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref.loc18_25 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc18_25.2: type = converted %.Self.ref.loc18_25, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %impl.elem0.loc18_25: type = impl_witness_access constants.%I.lookup_impl_witness.aee, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:       %.loc18_32.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc18_32.2: type = converted %.loc18_32.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc18_19.2: type = where_expr %.Self.2 [symbolic_self = constants.%I_where.type] {
// CHECK:STDOUT:         requirement_base_facet_type constants.%I.type.962
// CHECK:STDOUT:         requirement_rewrite %impl.elem0.loc18_25, %.loc18_32.2
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_6.2: %I_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc18_6.1 (constants.%T.735)]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc18_6.2: %I_where.type) {
// CHECK:STDOUT:   %T.loc18_6.1: %I_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc18_6.1 (constants.%T.735)]
// CHECK:STDOUT:   %T.as_type.loc18_39.1: type = facet_access_type %T.loc18_6.1 [symbolic = %T.as_type.loc18_39.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc18_6.1, @I, @I(constants.%.Self.644) [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness.262)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %empty_tuple.type {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc19_11: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc19_12: %empty_tuple.type = converted %.loc19_11, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     return %.loc19_12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.735) {
// CHECK:STDOUT:   %T.loc18_6.1 => constants.%T.735
// CHECK:STDOUT:   %T.as_type.loc18_39.1 => constants.%T.as_type
// CHECK:STDOUT:   %I.lookup_impl_witness => constants.%I.lookup_impl_witness.262
// CHECK:STDOUT: }
// CHECK:STDOUT:
