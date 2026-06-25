// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/type_param_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/type_param_scope.carbon

// --- local_let.carbon

library "[[@TEST_NAME]]";

fn F(T:! type, n: T*) -> T* {
  //@dump-sem-ir-begin
  let m: T* = n;
  return m;
  //@dump-sem-ir-end
}


// CHECK:STDOUT: --- local_let.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr [symbolic]
// CHECK:STDOUT:   %n.param_patt: %pattern_type.4f4 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %n.patt: %pattern_type.4f4 = at_binding_pattern n, %n.param_patt [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr [symbolic]
// CHECK:STDOUT:   %return.param_patt.27f: %pattern_type.4f4 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.d67: %pattern_type.4f4 = return_slot_pattern %return.param_patt.27f, %ptr [symbolic]
// CHECK:STDOUT:   %m.patt: %pattern_type.4f4 = value_binding_pattern m [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.1da: <witness> = lookup_impl_witness %ptr, @Copy [symbolic]
// CHECK:STDOUT:   %.0e9: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness.1da) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.884: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.be5: type = fn_type_with_self_type %Copy.WithSelf.Op.type.884, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.484: %.be5 = impl_witness_access %Copy.lookup_impl_witness.1da, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a76: <specific function> = specific_impl_function %impl.elem0.484, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_7.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %m.patt.loc6_8.2: @F.%pattern_type (%pattern_type.4f4) = value_binding_pattern m [symbolic = %m.patt.loc6_8.2 (constants.%m.patt)]
// CHECK:STDOUT:   %.loc7_10.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc4_7.1) [symbolic = %.loc7_10.3 (constants.%.0e9)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc4_20.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.1da)]
// CHECK:STDOUT:   %Copy.facet.loc7_10.3: %Copy.type = facet_value %ptr.loc4_20.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc7_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc7_10.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.884)]
// CHECK:STDOUT:   %.loc7_10.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc7_10.3 [symbolic = %.loc7_10.4 (constants.%.be5)]
// CHECK:STDOUT:   %impl.elem0.loc7_10.2: @F.%.loc7_10.4 (%.be5) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_10.2 (constants.%impl.elem0.484)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_10.2: <specific function> = specific_impl_function %impl.elem0.loc7_10.2, @Copy.WithSelf.Op(%Copy.facet.loc7_10.3) [symbolic = %specific_impl_fn.loc7_10.2 (constants.%specific_impl_fn.a76)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @F.%ptr.loc4_20.1 (%ptr)) -> out %return.param: @F.%ptr.loc4_20.1 (%ptr) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %m.patt.loc6_8.1: @F.%pattern_type (%pattern_type.4f4) = value_binding_pattern m [symbolic = %m.patt.loc6_8.2 (constants.%m.patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n.ref: @F.%ptr.loc4_20.1 (%ptr) = name_ref n, %n
// CHECK:STDOUT:     %.loc6: type = splice_block %ptr.loc6 [symbolic = %ptr.loc4_20.1 (constants.%ptr)] {
// CHECK:STDOUT:       %T.ref.loc6: type = name_ref T, %T.loc4_7.2 [symbolic = %T.loc4_7.1 (constants.%T.67d)]
// CHECK:STDOUT:       %ptr.loc6: type = ptr_type %T.ref.loc6 [symbolic = %ptr.loc4_20.1 (constants.%ptr)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %m: @F.%ptr.loc4_20.1 (%ptr) = value_binding m, %n.ref
// CHECK:STDOUT:     %m.ref: @F.%ptr.loc4_20.1 (%ptr) = name_ref m, %m
// CHECK:STDOUT:     %impl.elem0.loc7_10.1: @F.%.loc7_10.4 (%.be5) = impl_witness_access constants.%Copy.lookup_impl_witness.1da, element0 [symbolic = %impl.elem0.loc7_10.2 (constants.%impl.elem0.484)]
// CHECK:STDOUT:     %bound_method.loc7_10.1: <bound method> = bound_method %m.ref, %impl.elem0.loc7_10.1
// CHECK:STDOUT:     %Copy.facet.loc7_10.1: %Copy.type = facet_value constants.%ptr, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc7_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc7_10.1: %Copy.type = converted constants.%ptr, %Copy.facet.loc7_10.1 [symbolic = %Copy.facet.loc7_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %Copy.facet.loc7_10.2: %Copy.type = facet_value constants.%ptr, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc7_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %.loc7_10.2: %Copy.type = converted constants.%ptr, %Copy.facet.loc7_10.2 [symbolic = %Copy.facet.loc7_10.3 (constants.%Copy.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_10.1: <specific function> = specific_impl_function %impl.elem0.loc7_10.1, @Copy.WithSelf.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc7_10.2 (constants.%specific_impl_fn.a76)]
// CHECK:STDOUT:     %bound_method.loc7_10.2: <bound method> = bound_method %m.ref, %specific_impl_fn.loc7_10.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @F.%ptr.loc4_20.1 (%ptr) = call %bound_method.loc7_10.2(%m.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc4_7.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc4_7.1 => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc4_20.1 => constants.%ptr
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %n.param_patt.loc4_17.2 => constants.%n.param_patt
// CHECK:STDOUT:   %n.patt.loc4_17.2 => constants.%n.patt
// CHECK:STDOUT:   %.loc4_27.1 => constants.%.cb6
// CHECK:STDOUT:   %return.param_patt.loc4_27.2 => constants.%return.param_patt.27f
// CHECK:STDOUT:   %return.patt.loc4_23.2 => constants.%return.patt.d67
// CHECK:STDOUT: }
// CHECK:STDOUT:
