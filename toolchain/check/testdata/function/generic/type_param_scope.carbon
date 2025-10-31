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
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.d9f [symbolic]
// CHECK:STDOUT:   %pattern_type.a60: type = pattern_type %ptr [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.c66: <witness> = lookup_impl_witness %ptr, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr, (%Copy.lookup_impl_witness.c66) [symbolic]
// CHECK:STDOUT:   %.89d: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.928: %.89d = impl_witness_access %Copy.lookup_impl_witness.c66, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6a1: <specific function> = specific_impl_function %impl.elem0.928, @Copy.Op(%Copy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_6.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc4_20.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.c66)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc4_20.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc7_10.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc7_10.2 (constants.%.89d)]
// CHECK:STDOUT:   %impl.elem0.loc7_10.2: @F.%.loc7_10.2 (%.89d) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_10.2 (constants.%impl.elem0.928)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_10.2: <specific function> = specific_impl_function %impl.elem0.loc7_10.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc7_10.2 (constants.%specific_impl_fn.6a1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @F.%ptr.loc4_20.1 (%ptr)) -> @F.%ptr.loc4_20.1 (%ptr) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %m.patt: @F.%pattern_type (%pattern_type.a60) = value_binding_pattern m [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n.ref: @F.%ptr.loc4_20.1 (%ptr) = name_ref n, %n
// CHECK:STDOUT:     %.loc6: type = splice_block %ptr.loc6 [symbolic = %ptr.loc4_20.1 (constants.%ptr)] {
// CHECK:STDOUT:       %T.ref.loc6: type = name_ref T, %T.loc4_6.2 [symbolic = %T.loc4_6.1 (constants.%T.d9f)]
// CHECK:STDOUT:       %ptr.loc6: type = ptr_type %T.ref.loc6 [symbolic = %ptr.loc4_20.1 (constants.%ptr)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %m: @F.%ptr.loc4_20.1 (%ptr) = value_binding m, %n.ref
// CHECK:STDOUT:     %m.ref: @F.%ptr.loc4_20.1 (%ptr) = name_ref m, %m
// CHECK:STDOUT:     %impl.elem0.loc7_10.1: @F.%.loc7_10.2 (%.89d) = impl_witness_access constants.%Copy.lookup_impl_witness.c66, element0 [symbolic = %impl.elem0.loc7_10.2 (constants.%impl.elem0.928)]
// CHECK:STDOUT:     %bound_method.loc7_10.1: <bound method> = bound_method %m.ref, %impl.elem0.loc7_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_10.1: <specific function> = specific_impl_function %impl.elem0.loc7_10.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc7_10.2 (constants.%specific_impl_fn.6a1)]
// CHECK:STDOUT:     %bound_method.loc7_10.2: <bound method> = bound_method %m.ref, %specific_impl_fn.loc7_10.1
// CHECK:STDOUT:     %.loc7_10.1: init @F.%ptr.loc4_20.1 (%ptr) = call %bound_method.loc7_10.2(%m.ref)
// CHECK:STDOUT:     return %.loc7_10.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.d9f) {
// CHECK:STDOUT:   %T.loc4_6.1 => constants.%T.d9f
// CHECK:STDOUT:   %ptr.loc4_20.1 => constants.%ptr
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a60
// CHECK:STDOUT: }
// CHECK:STDOUT:
