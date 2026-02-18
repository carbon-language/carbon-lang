// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/indirect_generic_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/indirect_generic_type.carbon

// --- pointer_to_pointer.carbon

library "[[@TEST_NAME]]";

fn F(T:! type, p: T**) -> T* {
  //@dump-sem-ir-begin
  return *p;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- pointer_to_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %ptr.125: type = ptr_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %pattern_type.8bb: type = pattern_type %ptr.125 [symbolic]
// CHECK:STDOUT:   %.4fe: Core.Form = init_form %ptr.e8f, call_param1 [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %.2f2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2e6: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.2e6) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d82: type = fn_type @Copy.WithSelf.Op, @Copy(%Copy.facet) [symbolic]
// CHECK:STDOUT:   %.299: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d82, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.1c7: %.299 = impl_witness_access %Copy.lookup_impl_witness.2e6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.366: <specific function> = specific_impl_function %impl.elem0.1c7, @Copy.WithSelf.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:   %.loc6_10.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc4_6.1) [symbolic = %.loc6_10.3 (constants.%.2f2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc4_20.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2e6)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc4_20.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d82)]
// CHECK:STDOUT:   %.loc6_10.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc6_10.4 (constants.%.299)]
// CHECK:STDOUT:   %impl.elem0.loc6_10.2: @F.%.loc6_10.4 (%.299) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_10.2 (constants.%impl.elem0.1c7)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2: <specific function> = specific_impl_function %impl.elem0.loc6_10.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc6_10.2 (constants.%specific_impl_fn.366)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%ptr.loc4_21.1 (%ptr.125)) -> out %return.param: @F.%ptr.loc4_20.1 (%ptr.e8f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc4_21.1 (%ptr.125) = name_ref p, %p
// CHECK:STDOUT:     %.loc6_10.1: ref @F.%ptr.loc4_20.1 (%ptr.e8f) = deref %p.ref
// CHECK:STDOUT:     %.loc6_10.2: @F.%ptr.loc4_20.1 (%ptr.e8f) = acquire_value %.loc6_10.1
// CHECK:STDOUT:     %impl.elem0.loc6_10.1: @F.%.loc6_10.4 (%.299) = impl_witness_access constants.%Copy.lookup_impl_witness.2e6, element0 [symbolic = %impl.elem0.loc6_10.2 (constants.%impl.elem0.1c7)]
// CHECK:STDOUT:     %bound_method.loc6_10.1: <bound method> = bound_method %.loc6_10.2, %impl.elem0.loc6_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_10.1: <specific function> = specific_impl_function %impl.elem0.loc6_10.1, @Copy.WithSelf.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc6_10.2 (constants.%specific_impl_fn.366)]
// CHECK:STDOUT:     %bound_method.loc6_10.2: <bound method> = bound_method %.loc6_10.2, %specific_impl_fn.loc6_10.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @F.%ptr.loc4_20.1 (%ptr.e8f) = call %bound_method.loc6_10.2(%.loc6_10.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc4_6.1 => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc4_20.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %ptr.loc4_21.1 => constants.%ptr.125
// CHECK:STDOUT:   %pattern_type.loc4_16 => constants.%pattern_type.8bb
// CHECK:STDOUT:   %.loc4_28.1 => constants.%.4fe
// CHECK:STDOUT:   %pattern_type.loc4_24 => constants.%pattern_type.4f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
