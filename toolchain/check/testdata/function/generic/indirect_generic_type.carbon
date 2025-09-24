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
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %ptr.a13: type = ptr_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %pattern_type.12d: type = pattern_type %ptr.a13 [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.cb2: <witness> = lookup_impl_witness %ptr.79f, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.79f, (%Copy.lookup_impl_witness.cb2) [symbolic]
// CHECK:STDOUT:   %.1cc: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.751: %.1cc = impl_witness_access %Copy.lookup_impl_witness.cb2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f44: <specific function> = specific_impl_function %impl.elem0.751, @Copy.Op(%Copy.facet) [symbolic]
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
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc4_20.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.cb2)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.loc4_20.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet)]
// CHECK:STDOUT:   %.loc6_10.4: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc6_10.4 (constants.%.1cc)]
// CHECK:STDOUT:   %impl.elem0.loc6_10.2: @F.%.loc6_10.4 (%.1cc) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_10.2 (constants.%impl.elem0.751)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2: <specific function> = specific_impl_function %impl.elem0.loc6_10.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc6_10.2 (constants.%specific_impl_fn.f44)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%ptr.loc4_21.1 (%ptr.a13)) -> @F.%ptr.loc4_20.1 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc4_21.1 (%ptr.a13) = name_ref p, %p
// CHECK:STDOUT:     %.loc6_10.1: ref @F.%ptr.loc4_20.1 (%ptr.79f) = deref %p.ref
// CHECK:STDOUT:     %.loc6_10.2: @F.%ptr.loc4_20.1 (%ptr.79f) = bind_value %.loc6_10.1
// CHECK:STDOUT:     %impl.elem0.loc6_10.1: @F.%.loc6_10.4 (%.1cc) = impl_witness_access constants.%Copy.lookup_impl_witness.cb2, element0 [symbolic = %impl.elem0.loc6_10.2 (constants.%impl.elem0.751)]
// CHECK:STDOUT:     %bound_method.loc6_10.1: <bound method> = bound_method %.loc6_10.2, %impl.elem0.loc6_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_10.1: <specific function> = specific_impl_function %impl.elem0.loc6_10.1, @Copy.Op(constants.%Copy.facet) [symbolic = %specific_impl_fn.loc6_10.2 (constants.%specific_impl_fn.f44)]
// CHECK:STDOUT:     %bound_method.loc6_10.2: <bound method> = bound_method %.loc6_10.2, %specific_impl_fn.loc6_10.1
// CHECK:STDOUT:     %.loc6_10.3: init @F.%ptr.loc4_20.1 (%ptr.79f) = call %bound_method.loc6_10.2(%.loc6_10.2)
// CHECK:STDOUT:     return %.loc6_10.3 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc4_6.1 => constants.%T.8b3
// CHECK:STDOUT:   %ptr.loc4_20.1 => constants.%ptr.79f
// CHECK:STDOUT:   %ptr.loc4_21.1 => constants.%ptr.a13
// CHECK:STDOUT:   %pattern_type.loc4_16 => constants.%pattern_type.12d
// CHECK:STDOUT:   %pattern_type.loc4_24 => constants.%pattern_type.afe
// CHECK:STDOUT: }
// CHECK:STDOUT:
