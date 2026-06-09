// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/name_lookup_in_impl_definition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/name_lookup_in_impl_definition.carbon

// --- name_lookup_into_impl_as_target.carbon
library "[[@TEST_NAME]]";

interface I {
  let U:! type;
  fn F() -> U;
}

class C { adapt {}; }

impl forall [T:! type] T as I where .U = C {
  // We perform name lookup into `I` and resolve `U` to `T.(I.U)`.
  //@dump-sem-ir-begin
  fn F() -> U { return {} as C; }
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- name_lookup_into_impl_as_target.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @T.as.I.impl.%I.impl_witness_table, @T.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %.a44: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %T.as.I.impl.F.type: type = fn_type @T.as.I.impl.F, @T.as.I.impl(%T) [symbolic]
// CHECK:STDOUT:   %T.as.I.impl.F: %T.as.I.impl.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @T.as.I.impl(%T.loc10_15.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.I.impl.F.type: type = fn_type @T.as.I.impl.F, @T.as.I.impl(%T.loc10_15.1) [symbolic = %T.as.I.impl.F.type (constants.%T.as.I.impl.F.type)]
// CHECK:STDOUT:   %T.as.I.impl.F: @T.as.I.impl.%T.as.I.impl.F.type (%T.as.I.impl.F.type) = struct_value () [symbolic = %T.as.I.impl.F (constants.%T.as.I.impl.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %T.ref as %.loc10_31 {
// CHECK:STDOUT:     %T.as.I.impl.F.decl: @T.as.I.impl.%T.as.I.impl.F.type (%T.as.I.impl.F.type) = fn_decl @T.as.I.impl.F [symbolic = @T.as.I.impl.%T.as.I.impl.F (constants.%T.as.I.impl.F)] {
// CHECK:STDOUT:       %return.param_patt: %pattern_type.98b = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %U.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %I.facet.loc13_13.2: %I.type = facet_value @T.as.I.impl.%T.ref, (constants.%I.impl_witness) [symbolic = %I.facet.loc13_13.1 (constants.%I.facet)]
// CHECK:STDOUT:       %.loc13_13.1: %I.type = converted @T.as.I.impl.%T.ref, %I.facet.loc13_13.2 [symbolic = %I.facet.loc13_13.1 (constants.%I.facet)]
// CHECK:STDOUT:       %impl.elem0: type = impl_witness_access constants.%I.impl_witness, element0 [concrete = constants.%C]
// CHECK:STDOUT:       %U.ref: type = name_ref U, %impl.elem0 [concrete = constants.%C]
// CHECK:STDOUT:       %.loc13_13.2: Core.Form = init_form %U.ref [concrete = constants.%.a44]
// CHECK:STDOUT:       %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:       %return: ref %C = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .F = %T.as.I.impl.F.decl
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     extend %.loc10_31
// CHECK:STDOUT:     witness = %I.impl_witness.loc10_44.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @T.as.I.impl.F(@T.as.I.impl.%T.loc10_15.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @T.as.I.impl.%I.impl_witness_table, @T.as.I.impl(%T) [symbolic = %I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:   %I.facet.loc13_13.1: %I.type = facet_value %T, (%I.impl_witness) [symbolic = %I.facet.loc13_13.1 (constants.%I.facet)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %C {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc13_25: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc13_27.1: %C = as_compatible %empty_struct [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc13_27.2: %C = converted %.loc13_25, %.loc13_27.1 [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc13_31.1: %empty_struct_type = as_compatible %.loc13_27.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc13_31.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc13_31.3: init %C = as_compatible %.loc13_31.2 [concrete = constants.%C.val]
// CHECK:STDOUT:     %.loc13_31.4: init %C = converted %.loc13_27.2, %.loc13_31.3 [concrete = constants.%C.val]
// CHECK:STDOUT:     return %.loc13_31.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl(constants.%T) {
// CHECK:STDOUT:   %T.loc10_15.1 => constants.%T
// CHECK:STDOUT:   %I.impl_witness.loc10_44.2 => constants.%I.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as.I.impl.F.type => constants.%T.as.I.impl.F.type
// CHECK:STDOUT:   %T.as.I.impl.F => constants.%T.as.I.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T.as.I.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness
// CHECK:STDOUT:   %I.facet.loc13_13.1 => constants.%I.facet
// CHECK:STDOUT: }
// CHECK:STDOUT:
