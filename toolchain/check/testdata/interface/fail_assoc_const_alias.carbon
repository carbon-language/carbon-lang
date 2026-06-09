// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_assoc_const_alias.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_assoc_const_alias.carbon

// --- fail_alias_to_different_interface.carbon
library "[[@TEST_NAME]]";

interface I {
  let T:! type;
}

interface J {
  alias U = I.T;
  // CHECK:STDERR: fail_alias_to_different_interface.carbon:[[@LINE+4]]:13: error: cannot convert type `Self` that implements `J` into type implementing `I` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   fn F() -> U;
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  fn F() -> U;
}

// --- alias_to_different_interface_with_requires.carbon
library "[[@TEST_NAME]]";

interface I2 {
  let T2:! type;
}

interface J2;

impl forall [V:! J2] V as I2 where .T2 = () {}

//@dump-sem-ir-begin
interface J2 {
  alias U2 = I2.T2;
  fn F2() -> U2;
}
//@dump-sem-ir-end

// --- fail_call_method_alias.carbon
library "[[@TEST_NAME]]";

interface A {
  fn F(self) -> type;
}
interface B {
  alias F = A.F;
  // CHECK:STDERR: fail_call_method_alias.carbon:[[@LINE+4]]:13: error: cannot convert type `Self` that implements `B` into type implementing `A` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   fn G() -> F();
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  fn G() -> F();
}

// --- call_with_compound_member_access.carbon
library "[[@TEST_NAME]]";

interface C {
  fn F(self);
  // TODO: Add `default` once supported.
  fn G(self) {
    //@dump-sem-ir-begin
    self.F();
    self.(F)();
    //@dump-sem-ir-end
  }
}


// CHECK:STDOUT: --- alias_to_different_interface_with_requires.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I2.type: type = facet_type <@I2> [concrete]
// CHECK:STDOUT:   %I2.assoc_type: type = assoc_entity_type @I2 [concrete]
// CHECK:STDOUT:   %assoc0.d6c: %I2.assoc_type = assoc_entity element0, @I2.WithSelf.%T2 [concrete]
// CHECK:STDOUT:   %J2.type: type = facet_type <@J2> [concrete]
// CHECK:STDOUT:   %Self.c1f: %J2.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.c1f [symbolic]
// CHECK:STDOUT:   %I2.lookup_impl_witness.b5c: <witness> = lookup_impl_witness %Self.c1f, @I2 [symbolic]
// CHECK:STDOUT:   %.eef: require_specific_def_type = require_specific_def @V.as_type.as.I2.impl(%Self.c1f) [symbolic]
// CHECK:STDOUT:   %I2.facet.39b: %I2.type = facet_value %Self.as_type, (%I2.lookup_impl_witness.b5c) [symbolic]
// CHECK:STDOUT:   %impl.elem0.422: type = impl_witness_access %I2.lookup_impl_witness.b5c, element0 [symbolic]
// CHECK:STDOUT:   %.932: Core.Form = init_form %impl.elem0.422 [symbolic]
// CHECK:STDOUT:   %pattern_type.ff8: type = pattern_type %impl.elem0.422 [symbolic]
// CHECK:STDOUT:   %J2.WithSelf.F2.type: type = fn_type @J2.WithSelf.F2, @J2.WithSelf(%Self.c1f) [symbolic]
// CHECK:STDOUT:   %J2.WithSelf.F2: %J2.WithSelf.F2.type = struct_value () [symbolic]
// CHECK:STDOUT:   %J2.assoc_type: type = assoc_entity_type @J2 [concrete]
// CHECK:STDOUT:   %assoc0.253: %J2.assoc_type = assoc_entity element0, @J2.WithSelf.%J2.WithSelf.F2.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %J2.decl.loc12: type = interface_decl @J2 [concrete = constants.%J2.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J2 {
// CHECK:STDOUT:   %Self: %J2.type = symbolic_binding Self, 0 [symbolic = constants.%Self.c1f]
// CHECK:STDOUT:   %J2.WithSelf.decl = interface_with_self_decl @J2 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %I2.ref: type = name_ref I2, file.%I2.decl [concrete = constants.%I2.type]
// CHECK:STDOUT:   %T2.ref: %I2.assoc_type = name_ref T2, @T2.%assoc0 [concrete = constants.%assoc0.d6c]
// CHECK:STDOUT:   %U2: %I2.assoc_type = alias_binding U2, %T2.ref [concrete = constants.%assoc0.d6c]
// CHECK:STDOUT:   %J2.WithSelf.F2.decl: @J2.WithSelf.%J2.WithSelf.F2.type (%J2.WithSelf.F2.type) = fn_decl @J2.WithSelf.F2 [symbolic = @J2.WithSelf.%J2.WithSelf.F2 (constants.%J2.WithSelf.F2)] {
// CHECK:STDOUT:     %return.param_patt: @J2.WithSelf.F2.%pattern_type (%pattern_type.ff8) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @J2.WithSelf.F2.%pattern_type (%pattern_type.ff8) = return_slot_pattern %return.param_patt, %U2.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_type.loc14_14.2: type = facet_access_type @J2.%Self [symbolic = %Self.as_type.loc14_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     %I2.facet.loc14_14.2: %I2.type = facet_value %Self.as_type.loc14_14.2, (constants.%I2.lookup_impl_witness.b5c) [symbolic = %I2.facet.loc14_14.1 (constants.%I2.facet.39b)]
// CHECK:STDOUT:     %.loc14_14.3: %I2.type = converted @J2.%Self, %I2.facet.loc14_14.2 [symbolic = %I2.facet.loc14_14.1 (constants.%I2.facet.39b)]
// CHECK:STDOUT:     %impl.elem0.loc14_14.2: type = impl_witness_access constants.%I2.lookup_impl_witness.b5c, element0 [symbolic = %impl.elem0.loc14_14.1 (constants.%impl.elem0.422)]
// CHECK:STDOUT:     %U2.ref: type = name_ref U2, %impl.elem0.loc14_14.2 [symbolic = %impl.elem0.loc14_14.1 (constants.%impl.elem0.422)]
// CHECK:STDOUT:     %.loc14_14.4: Core.Form = init_form %U2.ref [symbolic = %.loc14_14.2 (constants.%.932)]
// CHECK:STDOUT:     %return.param: ref @J2.WithSelf.F2.%impl.elem0.loc14_14.1 (%impl.elem0.422) = out_param call_param0
// CHECK:STDOUT:     %return: ref @J2.WithSelf.F2.%impl.elem0.loc14_14.1 (%impl.elem0.422) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %J2.assoc_type = assoc_entity element0, %J2.WithSelf.F2.decl [concrete = constants.%assoc0.253]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .I2 = <poisoned>
// CHECK:STDOUT:   .I2 = <poisoned>
// CHECK:STDOUT:   .U2 = @J2.WithSelf.%U2
// CHECK:STDOUT:   .F2 = @J2.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@J2.WithSelf.%J2.WithSelf.F2.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @J2.WithSelf.F2(@J2.%Self: %J2.type) {
// CHECK:STDOUT:   %Self: %J2.type = symbolic_binding Self, 0 [symbolic = %Self (constants.%Self.c1f)]
// CHECK:STDOUT:   %Self.as_type.loc14_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %.loc14_14.1: require_specific_def_type = require_specific_def @V.as_type.as.I2.impl(%Self) [symbolic = %.loc14_14.1 (constants.%.eef)]
// CHECK:STDOUT:   %I2.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @I2 [symbolic = %I2.lookup_impl_witness (constants.%I2.lookup_impl_witness.b5c)]
// CHECK:STDOUT:   %I2.facet.loc14_14.1: %I2.type = facet_value %Self.as_type.loc14_14.1, (%I2.lookup_impl_witness) [symbolic = %I2.facet.loc14_14.1 (constants.%I2.facet.39b)]
// CHECK:STDOUT:   %impl.elem0.loc14_14.1: type = impl_witness_access %I2.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_14.1 (constants.%impl.elem0.422)]
// CHECK:STDOUT:   %.loc14_14.2: Core.Form = init_form %impl.elem0.loc14_14.1 [symbolic = %.loc14_14.2 (constants.%.932)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %impl.elem0.loc14_14.1 [symbolic = %pattern_type (constants.%pattern_type.ff8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @J2.WithSelf.F2.%impl.elem0.loc14_14.1 (%impl.elem0.422);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2.WithSelf(constants.%Self.c1f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @J2.WithSelf.F2(constants.%Self.c1f) {
// CHECK:STDOUT:   %Self => constants.%Self.c1f
// CHECK:STDOUT:   %Self.as_type.loc14_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %.loc14_14.1 => constants.%.eef
// CHECK:STDOUT:   %I2.lookup_impl_witness => constants.%I2.lookup_impl_witness.b5c
// CHECK:STDOUT:   %I2.facet.loc14_14.1 => constants.%I2.facet.39b
// CHECK:STDOUT:   %impl.elem0.loc14_14.1 => constants.%impl.elem0.422
// CHECK:STDOUT:   %.loc14_14.2 => constants.%.932
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ff8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_with_compound_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C.type: type = facet_type <@C> [concrete]
// CHECK:STDOUT:   %Self: %C.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %C.WithSelf.F.type: type = fn_type @C.WithSelf.F, @C.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.WithSelf.F: %C.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %C.assoc_type: type = assoc_entity_type @C [concrete]
// CHECK:STDOUT:   %assoc0: %C.assoc_type = assoc_entity element0, @C.WithSelf.%C.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %C.WithSelf.G.type: type = fn_type @C.WithSelf.G, @C.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %C.WithSelf.G: %C.WithSelf.G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %C.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @C [symbolic]
// CHECK:STDOUT:   %.b5a: type = fn_type_with_self_type %C.WithSelf.F.type, %Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.b5a = impl_witness_access %C.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @C.WithSelf.F(%Self) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @C {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @C.WithSelf.%assoc0
// CHECK:STDOUT:   .G = @C.WithSelf.%assoc1
// CHECK:STDOUT:   witness = (@C.WithSelf.%C.WithSelf.F.decl, @C.WithSelf.%C.WithSelf.G.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.WithSelf.G(@C.%Self: %C.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %C.WithSelf.F.type: type = fn_type @C.WithSelf.F, @C.WithSelf(%Self) [symbolic = %C.WithSelf.F.type (constants.%C.WithSelf.F.type)]
// CHECK:STDOUT:   %.loc8_9: type = fn_type_with_self_type %C.WithSelf.F.type, %Self [symbolic = %.loc8_9 (constants.%.b5a)]
// CHECK:STDOUT:   %C.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @C [symbolic = %C.lookup_impl_witness (constants.%C.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc8_9.2: @C.WithSelf.G.%.loc8_9 (%.b5a) = impl_witness_access %C.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8_9.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc8_9.2: <specific function> = specific_impl_function %impl.elem0.loc8_9.2, @C.WithSelf.F(%Self) [symbolic = %specific_impl_fn.loc8_9.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.WithSelf.G.%Self.as_type.loc6_8.1 (%Self.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref.loc8: @C.WithSelf.G.%Self.as_type.loc6_8.1 (%Self.as_type) = name_ref self, %self
// CHECK:STDOUT:     %F.ref.loc8: %C.assoc_type = name_ref F, @C.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc8_9.1: @C.WithSelf.G.%.loc8_9 (%.b5a) = impl_witness_access constants.%C.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8_9.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc8_9: <bound method> = bound_method %self.ref.loc8, %impl.elem0.loc8_9.1
// CHECK:STDOUT:     %.loc8_12: %C.type = converted constants.%Self.as_type, constants.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %specific_impl_fn.loc8_9.1: <specific function> = specific_impl_function %impl.elem0.loc8_9.1, @C.WithSelf.F(constants.%Self) [symbolic = %specific_impl_fn.loc8_9.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc8_12: <bound method> = bound_method %self.ref.loc8, %specific_impl_fn.loc8_9.1
// CHECK:STDOUT:     %C.WithSelf.F.call.loc8: init %empty_tuple.type = call %bound_method.loc8_12(%self.ref.loc8)
// CHECK:STDOUT:     %self.ref.loc9: @C.WithSelf.G.%Self.as_type.loc6_8.1 (%Self.as_type) = name_ref self, %self
// CHECK:STDOUT:     %impl.elem0.loc9: @C.WithSelf.G.%.loc8_9 (%.b5a) = impl_witness_access constants.%C.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc8_9.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %F.ref.loc9: @C.WithSelf.G.%.loc8_9 (%.b5a) = name_ref F, %impl.elem0.loc9 [symbolic = %impl.elem0.loc8_9.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc9_9: <bound method> = bound_method %self.ref.loc9, %F.ref.loc9
// CHECK:STDOUT:     %.loc9: %C.type = converted constants.%Self.as_type, constants.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:     %specific_impl_fn.loc9: <specific function> = specific_impl_function %F.ref.loc9, @C.WithSelf.F(constants.%Self) [symbolic = %specific_impl_fn.loc8_9.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc9_14: <bound method> = bound_method %self.ref.loc9, %specific_impl_fn.loc9
// CHECK:STDOUT:     %C.WithSelf.F.call.loc9: init %empty_tuple.type = call %bound_method.loc9_14(%self.ref.loc9)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %C.WithSelf.F.type => constants.%C.WithSelf.F.type
// CHECK:STDOUT:   %C.WithSelf.F => constants.%C.WithSelf.F
// CHECK:STDOUT:   %C.WithSelf.G.type => constants.%C.WithSelf.G.type
// CHECK:STDOUT:   %C.WithSelf.G => constants.%C.WithSelf.G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.WithSelf.G(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc6_8.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
