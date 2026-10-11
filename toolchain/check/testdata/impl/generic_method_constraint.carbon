// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/generic_method_constraint.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/generic_method_constraint.carbon

// Regression tests for https://github.com/carbon-language/carbon-lang/issues/7919:
// building an impl witness thunk must not crash when an associated function's
// generic parameter constraint depends on an enclosing interface parameter.
// Coverage is via file_test (testdata/impl/), not a C++ unit test in check/.

// --- dependent_on_interface_param.carbon

library "[[@TEST_NAME]]";

interface A(T: type) {
  fn F(self);
}

interface B(T: type) {
  fn G[U: A(T)](u: U);
}

class C {}

//@dump-sem-ir-begin
impl C as B(()) {
  fn G[U: A(())](unused u: U) {}
}
//@dump-sem-ir-end

// --- dependent_on_self.carbon

library "[[@TEST_NAME]]";

interface A(T: type) {
  fn F(self);
}

interface B(T: type) {
  fn G[U: A(Self)](u: U);
}

class C {}

//@dump-sem-ir-begin
impl C as B(()) {
  fn G[U: A(C)](unused u: U) {}
}
//@dump-sem-ir-end

// --- independent_constraint_still_ok.carbon

library "[[@TEST_NAME]]";

interface A {
  fn F(self);
}

interface B(T: type) {
  fn G[U: A](u: U, t: T);
}

class C {}

//@dump-sem-ir-begin
impl C as B(()) {
  fn G[U: A](unused u: U, unused t: ()) {}
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- dependent_on_interface_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %A.type.55a: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.55a = struct_value () [concrete]
// CHECK:STDOUT:   %B.type.d1b: type = generic_interface_type @B [concrete]
// CHECK:STDOUT:   %B.generic: %B.type.d1b = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %B.type.b36: type = facet_type <@B, @B(%empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %.aec: <witness> = impl_self_witness %C, @B, @B(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness @C.as.B.impl.%B.impl_witness_table [concrete]
// CHECK:STDOUT:   %A.type.b01: type = facet_type <@A, @A(%empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %pattern_type.941: type = pattern_type %A.type.b01 [concrete]
// CHECK:STDOUT:   %U.patt.a8d: %pattern_type.941 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.1ad: %A.type.b01 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type.fe5: type = facet_access_type %U.1ad [symbolic]
// CHECK:STDOUT:   %pattern_type.3ba: type = pattern_type %U.as_type.fe5 [symbolic]
// CHECK:STDOUT:   %u.param_patt.893: %pattern_type.3ba = value_param_pattern [symbolic]
// CHECK:STDOUT:   %u.patt.47d: %pattern_type.3ba = wrapper_binding_pattern u, %u.param_patt.893 [symbolic]
// CHECK:STDOUT:   %C.as.B.impl.G.type: type = fn_type @C.as.B.impl.G [concrete]
// CHECK:STDOUT:   %C.as.B.impl.G: %C.as.B.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.as_type.fe5 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @C.as.B.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %B.ref: %B.type.d1b = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:     %.loc15_14: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_15: type = converted %.loc15_14, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %B.type: type = facet_type <@B, @B(constants.%empty_tuple.type)> [concrete = constants.%B.type.b36]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: <witness> = impl_self_witness @C.as.B.impl.%C.ref, @B, @B(constants.%empty_tuple.type) [concrete = constants.%.aec]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.B.impl: %C.ref as %B.type {
// CHECK:STDOUT:   %C.as.B.impl.G.decl: %C.as.B.impl.G.type = fn_decl @C.as.B.impl.G [concrete = constants.%C.as.B.impl.G] {
// CHECK:STDOUT:     %U.patt.loc16_9.1: %pattern_type.941 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc16_9.2 (constants.%U.patt.a8d)]
// CHECK:STDOUT:     %u.param_patt.loc16_26.1: @C.as.B.impl.G.%pattern_type (%pattern_type.3ba) = value_param_pattern [symbolic = %u.param_patt.loc16_26.2 (constants.%u.param_patt.893)]
// CHECK:STDOUT:     %u.patt.loc16_26.1: @C.as.B.impl.G.%pattern_type (%pattern_type.3ba) = wrapper_binding_pattern u, %u.param_patt.loc16_26.1 [symbolic = %u.patt.loc16_26.2 (constants.%u.patt.47d)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_15.1: type = splice_block %A.type [concrete = constants.%A.type.b01] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %A.ref: %A.type.55a = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %.loc16_14: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc16_15.2: type = converted %.loc16_14, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%empty_tuple.type)> [concrete = constants.%A.type.b01]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc16_9.2: %A.type.b01 = symbolic_binding U, 0 [symbolic = %U.loc16_9.1 (constants.%U.1ad)]
// CHECK:STDOUT:     %u.param: @C.as.B.impl.G.%U.as_type.loc16_28.1 (%U.as_type.fe5) = value_param call_param0
// CHECK:STDOUT:     %.loc16_28.1: type = splice_block %.loc16_28.2 [symbolic = %U.as_type.loc16_28.1 (constants.%U.as_type.fe5)] {
// CHECK:STDOUT:       %U.ref: %A.type.b01 = name_ref U, %U.loc16_9.2 [symbolic = %U.loc16_9.1 (constants.%U.1ad)]
// CHECK:STDOUT:       %U.as_type.loc16_28.2: type = facet_access_type %U.ref [symbolic = %U.as_type.loc16_28.1 (constants.%U.as_type.fe5)]
// CHECK:STDOUT:       %.loc16_28.2: type = converted %U.ref, %U.as_type.loc16_28.2 [symbolic = %U.as_type.loc16_28.1 (constants.%U.as_type.fe5)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @C.as.B.impl.G.%U.as_type.loc16_28.1 (%U.as_type.fe5) = wrapper_binding u, %u.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.impl_witness_table = impl_witness_table (%C.as.B.impl.G.decl), @C.as.B.impl [concrete]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness %B.impl_witness_table [concrete = constants.%B.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .G = %C.as.B.impl.G.decl
// CHECK:STDOUT:   extend %B.type
// CHECK:STDOUT:   witness = %B.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.B.impl.G(%U.loc16_9.2: %A.type.b01) {
// CHECK:STDOUT:   %U.patt.loc16_9.2: %pattern_type.941 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc16_9.2 (constants.%U.patt.a8d)]
// CHECK:STDOUT:   %U.loc16_9.1: %A.type.b01 = symbolic_binding U, 0 [symbolic = %U.loc16_9.1 (constants.%U.1ad)]
// CHECK:STDOUT:   %U.as_type.loc16_28.1: type = facet_access_type %U.loc16_9.1 [symbolic = %U.as_type.loc16_28.1 (constants.%U.as_type.fe5)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.as_type.loc16_28.1 [symbolic = %pattern_type (constants.%pattern_type.3ba)]
// CHECK:STDOUT:   %u.param_patt.loc16_26.2: @C.as.B.impl.G.%pattern_type (%pattern_type.3ba) = value_param_pattern [symbolic = %u.param_patt.loc16_26.2 (constants.%u.param_patt.893)]
// CHECK:STDOUT:   %u.patt.loc16_26.2: @C.as.B.impl.G.%pattern_type (%pattern_type.3ba) = wrapper_binding_pattern u, %u.param_patt.loc16_26.2 [symbolic = %u.patt.loc16_26.2 (constants.%u.patt.47d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.as_type.loc16_28.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @C.as.B.impl.G.%U.as_type.loc16_28.1 (%U.as_type.fe5)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.B.impl.G(constants.%U.1ad) {
// CHECK:STDOUT:   %U.patt.loc16_9.2 => constants.%U.patt.a8d
// CHECK:STDOUT:   %U.loc16_9.1 => constants.%U.1ad
// CHECK:STDOUT:   %U.as_type.loc16_28.1 => constants.%U.as_type.fe5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3ba
// CHECK:STDOUT:   %u.param_patt.loc16_26.2 => constants.%u.param_patt.893
// CHECK:STDOUT:   %u.patt.loc16_26.2 => constants.%u.patt.47d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- dependent_on_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %A.type.55a: type = generic_interface_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.generic: %A.type.55a = struct_value () [concrete]
// CHECK:STDOUT:   %B.type.d1b: type = generic_interface_type @B [concrete]
// CHECK:STDOUT:   %B.generic: %B.type.d1b = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %B.type.b36: type = facet_type <@B, @B(%empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %.aec: <witness> = impl_self_witness %C, @B, @B(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness @C.as.B.impl.%B.impl_witness_table [concrete]
// CHECK:STDOUT:   %A.type.2d7: type = facet_type <@A, @A(%C)> [concrete]
// CHECK:STDOUT:   %pattern_type.6d7: type = pattern_type %A.type.2d7 [concrete]
// CHECK:STDOUT:   %U.patt.5b3: %pattern_type.6d7 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.6c9: %A.type.2d7 = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type.66a: type = facet_access_type %U.6c9 [symbolic]
// CHECK:STDOUT:   %pattern_type.36f: type = pattern_type %U.as_type.66a [symbolic]
// CHECK:STDOUT:   %u.param_patt.124: %pattern_type.36f = value_param_pattern [symbolic]
// CHECK:STDOUT:   %u.patt.3f9: %pattern_type.36f = wrapper_binding_pattern u, %u.param_patt.124 [symbolic]
// CHECK:STDOUT:   %C.as.B.impl.G.type: type = fn_type @C.as.B.impl.G [concrete]
// CHECK:STDOUT:   %C.as.B.impl.G: %C.as.B.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.as_type.66a [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @C.as.B.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %B.ref: %B.type.d1b = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:     %.loc15_14: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_15: type = converted %.loc15_14, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %B.type: type = facet_type <@B, @B(constants.%empty_tuple.type)> [concrete = constants.%B.type.b36]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: <witness> = impl_self_witness @C.as.B.impl.%C.ref, @B, @B(constants.%empty_tuple.type) [concrete = constants.%.aec]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.B.impl: %C.ref as %B.type {
// CHECK:STDOUT:   %C.as.B.impl.G.decl: %C.as.B.impl.G.type = fn_decl @C.as.B.impl.G [concrete = constants.%C.as.B.impl.G] {
// CHECK:STDOUT:     %U.patt.loc16_9.1: %pattern_type.6d7 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc16_9.2 (constants.%U.patt.5b3)]
// CHECK:STDOUT:     %u.param_patt.loc16_25.1: @C.as.B.impl.G.%pattern_type (%pattern_type.36f) = value_param_pattern [symbolic = %u.param_patt.loc16_25.2 (constants.%u.param_patt.124)]
// CHECK:STDOUT:     %u.patt.loc16_25.1: @C.as.B.impl.G.%pattern_type (%pattern_type.36f) = wrapper_binding_pattern u, %u.param_patt.loc16_25.1 [symbolic = %u.patt.loc16_25.2 (constants.%u.patt.3f9)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_14: type = splice_block %A.type [concrete = constants.%A.type.2d7] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %A.ref: %A.type.55a = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %A.type: type = facet_type <@A, @A(constants.%C)> [concrete = constants.%A.type.2d7]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc16_9.2: %A.type.2d7 = symbolic_binding U, 0 [symbolic = %U.loc16_9.1 (constants.%U.6c9)]
// CHECK:STDOUT:     %u.param: @C.as.B.impl.G.%U.as_type.loc16_27.1 (%U.as_type.66a) = value_param call_param0
// CHECK:STDOUT:     %.loc16_27.1: type = splice_block %.loc16_27.2 [symbolic = %U.as_type.loc16_27.1 (constants.%U.as_type.66a)] {
// CHECK:STDOUT:       %U.ref: %A.type.2d7 = name_ref U, %U.loc16_9.2 [symbolic = %U.loc16_9.1 (constants.%U.6c9)]
// CHECK:STDOUT:       %U.as_type.loc16_27.2: type = facet_access_type %U.ref [symbolic = %U.as_type.loc16_27.1 (constants.%U.as_type.66a)]
// CHECK:STDOUT:       %.loc16_27.2: type = converted %U.ref, %U.as_type.loc16_27.2 [symbolic = %U.as_type.loc16_27.1 (constants.%U.as_type.66a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @C.as.B.impl.G.%U.as_type.loc16_27.1 (%U.as_type.66a) = wrapper_binding u, %u.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.impl_witness_table = impl_witness_table (%C.as.B.impl.G.decl), @C.as.B.impl [concrete]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness %B.impl_witness_table [concrete = constants.%B.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .G = %C.as.B.impl.G.decl
// CHECK:STDOUT:   extend %B.type
// CHECK:STDOUT:   witness = %B.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.B.impl.G(%U.loc16_9.2: %A.type.2d7) {
// CHECK:STDOUT:   %U.patt.loc16_9.2: %pattern_type.6d7 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc16_9.2 (constants.%U.patt.5b3)]
// CHECK:STDOUT:   %U.loc16_9.1: %A.type.2d7 = symbolic_binding U, 0 [symbolic = %U.loc16_9.1 (constants.%U.6c9)]
// CHECK:STDOUT:   %U.as_type.loc16_27.1: type = facet_access_type %U.loc16_9.1 [symbolic = %U.as_type.loc16_27.1 (constants.%U.as_type.66a)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.as_type.loc16_27.1 [symbolic = %pattern_type (constants.%pattern_type.36f)]
// CHECK:STDOUT:   %u.param_patt.loc16_25.2: @C.as.B.impl.G.%pattern_type (%pattern_type.36f) = value_param_pattern [symbolic = %u.param_patt.loc16_25.2 (constants.%u.param_patt.124)]
// CHECK:STDOUT:   %u.patt.loc16_25.2: @C.as.B.impl.G.%pattern_type (%pattern_type.36f) = wrapper_binding_pattern u, %u.param_patt.loc16_25.2 [symbolic = %u.patt.loc16_25.2 (constants.%u.patt.3f9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.as_type.loc16_27.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @C.as.B.impl.G.%U.as_type.loc16_27.1 (%U.as_type.66a)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.B.impl.G(constants.%U.6c9) {
// CHECK:STDOUT:   %U.patt.loc16_9.2 => constants.%U.patt.5b3
// CHECK:STDOUT:   %U.loc16_9.1 => constants.%U.6c9
// CHECK:STDOUT:   %U.as_type.loc16_27.1 => constants.%U.as_type.66a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.36f
// CHECK:STDOUT:   %u.param_patt.loc16_25.2 => constants.%u.param_patt.124
// CHECK:STDOUT:   %u.patt.loc16_25.2 => constants.%u.patt.3f9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- independent_constraint_still_ok.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A.type: type = facet_type <@A> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %B.type.d1b: type = generic_interface_type @B [concrete]
// CHECK:STDOUT:   %B.generic: %B.type.d1b = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.029: type = pattern_type %A.type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %B.type.b36: type = facet_type <@B, @B(%empty_tuple.type)> [concrete]
// CHECK:STDOUT:   %.aec: <witness> = impl_self_witness %C, @B, @B(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness @C.as.B.impl.%B.impl_witness_table [concrete]
// CHECK:STDOUT:   %U.patt.a4b: %pattern_type.029 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.626: %A.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type.3b8: type = facet_access_type %U.626 [symbolic]
// CHECK:STDOUT:   %pattern_type.a0e: type = pattern_type %U.as_type.3b8 [symbolic]
// CHECK:STDOUT:   %u.param_patt.5f6: %pattern_type.a0e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %u.patt.863: %pattern_type.a0e = wrapper_binding_pattern u, %u.param_patt.5f6 [symbolic]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %t.param_patt.e86: %pattern_type.cb1 = value_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.efd: %pattern_type.cb1 = wrapper_binding_pattern t, %t.param_patt.e86 [concrete]
// CHECK:STDOUT:   %C.as.B.impl.G.type: type = fn_type @C.as.B.impl.G [concrete]
// CHECK:STDOUT:   %C.as.B.impl.G: %C.as.B.impl.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.as_type.3b8 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @C.as.B.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %B.ref: %B.type.d1b = name_ref B, file.%B.decl [concrete = constants.%B.generic]
// CHECK:STDOUT:     %.loc15_14: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_15: type = converted %.loc15_14, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %B.type: type = facet_type <@B, @B(constants.%empty_tuple.type)> [concrete = constants.%B.type.b36]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: <witness> = impl_self_witness @C.as.B.impl.%C.ref, @B, @B(constants.%empty_tuple.type) [concrete = constants.%.aec]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.B.impl: %C.ref as %B.type {
// CHECK:STDOUT:   %C.as.B.impl.G.decl: %C.as.B.impl.G.type = fn_decl @C.as.B.impl.G [concrete = constants.%C.as.B.impl.G] {
// CHECK:STDOUT:     %U.patt.loc16_9.1: %pattern_type.029 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc16_9.2 (constants.%U.patt.a4b)]
// CHECK:STDOUT:     %u.param_patt.loc16_22.1: @C.as.B.impl.G.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %u.param_patt.loc16_22.2 (constants.%u.param_patt.5f6)]
// CHECK:STDOUT:     %u.patt.loc16_22.1: @C.as.B.impl.G.%pattern_type (%pattern_type.a0e) = wrapper_binding_pattern u, %u.param_patt.loc16_22.1 [symbolic = %u.patt.loc16_22.2 (constants.%u.patt.863)]
// CHECK:STDOUT:     %t.param_patt: %pattern_type.cb1 = value_param_pattern [concrete = constants.%t.param_patt.e86]
// CHECK:STDOUT:     %t.patt: %pattern_type.cb1 = wrapper_binding_pattern t, %t.param_patt [concrete = constants.%t.patt.efd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_11: type = splice_block %A.ref [concrete = constants.%A.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc16_9.2: %A.type = symbolic_binding U, 0 [symbolic = %U.loc16_9.1 (constants.%U.626)]
// CHECK:STDOUT:     %u.param: @C.as.B.impl.G.%U.as_type.loc16_24.1 (%U.as_type.3b8) = value_param call_param0
// CHECK:STDOUT:     %.loc16_24.1: type = splice_block %.loc16_24.2 [symbolic = %U.as_type.loc16_24.1 (constants.%U.as_type.3b8)] {
// CHECK:STDOUT:       %U.ref: %A.type = name_ref U, %U.loc16_9.2 [symbolic = %U.loc16_9.1 (constants.%U.626)]
// CHECK:STDOUT:       %U.as_type.loc16_24.2: type = facet_access_type %U.ref [symbolic = %U.as_type.loc16_24.1 (constants.%U.as_type.3b8)]
// CHECK:STDOUT:       %.loc16_24.2: type = converted %U.ref, %U.as_type.loc16_24.2 [symbolic = %U.as_type.loc16_24.1 (constants.%U.as_type.3b8)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @C.as.B.impl.G.%U.as_type.loc16_24.1 (%U.as_type.3b8) = wrapper_binding u, %u.param
// CHECK:STDOUT:     %t.param: %empty_tuple.type = value_param call_param1
// CHECK:STDOUT:     %.loc16_38.1: type = splice_block %.loc16_38.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc16_38.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc16_38.3: type = converted %.loc16_38.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: %empty_tuple.type = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.impl_witness_table = impl_witness_table (%C.as.B.impl.G.decl), @C.as.B.impl [concrete]
// CHECK:STDOUT:   %B.impl_witness: <witness> = impl_witness %B.impl_witness_table [concrete = constants.%B.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .G = %C.as.B.impl.G.decl
// CHECK:STDOUT:   extend %B.type
// CHECK:STDOUT:   witness = %B.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.B.impl.G(%U.loc16_9.2: %A.type) {
// CHECK:STDOUT:   %U.patt.loc16_9.2: %pattern_type.029 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc16_9.2 (constants.%U.patt.a4b)]
// CHECK:STDOUT:   %U.loc16_9.1: %A.type = symbolic_binding U, 0 [symbolic = %U.loc16_9.1 (constants.%U.626)]
// CHECK:STDOUT:   %U.as_type.loc16_24.1: type = facet_access_type %U.loc16_9.1 [symbolic = %U.as_type.loc16_24.1 (constants.%U.as_type.3b8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.as_type.loc16_24.1 [symbolic = %pattern_type (constants.%pattern_type.a0e)]
// CHECK:STDOUT:   %u.param_patt.loc16_22.2: @C.as.B.impl.G.%pattern_type (%pattern_type.a0e) = value_param_pattern [symbolic = %u.param_patt.loc16_22.2 (constants.%u.param_patt.5f6)]
// CHECK:STDOUT:   %u.patt.loc16_22.2: @C.as.B.impl.G.%pattern_type (%pattern_type.a0e) = wrapper_binding_pattern u, %u.param_patt.loc16_22.2 [symbolic = %u.patt.loc16_22.2 (constants.%u.patt.863)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.as_type.loc16_24.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: @C.as.B.impl.G.%U.as_type.loc16_24.1 (%U.as_type.3b8), %t.param: %empty_tuple.type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.B.impl.G(constants.%U.626) {
// CHECK:STDOUT:   %U.patt.loc16_9.2 => constants.%U.patt.a4b
// CHECK:STDOUT:   %U.loc16_9.1 => constants.%U.626
// CHECK:STDOUT:   %U.as_type.loc16_24.1 => constants.%U.as_type.3b8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a0e
// CHECK:STDOUT:   %u.param_patt.loc16_22.2 => constants.%u.param_patt.5f6
// CHECK:STDOUT:   %u.patt.loc16_22.2 => constants.%u.patt.863
// CHECK:STDOUT: }
// CHECK:STDOUT:
