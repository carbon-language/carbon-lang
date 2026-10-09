// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/import_convert_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/import_convert_function.carbon

// --- library.carbon

package P library "[[@TEST_NAME]]";

class C(N: i32) {}
class D { var n: i32; var m: i32; }
fn Make() -> D { return {.n = 0, .m = 0}; }

impl C(0) as Core.ImplicitAs(D) { fn Convert(unused self) -> D { return Make(); } }
impl C(1) as Core.ImplicitAs(D) { fn Convert(unused self) -> D { return Make(); } }
impl C(2) as Core.ImplicitAs(D) { fn Convert(unused self) -> D { return Make(); } }
impl C(3) as Core.ImplicitAs(D) { fn Convert(unused self) -> D { return Make(); } }
impl C(4) as Core.ImplicitAs(D) { fn Convert(unused self) -> D { return Make(); } }
impl C(5) as Core.ImplicitAs(D) { fn Convert(unused self) -> D { return Make(); } }
impl C(6) as Core.ImplicitAs(D) { fn Convert(unused self) -> D { return Make(); } }
impl C(7) as Core.ImplicitAs(D) { fn Convert(unused self) -> D { return Make(); } }

// --- user.carbon

import P library "library";

// Check that we don't crash if we import functions and reallocate the function
// value store while handling a `return` statement.
fn F0(unused n: i32) -> P.D {
  if (false) { return {} as P.C(0); }
  if (false) { return {} as P.C(1); }
  if (false) { return {} as P.C(2); }
  if (false) { return {} as P.C(3); }
  if (false) { return {} as P.C(4); }
  if (false) { return {} as P.C(5); }
  if (false) { return {} as P.C(6); }
  if (false) { return {} as P.C(7); }
  return P.Make();
}


// CHECK:STDOUT: --- library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.b2d: %pattern_type.9d9 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.5d3: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.65b: type = class_type @C, @C(%N.5d3) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.523: type = struct_type {.n: %i32, .m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.a39: <witness> = complete_type_witness %struct_type.n.m.523 [concrete]
// CHECK:STDOUT:   %.128: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %pattern_type.b22: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %return.param_patt.dca: %pattern_type.b22 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.904: %pattern_type.b22 = return_slot_pattern %return.param_patt.dca, %D [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.819: type = struct_type {.n: Core.IntLiteral, .m: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.n.m.819 = struct_value (%int_0.5c6, %int_0.5c6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.09b: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ec2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eeb: %ImplicitAs.type.09b = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ec2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.eeb) [concrete]
// CHECK:STDOUT:   %.f0f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e04, %ImplicitAs.facet.eeb [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.935: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.614: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.2b2: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_0.2b2, %int_0.2b2) [concrete]
// CHECK:STDOUT:   %C.c81: type = class_type @C, @C(%int_0.2b2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.164: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %.907: <witness> = impl_self_witness %C.c81, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.8ed: <witness> = impl_witness @C.as.ImplicitAs.impl.aa1.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.6e1: type = pattern_type %C.c81 [concrete]
// CHECK:STDOUT:   %self.param_patt.36c: %pattern_type.6e1 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.5d9: %pattern_type.6e1 = wrapper_binding_pattern self, %self.param_patt.36c [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.1: type = fn_type @C.as.ImplicitAs.impl.Convert.loc8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.1: %C.as.ImplicitAs.impl.Convert.type.860b0a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8f9: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.aa2: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.8e8: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.d0f: type = class_type @C, @C(%int_1.8e8) [concrete]
// CHECK:STDOUT:   %.402: <witness> = impl_self_witness %C.d0f, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.7ef: <witness> = impl_witness @C.as.ImplicitAs.impl.d17.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.a65: type = pattern_type %C.d0f [concrete]
// CHECK:STDOUT:   %self.param_patt.afa: %pattern_type.a65 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e09: %pattern_type.a65 = wrapper_binding_pattern self, %self.param_patt.afa [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.2: type = fn_type @C.as.ImplicitAs.impl.Convert.loc9 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.2: %C.as.ImplicitAs.impl.Convert.type.860b0a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8fb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.3e0: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.cb7: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.d30: type = class_type @C, @C(%int_2.cb7) [concrete]
// CHECK:STDOUT:   %.b0d: <witness> = impl_self_witness %C.d30, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d13: <witness> = impl_witness @C.as.ImplicitAs.impl.7bc.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.e2a: type = pattern_type %C.d30 [concrete]
// CHECK:STDOUT:   %self.param_patt.259: %pattern_type.e2a = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.962: %pattern_type.e2a = wrapper_binding_pattern self, %self.param_patt.259 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.3: type = fn_type @C.as.ImplicitAs.impl.Convert.loc10 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.3: %C.as.ImplicitAs.impl.Convert.type.860b0a.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2b8: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.d29: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.af5: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.d26: type = class_type @C, @C(%int_3.af5) [concrete]
// CHECK:STDOUT:   %.ba9: <witness> = impl_self_witness %C.d26, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9a2: <witness> = impl_witness @C.as.ImplicitAs.impl.347.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.626: type = pattern_type %C.d26 [concrete]
// CHECK:STDOUT:   %self.param_patt.408: %pattern_type.626 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1be: %pattern_type.626 = wrapper_binding_pattern self, %self.param_patt.408 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.4: type = fn_type @C.as.ImplicitAs.impl.Convert.loc11 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.4: %C.as.ImplicitAs.impl.Convert.type.860b0a.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.fa9: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.a80: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.380: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %C.ddc: type = class_type @C, @C(%int_4.380) [concrete]
// CHECK:STDOUT:   %.e5f: <witness> = impl_self_witness %C.ddc, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6ef: <witness> = impl_witness @C.as.ImplicitAs.impl.606.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.fac: type = pattern_type %C.ddc [concrete]
// CHECK:STDOUT:   %self.param_patt.dae: %pattern_type.fac = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.578: %pattern_type.fac = wrapper_binding_pattern self, %self.param_patt.dae [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.5: type = fn_type @C.as.ImplicitAs.impl.Convert.loc12 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.5: %C.as.ImplicitAs.impl.Convert.type.860b0a.5 = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.669: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.019: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.c30: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %C.d90: type = class_type @C, @C(%int_5.c30) [concrete]
// CHECK:STDOUT:   %.22c: <witness> = impl_self_witness %C.d90, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.b75: <witness> = impl_witness @C.as.ImplicitAs.impl.c44.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.0f2: type = pattern_type %C.d90 [concrete]
// CHECK:STDOUT:   %self.param_patt.b07: %pattern_type.0f2 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.6de: %pattern_type.0f2 = wrapper_binding_pattern self, %self.param_patt.b07 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.6: type = fn_type @C.as.ImplicitAs.impl.Convert.loc13 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.6: %C.as.ImplicitAs.impl.Convert.type.860b0a.6 = struct_value () [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8f1: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.5fe: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.b5e: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %C.a8a: type = class_type @C, @C(%int_6.b5e) [concrete]
// CHECK:STDOUT:   %.f22: <witness> = impl_self_witness %C.a8a, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9e0: <witness> = impl_witness @C.as.ImplicitAs.impl.911.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.691: type = pattern_type %C.a8a [concrete]
// CHECK:STDOUT:   %self.param_patt.fef: %pattern_type.691 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1a1: %pattern_type.691 = wrapper_binding_pattern self, %self.param_patt.fef [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.7: type = fn_type @C.as.ImplicitAs.impl.Convert.loc14 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.7: %C.as.ImplicitAs.impl.Convert.type.860b0a.7 = struct_value () [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.27f: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.f1e: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.b03: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %C.1c9: type = class_type @C, @C(%int_7.b03) [concrete]
// CHECK:STDOUT:   %.f5b: <witness> = impl_self_witness %C.1c9, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c48: <witness> = impl_witness @C.as.ImplicitAs.impl.29b.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.d64: type = pattern_type %C.1c9 [concrete]
// CHECK:STDOUT:   %self.param_patt.6d1: %pattern_type.d64 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.051: %pattern_type.d64 = wrapper_binding_pattern self, %self.param_patt.6d1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.8: type = fn_type @C.as.ImplicitAs.impl.Convert.loc15 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.8: %C.as.ImplicitAs.impl.Convert.type.860b0a.8 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %N.patt.loc4_10.1: %pattern_type.9d9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_10.2 (constants.%N.patt.b2d)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_10.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_10.1 (constants.%N.5d3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc6_14: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.aa1 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:     %bound_method.loc8_9.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.935]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_9.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.614]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_9.2(%int_0) [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:     %.loc8_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:     %.loc8_9.2: %i32 = converted %int_0, %.loc8_9.1 [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_0.2b2) [concrete = constants.%C.c81]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = impl_self_witness @C.as.ImplicitAs.impl.aa1.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.907]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.d17 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %impl.elem0: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:     %bound_method.loc9_9.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8f9]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_9.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method.aa2]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_9.2(%int_1) [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:     %.loc9_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:     %.loc9_9.2: %i32 = converted %int_1, %.loc9_9.1 [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_1.8e8) [concrete = constants.%C.d0f]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: <witness> = impl_self_witness @C.as.ImplicitAs.impl.d17.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.402]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.7bc [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:     %bound_method.loc10_9.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8fb]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_9.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.3e0]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc10_9.2(%int_2) [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:     %.loc10_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:     %.loc10_9.2: %i32 = converted %int_2, %.loc10_9.1 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_2.cb7) [concrete = constants.%C.d30]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <witness> = impl_self_witness @C.as.ImplicitAs.impl.7bc.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.b0d]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.347 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:     %bound_method.loc11_9.1: <bound method> = bound_method %int_3, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2b8]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_9.2: <bound method> = bound_method %int_3, %specific_fn [concrete = constants.%bound_method.d29]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_9.2(%int_3) [concrete = constants.%int_3.af5]
// CHECK:STDOUT:     %.loc11_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.af5]
// CHECK:STDOUT:     %.loc11_9.2: %i32 = converted %int_3, %.loc11_9.1 [concrete = constants.%int_3.af5]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_3.af5) [concrete = constants.%C.d26]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = impl_self_witness @C.as.ImplicitAs.impl.347.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.ba9]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.606 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %impl.elem0: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:     %bound_method.loc12_9.1: <bound method> = bound_method %int_4, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.fa9]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc12_9.2: <bound method> = bound_method %int_4, %specific_fn [concrete = constants.%bound_method.a80]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_9.2(%int_4) [concrete = constants.%int_4.380]
// CHECK:STDOUT:     %.loc12_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_4.380]
// CHECK:STDOUT:     %.loc12_9.2: %i32 = converted %int_4, %.loc12_9.1 [concrete = constants.%int_4.380]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_4.380) [concrete = constants.%C.ddc]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: <witness> = impl_self_witness @C.as.ImplicitAs.impl.606.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.e5f]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.c44 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:     %impl.elem0: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:     %bound_method.loc13_9.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.669]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc13_9.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method.019]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_9.2(%int_5) [concrete = constants.%int_5.c30]
// CHECK:STDOUT:     %.loc13_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.c30]
// CHECK:STDOUT:     %.loc13_9.2: %i32 = converted %int_5, %.loc13_9.1 [concrete = constants.%int_5.c30]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_5.c30) [concrete = constants.%C.d90]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = impl_self_witness @C.as.ImplicitAs.impl.c44.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.22c]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.911 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:     %impl.elem0: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %int_6, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8f1]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %int_6, %specific_fn [concrete = constants.%bound_method.5fe]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc14_9.2(%int_6) [concrete = constants.%int_6.b5e]
// CHECK:STDOUT:     %.loc14_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_6.b5e]
// CHECK:STDOUT:     %.loc14_9.2: %i32 = converted %int_6, %.loc14_9.1 [concrete = constants.%int_6.b5e]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_6.b5e) [concrete = constants.%C.a8a]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: <witness> = impl_self_witness @C.as.ImplicitAs.impl.911.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.f22]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.29b [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:     %impl.elem0: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:     %bound_method.loc15_9.1: <bound method> = bound_method %int_7, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.27f]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc15_9.2: <bound method> = bound_method %int_7, %specific_fn [concrete = constants.%bound_method.f1e]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_9.2(%int_7) [concrete = constants.%int_7.b03]
// CHECK:STDOUT:     %.loc15_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_7.b03]
// CHECK:STDOUT:     %.loc15_9.2: %i32 = converted %int_7, %.loc15_9.1 [concrete = constants.%int_7.b03]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_7.b03) [concrete = constants.%C.1c9]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: <witness> = impl_self_witness @C.as.ImplicitAs.impl.29b.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.f5b]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.aa1: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.1 = fn_decl @C.as.ImplicitAs.impl.Convert.loc8 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6e1 = value_param_pattern [concrete = constants.%self.param_patt.36c]
// CHECK:STDOUT:     %self.patt: %pattern_type.6e1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5d9]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc8_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.c81 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.aa1.%C [concrete = constants.%C.c81]
// CHECK:STDOUT:     %self: %C.c81 = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.aa1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.8ed]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.d17: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.2 = fn_decl @C.as.ImplicitAs.impl.Convert.loc9 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a65 = value_param_pattern [concrete = constants.%self.param_patt.afa]
// CHECK:STDOUT:     %self.patt: %pattern_type.a65 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e09]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc9_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.d0f = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.d17.%C [concrete = constants.%C.d0f]
// CHECK:STDOUT:     %self: %C.d0f = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.d17 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.7ef]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.7bc: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.3 = fn_decl @C.as.ImplicitAs.impl.Convert.loc10 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e2a = value_param_pattern [concrete = constants.%self.param_patt.259]
// CHECK:STDOUT:     %self.patt: %pattern_type.e2a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.962]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc10_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.d30 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.7bc.%C [concrete = constants.%C.d30]
// CHECK:STDOUT:     %self: %C.d30 = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.7bc [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.d13]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.347: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.4 = fn_decl @C.as.ImplicitAs.impl.Convert.loc11 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.626 = value_param_pattern [concrete = constants.%self.param_patt.408]
// CHECK:STDOUT:     %self.patt: %pattern_type.626 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1be]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc11_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.d26 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.347.%C [concrete = constants.%C.d26]
// CHECK:STDOUT:     %self: %C.d26 = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.347 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.9a2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.606: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.5 = fn_decl @C.as.ImplicitAs.impl.Convert.loc12 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fac = value_param_pattern [concrete = constants.%self.param_patt.dae]
// CHECK:STDOUT:     %self.patt: %pattern_type.fac = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.578]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc12_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.ddc = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.606.%C [concrete = constants.%C.ddc]
// CHECK:STDOUT:     %self: %C.ddc = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.606 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.6ef]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.c44: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.6 = fn_decl @C.as.ImplicitAs.impl.Convert.loc13 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0f2 = value_param_pattern [concrete = constants.%self.param_patt.b07]
// CHECK:STDOUT:     %self.patt: %pattern_type.0f2 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.6de]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc13_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.d90 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.c44.%C [concrete = constants.%C.d90]
// CHECK:STDOUT:     %self: %C.d90 = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.c44 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.b75]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.911: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.7 = fn_decl @C.as.ImplicitAs.impl.Convert.loc14 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.691 = value_param_pattern [concrete = constants.%self.param_patt.fef]
// CHECK:STDOUT:     %self.patt: %pattern_type.691 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1a1]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc14_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.a8a = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.911.%C [concrete = constants.%C.a8a]
// CHECK:STDOUT:     %self: %C.a8a = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.911 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.9e0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.29b: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.8 = fn_decl @C.as.ImplicitAs.impl.Convert.loc15 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d64 = value_param_pattern [concrete = constants.%self.param_patt.6d1]
// CHECK:STDOUT:     %self.patt: %pattern_type.d64 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.051]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc15_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.1c9 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.29b.%C [concrete = constants.%C.1c9]
// CHECK:STDOUT:     %self: %C.1c9 = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.29b [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.c48]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%N.loc4_10.2: %i32) {
// CHECK:STDOUT:   %N.patt.loc4_10.2: %pattern_type.9d9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_10.2 (constants.%N.patt.b2d)]
// CHECK:STDOUT:   %N.loc4_10.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_10.1 (constants.%N.5d3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.65b
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %field_decl.loc5_16: %D.elem = field_decl n, element0, %i32.loc5_18 in [concrete] {
// CHECK:STDOUT:     %i32.loc5_18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %field_decl.loc5_28: %D.elem = field_decl m, element1, %i32.loc5_30 in [concrete] {
// CHECK:STDOUT:     %i32.loc5_30: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n.m.523 [concrete = constants.%complete_type.a39]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .n = %field_decl.loc5_16
// CHECK:STDOUT:   .m = %field_decl.loc5_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc6_31: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_0.loc6_39: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc6_40.1: %struct_type.n.m.819 = struct_literal (%int_0.loc6_31, %int_0.loc6_39) [concrete = constants.%struct]
// CHECK:STDOUT:   %impl.elem0.loc6_40.1: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc6_40.1: <bound method> = bound_method %int_0.loc6_31, %impl.elem0.loc6_40.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.935]
// CHECK:STDOUT:   %specific_fn.loc6_40.1: <specific function> = specific_function %impl.elem0.loc6_40.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.2: <bound method> = bound_method %int_0.loc6_31, %specific_fn.loc6_40.1 [concrete = constants.%bound_method.614]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.1: init %i32 = call %bound_method.loc6_40.2(%int_0.loc6_31) [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:   %.loc6_40.2: init %i32 = converted %int_0.loc6_31, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.1 [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:   %.loc6_40.3: ref %i32 = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc6_40.4: init %i32 to %.loc6_40.3 = in_place_init %.loc6_40.2 [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:   %impl.elem0.loc6_40.2: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc6_40.3: <bound method> = bound_method %int_0.loc6_39, %impl.elem0.loc6_40.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.935]
// CHECK:STDOUT:   %specific_fn.loc6_40.2: <specific function> = specific_function %impl.elem0.loc6_40.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.4: <bound method> = bound_method %int_0.loc6_39, %specific_fn.loc6_40.2 [concrete = constants.%bound_method.614]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.2: init %i32 = call %bound_method.loc6_40.4(%int_0.loc6_39) [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:   %.loc6_40.5: init %i32 = converted %int_0.loc6_39, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.2 [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:   %.loc6_40.6: ref %i32 = class_element_access %return.param, element1
// CHECK:STDOUT:   %.loc6_40.7: init %i32 to %.loc6_40.6 = in_place_init %.loc6_40.5 [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:   %.loc6_40.8: init %D to %return.param = class_init (%.loc6_40.4, %.loc6_40.7) [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc6_41: init %D = converted %.loc6_40.1, %.loc6_40.8 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc6_41 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc8(%self.param: %C.c81) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc8_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc8_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc9(%self.param: %C.d0f) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc9_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc9_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc10(%self.param: %C.d30) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc10_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc10_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc11(%self.param: %C.d26) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc11_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc11_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc12(%self.param: %C.ddc) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc12_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc12_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc13(%self.param: %C.d90) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc13_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc14(%self.param: %C.a8a) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc14_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc15(%self.param: %C.1c9) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc15_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc15_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.5d3) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.b2d
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%N.5d3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.2b2) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.b2d
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_0.2b2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.8e8) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.b2d
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_1.8e8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.cb7) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.b2d
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_2.cb7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.af5) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.b2d
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_3.af5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.380) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.b2d
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_4.380
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.c30) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.b2d
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_5.c30
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.b5e) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.b2d
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_6.b5e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.b03) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.b2d
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_7.b03
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %n.param_patt: %pattern_type.9d9 = value_param_pattern [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.9d9 = wrapper_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %struct_type.n.m: type = struct_type {.n: %i32, .m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.a39: <witness> = complete_type_witness %struct_type.n.m [concrete]
// CHECK:STDOUT:   %.128: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %pattern_type.b22: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %return.param_patt.dca: %pattern_type.b22 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.904: %pattern_type.b22 = return_slot_pattern %return.param_patt.dca, %D [concrete]
// CHECK:STDOUT:   %F0.type: type = fn_type @F0 [concrete]
// CHECK:STDOUT:   %F0: %F0.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %N.5d3: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.b2d: %pattern_type.9d9 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.09b: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ec2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eeb: %ImplicitAs.type.09b = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ec2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.eeb) [concrete]
// CHECK:STDOUT:   %.f0f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e04, %ImplicitAs.facet.eeb [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.935: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.614: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.2b2: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %C.c81: type = class_type @C, @C(%int_0.2b2) [concrete]
// CHECK:STDOUT:   %C.val.c2e: %C.c81 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.164: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %int_1.8e8: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.d0f: type = class_type @C, @C(%int_1.8e8) [concrete]
// CHECK:STDOUT:   %int_2.cb7: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.d30: type = class_type @C, @C(%int_2.cb7) [concrete]
// CHECK:STDOUT:   %int_3.af5: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.d26: type = class_type @C, @C(%int_3.af5) [concrete]
// CHECK:STDOUT:   %int_4.380: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %C.ddc: type = class_type @C, @C(%int_4.380) [concrete]
// CHECK:STDOUT:   %int_5.c30: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %C.d90: type = class_type @C, @C(%int_5.c30) [concrete]
// CHECK:STDOUT:   %int_6.b5e: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %C.a8a: type = class_type @C, @C(%int_6.b5e) [concrete]
// CHECK:STDOUT:   %int_7.b03: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %C.1c9: type = class_type @C, @C(%int_7.b03) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4c8: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f50 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.bed: %ImplicitAs.type.164 = facet_value %C.c81, (%ImplicitAs.impl_witness.4c8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b10: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.bed) [concrete]
// CHECK:STDOUT:   %.1cf: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b10, %ImplicitAs.facet.bed [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.1: type = fn_type @C.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.1: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.6e1: type = pattern_type %C.c81 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.2f0: <bound method> = bound_method %C.val.c2e, %C.as.ImplicitAs.impl.Convert.3cefb0.1 [concrete]
// CHECK:STDOUT:   %.a11: ref %C.c81 = temporary invalid, %C.val.c2e [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc7_26.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc7_26.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.f35: %pattern_type.6e1 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.8d4: %pattern_type.6e1 = wrapper_binding_pattern self, %self.param_patt.f35 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc7_26.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc7_26.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc7_26.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.3c2: <bound method> = bound_method %.a11, %Destroy.WithSelf.SelfDestruct.db3fdb.2 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8f9: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.aa2: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.ad0: %C.d0f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d0b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.98c [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.ecc: %ImplicitAs.type.164 = facet_value %C.d0f, (%ImplicitAs.impl_witness.d0b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.7c8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.ecc) [concrete]
// CHECK:STDOUT:   %.a16: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.7c8, %ImplicitAs.facet.ecc [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.2: type = fn_type @C.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.2: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a65: type = pattern_type %C.d0f [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.b0c: <bound method> = bound_method %C.val.ad0, %C.as.ImplicitAs.impl.Convert.3cefb0.2 [concrete]
// CHECK:STDOUT:   %.22f: ref %C.d0f = temporary invalid, %C.val.ad0 [concrete]
// CHECK:STDOUT:   %self.param_patt.4a5: %pattern_type.a65 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.3ff: %pattern_type.a65 = wrapper_binding_pattern self, %self.param_patt.4a5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.b54: <bound method> = bound_method %.22f, %Destroy.WithSelf.SelfDestruct.db3fdb.3 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8fb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.3e0: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.9e6: %C.d30 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a5e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.044 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.7fa: %ImplicitAs.type.164 = facet_value %C.d30, (%ImplicitAs.impl_witness.a5e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.185: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.7fa) [concrete]
// CHECK:STDOUT:   %.942: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.185, %ImplicitAs.facet.7fa [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.3: type = fn_type @C.as.ImplicitAs.impl.Convert.3 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.3: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.e2a: type = pattern_type %C.d30 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.3cb: <bound method> = bound_method %C.val.9e6, %C.as.ImplicitAs.impl.Convert.3cefb0.3 [concrete]
// CHECK:STDOUT:   %.b31: ref %C.d30 = temporary invalid, %C.val.9e6 [concrete]
// CHECK:STDOUT:   %self.param_patt.b24: %pattern_type.e2a = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c5b: %pattern_type.e2a = wrapper_binding_pattern self, %self.param_patt.b24 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.bb0: <bound method> = bound_method %.b31, %Destroy.WithSelf.SelfDestruct.db3fdb.4 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2b8: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.d29: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.51f: %C.d26 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.20c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c02 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d00: %ImplicitAs.type.164 = facet_value %C.d26, (%ImplicitAs.impl_witness.20c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e9f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.d00) [concrete]
// CHECK:STDOUT:   %.d18: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e9f, %ImplicitAs.facet.d00 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.4: type = fn_type @C.as.ImplicitAs.impl.Convert.4 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.4: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.4 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.626: type = pattern_type %C.d26 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.956: <bound method> = bound_method %C.val.51f, %C.as.ImplicitAs.impl.Convert.3cefb0.4 [concrete]
// CHECK:STDOUT:   %.0e4: ref %C.d26 = temporary invalid, %C.val.51f [concrete]
// CHECK:STDOUT:   %self.param_patt.433: %pattern_type.626 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c89: %pattern_type.626 = wrapper_binding_pattern self, %self.param_patt.433 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.4f3: <bound method> = bound_method %.0e4, %Destroy.WithSelf.SelfDestruct.db3fdb.5 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.fa9: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.a80: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.96c: %C.ddc = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6ee: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.139 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9fc: %ImplicitAs.type.164 = facet_value %C.ddc, (%ImplicitAs.impl_witness.6ee) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6b5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.9fc) [concrete]
// CHECK:STDOUT:   %.8ea: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6b5, %ImplicitAs.facet.9fc [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.5: type = fn_type @C.as.ImplicitAs.impl.Convert.5 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.5: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.5 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.fac: type = pattern_type %C.ddc [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.4fe: <bound method> = bound_method %C.val.96c, %C.as.ImplicitAs.impl.Convert.3cefb0.5 [concrete]
// CHECK:STDOUT:   %.b69: ref %C.ddc = temporary invalid, %C.val.96c [concrete]
// CHECK:STDOUT:   %self.param_patt.bed: %pattern_type.fac = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.6b9: %pattern_type.fac = wrapper_binding_pattern self, %self.param_patt.bed [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc11 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc11 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.6: type = fn_type @Destroy.WithSelf.SelfDestruct.loc11 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.6: %Destroy.WithSelf.SelfDestruct.type.fbceb5.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.e38: <bound method> = bound_method %.b69, %Destroy.WithSelf.SelfDestruct.db3fdb.6 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.669: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.019: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.26c: %C.d90 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.19f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.086 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.8b4: %ImplicitAs.type.164 = facet_value %C.d90, (%ImplicitAs.impl_witness.19f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.170: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.8b4) [concrete]
// CHECK:STDOUT:   %.cb0: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.170, %ImplicitAs.facet.8b4 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.6: type = fn_type @C.as.ImplicitAs.impl.Convert.6 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.6: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.0f2: type = pattern_type %C.d90 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.791: <bound method> = bound_method %C.val.26c, %C.as.ImplicitAs.impl.Convert.3cefb0.6 [concrete]
// CHECK:STDOUT:   %.4a6: ref %C.d90 = temporary invalid, %C.val.26c [concrete]
// CHECK:STDOUT:   %self.param_patt.281: %pattern_type.0f2 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.6c3: %pattern_type.0f2 = wrapper_binding_pattern self, %self.param_patt.281 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc12 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc12 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.7: type = fn_type @Destroy.WithSelf.SelfDestruct.loc12 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.7: %Destroy.WithSelf.SelfDestruct.type.fbceb5.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.a46: <bound method> = bound_method %.4a6, %Destroy.WithSelf.SelfDestruct.db3fdb.7 [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8f1: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.5fe: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.596: %C.a8a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5ff: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5ab [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.64b: %ImplicitAs.type.164 = facet_value %C.a8a, (%ImplicitAs.impl_witness.5ff) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.08d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.64b) [concrete]
// CHECK:STDOUT:   %.812: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.08d, %ImplicitAs.facet.64b [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.7: type = fn_type @C.as.ImplicitAs.impl.Convert.7 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.7: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.7 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.691: type = pattern_type %C.a8a [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.2ec: <bound method> = bound_method %C.val.596, %C.as.ImplicitAs.impl.Convert.3cefb0.7 [concrete]
// CHECK:STDOUT:   %.569: ref %C.a8a = temporary invalid, %C.val.596 [concrete]
// CHECK:STDOUT:   %self.param_patt.fd7: %pattern_type.691 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cd3: %pattern_type.691 = wrapper_binding_pattern self, %self.param_patt.fd7 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc13 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc13 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc13 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.103: <bound method> = bound_method %.569, %Destroy.WithSelf.SelfDestruct.db3fdb.8 [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.27f: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %bound_method.f1e: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.303: %C.1c9 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d94: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.301 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.413: %ImplicitAs.type.164 = facet_value %C.1c9, (%ImplicitAs.impl_witness.d94) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dac: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.413) [concrete]
// CHECK:STDOUT:   %.7d2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.dac, %ImplicitAs.facet.413 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.8: type = fn_type @C.as.ImplicitAs.impl.Convert.8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.8: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.8 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.d64: type = pattern_type %C.1c9 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.1cf: <bound method> = bound_method %C.val.303, %C.as.ImplicitAs.impl.Convert.3cefb0.8 [concrete]
// CHECK:STDOUT:   %.818: ref %C.1c9 = temporary invalid, %C.val.303 [concrete]
// CHECK:STDOUT:   %self.param_patt.e21: %pattern_type.d64 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c71: %pattern_type.d64 = wrapper_binding_pattern self, %self.param_patt.e21 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc14 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.9: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.loc14 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.loc14 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.406: <bound method> = bound_method %.818, %Destroy.WithSelf.SelfDestruct.db3fdb.9 [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %P: <namespace> = namespace file.%P.import, [concrete] {
// CHECK:STDOUT:     .D = %P.D
// CHECK:STDOUT:     .C = %P.C
// CHECK:STDOUT:     .Make = %P.Make
// CHECK:STDOUT:     import P//library
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %P.D: type = import_ref P//library, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %P.import_ref.528: <witness> = import_ref P//library, loc5_35, loaded [concrete = constants.%complete_type.a39]
// CHECK:STDOUT:   %P.import_ref.737 = import_ref P//library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.0d7 = import_ref P//library, loc5_16, unloaded
// CHECK:STDOUT:   %P.import_ref.105 = import_ref P//library, loc5_28, unloaded
// CHECK:STDOUT:   %P.C: %C.type = import_ref P//library, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %P.import_ref.8f2: <witness> = import_ref P//library, loc4_18, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %P.import_ref.0f2 = import_ref P//library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.2c7: %i32 = import_ref P//library, loc4_10, loaded [symbolic = @C.%N (constants.%N.5d3)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %P.import_ref.5ed: <witness> = import_ref P//library, loc8_33, loaded [concrete = constants.%ImplicitAs.impl_witness.4c8]
// CHECK:STDOUT:   %P.import_ref.191: type = import_ref P//library, loc8_9, loaded [concrete = constants.%C.c81]
// CHECK:STDOUT:   %P.import_ref.0d5cf9.1: type = import_ref P//library, loc8_31, loaded [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   %P.import_ref.470: <witness> = import_ref P//library, loc9_33, loaded [concrete = constants.%ImplicitAs.impl_witness.d0b]
// CHECK:STDOUT:   %P.import_ref.133: type = import_ref P//library, loc9_9, loaded [concrete = constants.%C.d0f]
// CHECK:STDOUT:   %P.import_ref.0d5cf9.2: type = import_ref P//library, loc9_31, loaded [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   %P.import_ref.158: <witness> = import_ref P//library, loc10_33, loaded [concrete = constants.%ImplicitAs.impl_witness.a5e]
// CHECK:STDOUT:   %P.import_ref.2e4: type = import_ref P//library, loc10_9, loaded [concrete = constants.%C.d30]
// CHECK:STDOUT:   %P.import_ref.0d5cf9.3: type = import_ref P//library, loc10_31, loaded [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   %P.import_ref.866: <witness> = import_ref P//library, loc11_33, loaded [concrete = constants.%ImplicitAs.impl_witness.20c]
// CHECK:STDOUT:   %P.import_ref.d0f: type = import_ref P//library, loc11_9, loaded [concrete = constants.%C.d26]
// CHECK:STDOUT:   %P.import_ref.0d5cf9.4: type = import_ref P//library, loc11_31, loaded [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   %P.import_ref.4db: <witness> = import_ref P//library, loc12_33, loaded [concrete = constants.%ImplicitAs.impl_witness.6ee]
// CHECK:STDOUT:   %P.import_ref.bd1: type = import_ref P//library, loc12_9, loaded [concrete = constants.%C.ddc]
// CHECK:STDOUT:   %P.import_ref.0d5cf9.5: type = import_ref P//library, loc12_31, loaded [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   %P.import_ref.91d: <witness> = import_ref P//library, loc13_33, loaded [concrete = constants.%ImplicitAs.impl_witness.19f]
// CHECK:STDOUT:   %P.import_ref.f5c: type = import_ref P//library, loc13_9, loaded [concrete = constants.%C.d90]
// CHECK:STDOUT:   %P.import_ref.0d5cf9.6: type = import_ref P//library, loc13_31, loaded [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   %P.import_ref.61b: <witness> = import_ref P//library, loc14_33, loaded [concrete = constants.%ImplicitAs.impl_witness.5ff]
// CHECK:STDOUT:   %P.import_ref.cd4: type = import_ref P//library, loc14_9, loaded [concrete = constants.%C.a8a]
// CHECK:STDOUT:   %P.import_ref.0d5cf9.7: type = import_ref P//library, loc14_31, loaded [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   %P.import_ref.702: <witness> = import_ref P//library, loc15_33, loaded [concrete = constants.%ImplicitAs.impl_witness.d94]
// CHECK:STDOUT:   %P.import_ref.fa9: type = import_ref P//library, loc15_9, loaded [concrete = constants.%C.1c9]
// CHECK:STDOUT:   %P.import_ref.0d5cf9.8: type = import_ref P//library, loc15_31, loaded [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   %P.import_ref.b89e2b.1: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = import_ref P//library, loc8_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f50 = impl_witness_table (%P.import_ref.b89e2b.1), @C.as.ImplicitAs.impl.e4c [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %P.import_ref.b89e2b.2: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = import_ref P//library, loc9_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.98c = impl_witness_table (%P.import_ref.b89e2b.2), @C.as.ImplicitAs.impl.cc9 [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.3: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.3 = import_ref P//library, loc10_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.3]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.044 = impl_witness_table (%P.import_ref.b89e2b.3), @C.as.ImplicitAs.impl.71c [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.4: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.4 = import_ref P//library, loc11_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.4]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c02 = impl_witness_table (%P.import_ref.b89e2b.4), @C.as.ImplicitAs.impl.50b [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.5: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.5 = import_ref P//library, loc12_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.5]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.139 = impl_witness_table (%P.import_ref.b89e2b.5), @C.as.ImplicitAs.impl.fd9 [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.6: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.6 = import_ref P//library, loc13_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.6]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.086 = impl_witness_table (%P.import_ref.b89e2b.6), @C.as.ImplicitAs.impl.49b [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.7: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.7 = import_ref P//library, loc14_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.7]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5ab = impl_witness_table (%P.import_ref.b89e2b.7), @C.as.ImplicitAs.impl.1b1 [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.8: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.8 = import_ref P//library, loc15_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.8]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.301 = impl_witness_table (%P.import_ref.b89e2b.8), @C.as.ImplicitAs.impl.049 [concrete]
// CHECK:STDOUT:   %P.Make: %Make.type = import_ref P//library, Make, loaded [concrete = constants.%Make]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc7_26.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc7_26.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc7_26.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6e1 = ref_param_pattern [concrete = constants.%self.param_patt.f35]
// CHECK:STDOUT:     %self.patt: %pattern_type.6e1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.8d4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.c81 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.c81 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc7_26.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6e1 = ref_param_pattern [concrete = constants.%self.param_patt.f35]
// CHECK:STDOUT:     %self.patt: %pattern_type.6e1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.8d4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.c81 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.c81 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a65 = ref_param_pattern [concrete = constants.%self.param_patt.4a5]
// CHECK:STDOUT:     %self.patt: %pattern_type.a65 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.3ff]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.d0f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.d0f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a65 = ref_param_pattern [concrete = constants.%self.param_patt.4a5]
// CHECK:STDOUT:     %self.patt: %pattern_type.a65 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.3ff]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.d0f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.d0f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e2a = ref_param_pattern [concrete = constants.%self.param_patt.b24]
// CHECK:STDOUT:     %self.patt: %pattern_type.e2a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c5b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.d30 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.d30 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc9 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e2a = ref_param_pattern [concrete = constants.%self.param_patt.b24]
// CHECK:STDOUT:     %self.patt: %pattern_type.e2a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c5b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.d30 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.d30 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.626 = ref_param_pattern [concrete = constants.%self.param_patt.433]
// CHECK:STDOUT:     %self.patt: %pattern_type.626 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c89]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.d26 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.d26 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc10 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.626 = ref_param_pattern [concrete = constants.%self.param_patt.433]
// CHECK:STDOUT:     %self.patt: %pattern_type.626 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c89]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.d26 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.d26 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc11 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fac = ref_param_pattern [concrete = constants.%self.param_patt.bed]
// CHECK:STDOUT:     %self.patt: %pattern_type.fac = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.6b9]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.ddc = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.ddc = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc11 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fac = ref_param_pattern [concrete = constants.%self.param_patt.bed]
// CHECK:STDOUT:     %self.patt: %pattern_type.fac = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.6b9]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.ddc = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.ddc = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc12 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0f2 = ref_param_pattern [concrete = constants.%self.param_patt.281]
// CHECK:STDOUT:     %self.patt: %pattern_type.0f2 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.6c3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.d90 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.d90 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc12 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0f2 = ref_param_pattern [concrete = constants.%self.param_patt.281]
// CHECK:STDOUT:     %self.patt: %pattern_type.0f2 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.6c3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.d90 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.d90 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc13 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.691 = ref_param_pattern [concrete = constants.%self.param_patt.fd7]
// CHECK:STDOUT:     %self.patt: %pattern_type.691 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cd3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.a8a = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.a8a = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.8: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc13 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.691 = ref_param_pattern [concrete = constants.%self.param_patt.fd7]
// CHECK:STDOUT:     %self.patt: %pattern_type.691 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cd3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.a8a = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.a8a = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.9: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc14 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.9] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d64 = ref_param_pattern [concrete = constants.%self.param_patt.e21]
// CHECK:STDOUT:     %self.patt: %pattern_type.d64 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c71]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.1c9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.1c9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.9: %Destroy.WithSelf.Op.type.ef016f.9 = fn_decl @Destroy.WithSelf.Op.loc14 [concrete = constants.%Destroy.WithSelf.Op.403171.9] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d64 = ref_param_pattern [concrete = constants.%self.param_patt.e21]
// CHECK:STDOUT:     %self.patt: %pattern_type.d64 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c71]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.1c9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.1c9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .P = imports.%P
// CHECK:STDOUT:     .F0 = %F0.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %P.import = import P
// CHECK:STDOUT:   %F0.decl: %F0.type = fn_decl @F0 [concrete = constants.%F0] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.9d9 = value_param_pattern [concrete = constants.%n.param_patt]
// CHECK:STDOUT:     %n.patt: %pattern_type.9d9 = wrapper_binding_pattern n, %n.param_patt [concrete = constants.%n.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %P.ref.loc6: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%P.D [concrete = constants.%D]
// CHECK:STDOUT:     %.loc6_26.10: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = wrapper_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.e4c: imports.%P.import_ref.191 as imports.%P.import_ref.0d5cf9.1 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.5ed
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.cc9: imports.%P.import_ref.133 as imports.%P.import_ref.0d5cf9.2 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.470
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.71c: imports.%P.import_ref.2e4 as imports.%P.import_ref.0d5cf9.3 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.50b: imports.%P.import_ref.d0f as imports.%P.import_ref.0d5cf9.4 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.866
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.fd9: imports.%P.import_ref.bd1 as imports.%P.import_ref.0d5cf9.5 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.4db
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.49b: imports.%P.import_ref.f5c as imports.%P.import_ref.0d5cf9.6 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.91d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.1b1: imports.%P.import_ref.cd4 as imports.%P.import_ref.0d5cf9.7 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.61b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.049: imports.%P.import_ref.fa9 as imports.%P.import_ref.0d5cf9.8 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.702
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "library.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%P.import_ref.528
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%P.import_ref.737
// CHECK:STDOUT:   .n = imports.%P.import_ref.0d7
// CHECK:STDOUT:   .m = imports.%P.import_ref.105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%P.import_ref.2c7: %i32) [from "library.carbon"] {
// CHECK:STDOUT:   %N.patt: %pattern_type.9d9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.b2d)]
// CHECK:STDOUT:   %N: %i32 = symbolic_binding N, 0 [symbolic = %N (constants.%N.5d3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%P.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%P.import_ref.0f2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F0(%n.param: %i32) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %false.loc7: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc7 br !if.then.loc7 else br !if.else.loc7
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc7:
// CHECK:STDOUT:   %.loc7_24.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc7: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc7: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc7_34: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc7_34.1: <bound method> = bound_method %int_0, %impl.elem0.loc7_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.935]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_34.2: <bound method> = bound_method %int_0, %specific_fn.loc7 [concrete = constants.%bound_method.614]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_34.2(%int_0) [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:   %.loc7_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:   %.loc7_34.2: %i32 = converted %int_0, %.loc7_34.1 [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:   %C.loc7: type = class_type @C, @C(constants.%int_0.2b2) [concrete = constants.%C.c81]
// CHECK:STDOUT:   %.loc7_24.2: ref %C.c81 = temporary_storage
// CHECK:STDOUT:   %.loc7_24.3: init %C.c81 to %.loc7_24.2 = class_init () [concrete = constants.%C.val.c2e]
// CHECK:STDOUT:   %.loc7_26.1: init %C.c81 = converted %.loc7_24.1, %.loc7_24.3 [concrete = constants.%C.val.c2e]
// CHECK:STDOUT:   %impl.elem0.loc7_35: %.1cf = impl_witness_access constants.%ImplicitAs.impl_witness.4c8, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %bound_method.loc7_35: <bound method> = bound_method %.loc7_26.1, %impl.elem0.loc7_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.2f0]
// CHECK:STDOUT:   %.loc6_26.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc7_26.2: ref %C.c81 = temporary %.loc7_24.2, %.loc7_26.1 [concrete = constants.%.a11]
// CHECK:STDOUT:   %.loc7_26.3: %C.c81 = acquire_value %.loc7_26.2 [concrete = constants.%C.val.c2e]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc7: init %D to %.loc6_26.1 = call %bound_method.loc7_35(%.loc7_26.3)
// CHECK:STDOUT:   %.loc7_35: init %D = converted %.loc7_26.1, %C.as.ImplicitAs.impl.Convert.call.loc7
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc7: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.3c2(constants.%.a11)
// CHECK:STDOUT:   return %.loc7_35 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc7:
// CHECK:STDOUT:   %false.loc8: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc8 br !if.then.loc8 else br !if.else.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc8:
// CHECK:STDOUT:   %.loc8_24.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc8: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc8: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8_34: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8f9]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_1, %specific_fn.loc8 [concrete = constants.%bound_method.aa2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_34.2(%int_1) [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_1, %.loc8_34.1 [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:   %C.loc8: type = class_type @C, @C(constants.%int_1.8e8) [concrete = constants.%C.d0f]
// CHECK:STDOUT:   %.loc8_24.2: ref %C.d0f = temporary_storage
// CHECK:STDOUT:   %.loc8_24.3: init %C.d0f to %.loc8_24.2 = class_init () [concrete = constants.%C.val.ad0]
// CHECK:STDOUT:   %.loc8_26.1: init %C.d0f = converted %.loc8_24.1, %.loc8_24.3 [concrete = constants.%C.val.ad0]
// CHECK:STDOUT:   %impl.elem0.loc8_35: %.a16 = impl_witness_access constants.%ImplicitAs.impl_witness.d0b, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:   %bound_method.loc8_35: <bound method> = bound_method %.loc8_26.1, %impl.elem0.loc8_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.b0c]
// CHECK:STDOUT:   %.loc6_26.2: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc8_26.2: ref %C.d0f = temporary %.loc8_24.2, %.loc8_26.1 [concrete = constants.%.22f]
// CHECK:STDOUT:   %.loc8_26.3: %C.d0f = acquire_value %.loc8_26.2 [concrete = constants.%C.val.ad0]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc8: init %D to %.loc6_26.2 = call %bound_method.loc8_35(%.loc8_26.3)
// CHECK:STDOUT:   %.loc8_35: init %D = converted %.loc8_26.1, %C.as.ImplicitAs.impl.Convert.call.loc8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc8: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.b54(constants.%.22f)
// CHECK:STDOUT:   return %.loc8_35 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc8:
// CHECK:STDOUT:   %false.loc9: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc9 br !if.then.loc9 else br !if.else.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc9:
// CHECK:STDOUT:   %.loc9_24.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc9: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc9: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc9_34: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %int_2, %impl.elem0.loc9_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8fb]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %int_2, %specific_fn.loc9 [concrete = constants.%bound_method.3e0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_34.2(%int_2) [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc9_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc9_34.2: %i32 = converted %int_2, %.loc9_34.1 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %C.loc9: type = class_type @C, @C(constants.%int_2.cb7) [concrete = constants.%C.d30]
// CHECK:STDOUT:   %.loc9_24.2: ref %C.d30 = temporary_storage
// CHECK:STDOUT:   %.loc9_24.3: init %C.d30 to %.loc9_24.2 = class_init () [concrete = constants.%C.val.9e6]
// CHECK:STDOUT:   %.loc9_26.1: init %C.d30 = converted %.loc9_24.1, %.loc9_24.3 [concrete = constants.%C.val.9e6]
// CHECK:STDOUT:   %impl.elem0.loc9_35: %.942 = impl_witness_access constants.%ImplicitAs.impl_witness.a5e, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.3]
// CHECK:STDOUT:   %bound_method.loc9_35: <bound method> = bound_method %.loc9_26.1, %impl.elem0.loc9_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.3cb]
// CHECK:STDOUT:   %.loc6_26.3: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc9_26.2: ref %C.d30 = temporary %.loc9_24.2, %.loc9_26.1 [concrete = constants.%.b31]
// CHECK:STDOUT:   %.loc9_26.3: %C.d30 = acquire_value %.loc9_26.2 [concrete = constants.%C.val.9e6]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc9: init %D to %.loc6_26.3 = call %bound_method.loc9_35(%.loc9_26.3)
// CHECK:STDOUT:   %.loc9_35: init %D = converted %.loc9_26.1, %C.as.ImplicitAs.impl.Convert.call.loc9
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.bb0(constants.%.b31)
// CHECK:STDOUT:   return %.loc9_35 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc9:
// CHECK:STDOUT:   %false.loc10: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc10 br !if.then.loc10 else br !if.else.loc10
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc10:
// CHECK:STDOUT:   %.loc10_24.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc10: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc10: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc10_34: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc10_34.1: <bound method> = bound_method %int_3, %impl.elem0.loc10_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2b8]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_34.2: <bound method> = bound_method %int_3, %specific_fn.loc10 [concrete = constants.%bound_method.d29]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %i32 = call %bound_method.loc10_34.2(%int_3) [concrete = constants.%int_3.af5]
// CHECK:STDOUT:   %.loc10_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_3.af5]
// CHECK:STDOUT:   %.loc10_34.2: %i32 = converted %int_3, %.loc10_34.1 [concrete = constants.%int_3.af5]
// CHECK:STDOUT:   %C.loc10: type = class_type @C, @C(constants.%int_3.af5) [concrete = constants.%C.d26]
// CHECK:STDOUT:   %.loc10_24.2: ref %C.d26 = temporary_storage
// CHECK:STDOUT:   %.loc10_24.3: init %C.d26 to %.loc10_24.2 = class_init () [concrete = constants.%C.val.51f]
// CHECK:STDOUT:   %.loc10_26.1: init %C.d26 = converted %.loc10_24.1, %.loc10_24.3 [concrete = constants.%C.val.51f]
// CHECK:STDOUT:   %impl.elem0.loc10_35: %.d18 = impl_witness_access constants.%ImplicitAs.impl_witness.20c, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.4]
// CHECK:STDOUT:   %bound_method.loc10_35: <bound method> = bound_method %.loc10_26.1, %impl.elem0.loc10_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.956]
// CHECK:STDOUT:   %.loc6_26.4: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc10_26.2: ref %C.d26 = temporary %.loc10_24.2, %.loc10_26.1 [concrete = constants.%.0e4]
// CHECK:STDOUT:   %.loc10_26.3: %C.d26 = acquire_value %.loc10_26.2 [concrete = constants.%C.val.51f]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc10: init %D to %.loc6_26.4 = call %bound_method.loc10_35(%.loc10_26.3)
// CHECK:STDOUT:   %.loc10_35: init %D = converted %.loc10_26.1, %C.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.4f3(constants.%.0e4)
// CHECK:STDOUT:   return %.loc10_35 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc10:
// CHECK:STDOUT:   %false.loc11: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc11 br !if.then.loc11 else br !if.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11:
// CHECK:STDOUT:   %.loc11_24.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc11: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc11: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc11_34: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc11_34.1: <bound method> = bound_method %int_4, %impl.elem0.loc11_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.fa9]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_34.2: <bound method> = bound_method %int_4, %specific_fn.loc11 [concrete = constants.%bound_method.a80]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %i32 = call %bound_method.loc11_34.2(%int_4) [concrete = constants.%int_4.380]
// CHECK:STDOUT:   %.loc11_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_4.380]
// CHECK:STDOUT:   %.loc11_34.2: %i32 = converted %int_4, %.loc11_34.1 [concrete = constants.%int_4.380]
// CHECK:STDOUT:   %C.loc11: type = class_type @C, @C(constants.%int_4.380) [concrete = constants.%C.ddc]
// CHECK:STDOUT:   %.loc11_24.2: ref %C.ddc = temporary_storage
// CHECK:STDOUT:   %.loc11_24.3: init %C.ddc to %.loc11_24.2 = class_init () [concrete = constants.%C.val.96c]
// CHECK:STDOUT:   %.loc11_26.1: init %C.ddc = converted %.loc11_24.1, %.loc11_24.3 [concrete = constants.%C.val.96c]
// CHECK:STDOUT:   %impl.elem0.loc11_35: %.8ea = impl_witness_access constants.%ImplicitAs.impl_witness.6ee, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.5]
// CHECK:STDOUT:   %bound_method.loc11_35: <bound method> = bound_method %.loc11_26.1, %impl.elem0.loc11_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.4fe]
// CHECK:STDOUT:   %.loc6_26.5: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc11_26.2: ref %C.ddc = temporary %.loc11_24.2, %.loc11_26.1 [concrete = constants.%.b69]
// CHECK:STDOUT:   %.loc11_26.3: %C.ddc = acquire_value %.loc11_26.2 [concrete = constants.%C.val.96c]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc11: init %D to %.loc6_26.5 = call %bound_method.loc11_35(%.loc11_26.3)
// CHECK:STDOUT:   %.loc11_35: init %D = converted %.loc11_26.1, %C.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc11: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.e38(constants.%.b69)
// CHECK:STDOUT:   return %.loc11_35 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11:
// CHECK:STDOUT:   %false.loc12: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc12 br !if.then.loc12 else br !if.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc12:
// CHECK:STDOUT:   %.loc12_24.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc12: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc12: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc12_34: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc12_34.1: <bound method> = bound_method %int_5, %impl.elem0.loc12_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.669]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_34.2: <bound method> = bound_method %int_5, %specific_fn.loc12 [concrete = constants.%bound_method.019]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %i32 = call %bound_method.loc12_34.2(%int_5) [concrete = constants.%int_5.c30]
// CHECK:STDOUT:   %.loc12_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_5.c30]
// CHECK:STDOUT:   %.loc12_34.2: %i32 = converted %int_5, %.loc12_34.1 [concrete = constants.%int_5.c30]
// CHECK:STDOUT:   %C.loc12: type = class_type @C, @C(constants.%int_5.c30) [concrete = constants.%C.d90]
// CHECK:STDOUT:   %.loc12_24.2: ref %C.d90 = temporary_storage
// CHECK:STDOUT:   %.loc12_24.3: init %C.d90 to %.loc12_24.2 = class_init () [concrete = constants.%C.val.26c]
// CHECK:STDOUT:   %.loc12_26.1: init %C.d90 = converted %.loc12_24.1, %.loc12_24.3 [concrete = constants.%C.val.26c]
// CHECK:STDOUT:   %impl.elem0.loc12_35: %.cb0 = impl_witness_access constants.%ImplicitAs.impl_witness.19f, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.6]
// CHECK:STDOUT:   %bound_method.loc12_35: <bound method> = bound_method %.loc12_26.1, %impl.elem0.loc12_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.791]
// CHECK:STDOUT:   %.loc6_26.6: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc12_26.2: ref %C.d90 = temporary %.loc12_24.2, %.loc12_26.1 [concrete = constants.%.4a6]
// CHECK:STDOUT:   %.loc12_26.3: %C.d90 = acquire_value %.loc12_26.2 [concrete = constants.%C.val.26c]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc12: init %D to %.loc6_26.6 = call %bound_method.loc12_35(%.loc12_26.3)
// CHECK:STDOUT:   %.loc12_35: init %D = converted %.loc12_26.1, %C.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc12: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.a46(constants.%.4a6)
// CHECK:STDOUT:   return %.loc12_35 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc12:
// CHECK:STDOUT:   %false.loc13: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc13 br !if.then.loc13 else br !if.else.loc13
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc13:
// CHECK:STDOUT:   %.loc13_24.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc13: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc13: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:   %impl.elem0.loc13_34: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc13_34.1: <bound method> = bound_method %int_6, %impl.elem0.loc13_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8f1]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_34.2: <bound method> = bound_method %int_6, %specific_fn.loc13 [concrete = constants.%bound_method.5fe]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %i32 = call %bound_method.loc13_34.2(%int_6) [concrete = constants.%int_6.b5e]
// CHECK:STDOUT:   %.loc13_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%int_6.b5e]
// CHECK:STDOUT:   %.loc13_34.2: %i32 = converted %int_6, %.loc13_34.1 [concrete = constants.%int_6.b5e]
// CHECK:STDOUT:   %C.loc13: type = class_type @C, @C(constants.%int_6.b5e) [concrete = constants.%C.a8a]
// CHECK:STDOUT:   %.loc13_24.2: ref %C.a8a = temporary_storage
// CHECK:STDOUT:   %.loc13_24.3: init %C.a8a to %.loc13_24.2 = class_init () [concrete = constants.%C.val.596]
// CHECK:STDOUT:   %.loc13_26.1: init %C.a8a = converted %.loc13_24.1, %.loc13_24.3 [concrete = constants.%C.val.596]
// CHECK:STDOUT:   %impl.elem0.loc13_35: %.812 = impl_witness_access constants.%ImplicitAs.impl_witness.5ff, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.7]
// CHECK:STDOUT:   %bound_method.loc13_35: <bound method> = bound_method %.loc13_26.1, %impl.elem0.loc13_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.2ec]
// CHECK:STDOUT:   %.loc6_26.7: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc13_26.2: ref %C.a8a = temporary %.loc13_24.2, %.loc13_26.1 [concrete = constants.%.569]
// CHECK:STDOUT:   %.loc13_26.3: %C.a8a = acquire_value %.loc13_26.2 [concrete = constants.%C.val.596]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc13: init %D to %.loc6_26.7 = call %bound_method.loc13_35(%.loc13_26.3)
// CHECK:STDOUT:   %.loc13_35: init %D = converted %.loc13_26.1, %C.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc13: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.103(constants.%.569)
// CHECK:STDOUT:   return %.loc13_35 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc13:
// CHECK:STDOUT:   %false.loc14: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc14 br !if.then.loc14 else br !if.else.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc14:
// CHECK:STDOUT:   %.loc14_24.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc14: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc14: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:   %impl.elem0.loc14_34: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc14_34.1: <bound method> = bound_method %int_7, %impl.elem0.loc14_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.27f]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.2: <bound method> = bound_method %int_7, %specific_fn.loc14 [concrete = constants.%bound_method.f1e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14: init %i32 = call %bound_method.loc14_34.2(%int_7) [concrete = constants.%int_7.b03]
// CHECK:STDOUT:   %.loc14_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14 [concrete = constants.%int_7.b03]
// CHECK:STDOUT:   %.loc14_34.2: %i32 = converted %int_7, %.loc14_34.1 [concrete = constants.%int_7.b03]
// CHECK:STDOUT:   %C.loc14: type = class_type @C, @C(constants.%int_7.b03) [concrete = constants.%C.1c9]
// CHECK:STDOUT:   %.loc14_24.2: ref %C.1c9 = temporary_storage
// CHECK:STDOUT:   %.loc14_24.3: init %C.1c9 to %.loc14_24.2 = class_init () [concrete = constants.%C.val.303]
// CHECK:STDOUT:   %.loc14_26.1: init %C.1c9 = converted %.loc14_24.1, %.loc14_24.3 [concrete = constants.%C.val.303]
// CHECK:STDOUT:   %impl.elem0.loc14_35: %.7d2 = impl_witness_access constants.%ImplicitAs.impl_witness.d94, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.8]
// CHECK:STDOUT:   %bound_method.loc14_35: <bound method> = bound_method %.loc14_26.1, %impl.elem0.loc14_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.1cf]
// CHECK:STDOUT:   %.loc6_26.8: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc14_26.2: ref %C.1c9 = temporary %.loc14_24.2, %.loc14_26.1 [concrete = constants.%.818]
// CHECK:STDOUT:   %.loc14_26.3: %C.1c9 = acquire_value %.loc14_26.2 [concrete = constants.%C.val.303]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc14: init %D to %.loc6_26.8 = call %bound_method.loc14_35(%.loc14_26.3)
// CHECK:STDOUT:   %.loc14_35: init %D = converted %.loc14_26.1, %C.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc14: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.406(constants.%.818)
// CHECK:STDOUT:   return %.loc14_35 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc14:
// CHECK:STDOUT:   %P.ref.loc15: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%P.Make [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc6_26.9: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc6_26.9 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.1 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc7_26.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc7_26.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc7_26.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc7_26.2(%self.param: ref %C.c81) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc7_26.2(%self.param: ref %C.c81) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc7_26.2(%self.param: ref %C.c81) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.2 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %C.d0f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %C.d0f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %C.d0f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.3 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9(%self.param: ref %C.d30) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9(%self.param: ref %C.d30) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9(%self.param: ref %C.d30) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.4 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10(%self.param: ref %C.d26) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10(%self.param: ref %C.d26) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10(%self.param: ref %C.d26) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.5 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc11(%self.param: ref %C.ddc) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc11(%self.param: ref %C.ddc) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc11(%self.param: ref %C.ddc) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.6 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc12(%self.param: ref %C.d90) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc12(%self.param: ref %C.d90) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc12(%self.param: ref %C.d90) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.7(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.7(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.7 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc13(%self.param: ref %C.a8a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc13(%self.param: ref %C.a8a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc13(%self.param: ref %C.a8a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.8(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.8(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.8 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc14(%self.param: ref %C.1c9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc14(%self.param: ref %C.1c9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc14(%self.param: ref %C.1c9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.9(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.9(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.5d3) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.b2d
// CHECK:STDOUT:   %N => constants.%N.5d3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.2b2) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.b2d
// CHECK:STDOUT:   %N => constants.%int_0.2b2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.8e8) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.b2d
// CHECK:STDOUT:   %N => constants.%int_1.8e8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.cb7) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.b2d
// CHECK:STDOUT:   %N => constants.%int_2.cb7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.af5) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.b2d
// CHECK:STDOUT:   %N => constants.%int_3.af5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.380) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.b2d
// CHECK:STDOUT:   %N => constants.%int_4.380
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.c30) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.b2d
// CHECK:STDOUT:   %N => constants.%int_5.c30
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.b5e) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.b2d
// CHECK:STDOUT:   %N => constants.%int_6.b5e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.b03) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.b2d
// CHECK:STDOUT:   %N => constants.%int_7.b03
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
