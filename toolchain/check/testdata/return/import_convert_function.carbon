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

class C(N:! i32) {}
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.38a: %pattern_type.6b6 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.37f: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.bf4: type = class_type @C, @C(%N.37f) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.fa1: type = struct_type {.n: %i32, .m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.237: <witness> = complete_type_witness %struct_type.n.m.fa1 [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9f1: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.9f1) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet.9f1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.6e0: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_0.3c0, %int_0.3c0) [concrete]
// CHECK:STDOUT:   %C.771: type = class_type @C, @C(%int_0.3c0) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.24d: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4c1: <witness> = impl_witness @C.as.ImplicitAs.impl.0e1.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.84d: type = pattern_type %C.771 [concrete]
// CHECK:STDOUT:   %self.param_patt.98b: %pattern_type.84d = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.975: %pattern_type.84d = at_binding_pattern self, %self.param_patt.98b [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.1: type = fn_type @C.as.ImplicitAs.impl.Convert.loc8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.1: %C.as.ImplicitAs.impl.Convert.type.860b0a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.83b: type = class_type @C, @C(%int_1.0c6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c46: <witness> = impl_witness @C.as.ImplicitAs.impl.f6a.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.91a: type = pattern_type %C.83b [concrete]
// CHECK:STDOUT:   %self.param_patt.327: %pattern_type.91a = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.533: %pattern_type.91a = at_binding_pattern self, %self.param_patt.327 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.2: type = fn_type @C.as.ImplicitAs.impl.Convert.loc9 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.2: %C.as.ImplicitAs.impl.Convert.type.860b0a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.5ce: type = class_type @C, @C(%int_2.295) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.430: <witness> = impl_witness @C.as.ImplicitAs.impl.aab.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7dd: type = pattern_type %C.5ce [concrete]
// CHECK:STDOUT:   %self.param_patt.e14: %pattern_type.7dd = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.426: %pattern_type.7dd = at_binding_pattern self, %self.param_patt.e14 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.3: type = fn_type @C.as.ImplicitAs.impl.Convert.loc10 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.3: %C.as.ImplicitAs.impl.Convert.type.860b0a.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.763: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.410: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.b3e: type = class_type @C, @C(%int_3.410) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.b6d: <witness> = impl_witness @C.as.ImplicitAs.impl.203.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.26a: type = pattern_type %C.b3e [concrete]
// CHECK:STDOUT:   %self.param_patt.0fc: %pattern_type.26a = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.450: %pattern_type.26a = at_binding_pattern self, %self.param_patt.0fc [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.4: type = fn_type @C.as.ImplicitAs.impl.Convert.loc11 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.4: %C.as.ImplicitAs.impl.Convert.type.860b0a.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aad: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.b7a: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.4eb: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %C.8ee: type = class_type @C, @C(%int_4.4eb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c85: <witness> = impl_witness @C.as.ImplicitAs.impl.371.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.387: type = pattern_type %C.8ee [concrete]
// CHECK:STDOUT:   %self.param_patt.826: %pattern_type.387 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.401: %pattern_type.387 = at_binding_pattern self, %self.param_patt.826 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.5: type = fn_type @C.as.ImplicitAs.impl.Convert.loc12 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.5: %C.as.ImplicitAs.impl.Convert.type.860b0a.5 = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f6c: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.c9d: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.eef: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %C.5c1: type = class_type @C, @C(%int_5.eef) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1af: <witness> = impl_witness @C.as.ImplicitAs.impl.c73.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.05e: type = pattern_type %C.5c1 [concrete]
// CHECK:STDOUT:   %self.param_patt.f7b: %pattern_type.05e = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.283: %pattern_type.05e = at_binding_pattern self, %self.param_patt.f7b [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.6: type = fn_type @C.as.ImplicitAs.impl.Convert.loc13 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.6: %C.as.ImplicitAs.impl.Convert.type.860b0a.6 = struct_value () [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.13d: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.be9: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.439: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %C.57b: type = class_type @C, @C(%int_6.439) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c9c: <witness> = impl_witness @C.as.ImplicitAs.impl.2f3.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.4b0: type = pattern_type %C.57b [concrete]
// CHECK:STDOUT:   %self.param_patt.e33: %pattern_type.4b0 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.7b4: %pattern_type.4b0 = at_binding_pattern self, %self.param_patt.e33 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.7: type = fn_type @C.as.ImplicitAs.impl.Convert.loc14 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.7: %C.as.ImplicitAs.impl.Convert.type.860b0a.7 = struct_value () [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.66b: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.7ff: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.690: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %C.b0a: type = class_type @C, @C(%int_7.690) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.642: <witness> = impl_witness @C.as.ImplicitAs.impl.28a.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.bc2: type = pattern_type %C.b0a [concrete]
// CHECK:STDOUT:   %self.param_patt.21a: %pattern_type.bc2 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.94c: %pattern_type.bc2 = at_binding_pattern self, %self.param_patt.21a [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %N.patt.loc4_10.1: %pattern_type.6b6 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_10.2 (constants.%N.patt.38a)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_10.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_10.1 (constants.%N.37f)]
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
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.0e1 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc8_9.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_9.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.6e0]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_9.2(%int_0) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:     %.loc8_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:     %.loc8_9.2: %i32 = converted %int_0, %.loc8_9.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_0.3c0) [concrete = constants.%C.771]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.f6a [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %impl.elem0: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc9_9.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_9.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method.953]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_9.2(%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:     %.loc9_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:     %.loc9_9.2: %i32 = converted %int_1, %.loc9_9.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_1.0c6) [concrete = constants.%C.83b]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.aab [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc10_9.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_9.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc10_9.2(%int_2) [concrete = constants.%int_2.295]
// CHECK:STDOUT:     %.loc10_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.295]
// CHECK:STDOUT:     %.loc10_9.2: %i32 = converted %int_2, %.loc10_9.1 [concrete = constants.%int_2.295]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_2.295) [concrete = constants.%C.5ce]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.203 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc11_9.1: <bound method> = bound_method %int_3, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_9.2: <bound method> = bound_method %int_3, %specific_fn [concrete = constants.%bound_method.763]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_9.2(%int_3) [concrete = constants.%int_3.410]
// CHECK:STDOUT:     %.loc11_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.410]
// CHECK:STDOUT:     %.loc11_9.2: %i32 = converted %int_3, %.loc11_9.1 [concrete = constants.%int_3.410]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_3.410) [concrete = constants.%C.b3e]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.371 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %impl.elem0: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc12_9.1: <bound method> = bound_method %int_4, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aad]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc12_9.2: <bound method> = bound_method %int_4, %specific_fn [concrete = constants.%bound_method.b7a]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_9.2(%int_4) [concrete = constants.%int_4.4eb]
// CHECK:STDOUT:     %.loc12_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_4.4eb]
// CHECK:STDOUT:     %.loc12_9.2: %i32 = converted %int_4, %.loc12_9.1 [concrete = constants.%int_4.4eb]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_4.4eb) [concrete = constants.%C.8ee]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.c73 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:     %impl.elem0: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc13_9.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f6c]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc13_9.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method.c9d]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_9.2(%int_5) [concrete = constants.%int_5.eef]
// CHECK:STDOUT:     %.loc13_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.eef]
// CHECK:STDOUT:     %.loc13_9.2: %i32 = converted %int_5, %.loc13_9.1 [concrete = constants.%int_5.eef]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_5.eef) [concrete = constants.%C.5c1]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.2f3 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:     %impl.elem0: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %int_6, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.13d]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %int_6, %specific_fn [concrete = constants.%bound_method.be9]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc14_9.2(%int_6) [concrete = constants.%int_6.439]
// CHECK:STDOUT:     %.loc14_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_6.439]
// CHECK:STDOUT:     %.loc14_9.2: %i32 = converted %int_6, %.loc14_9.1 [concrete = constants.%int_6.439]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_6.439) [concrete = constants.%C.57b]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.28a [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:     %impl.elem0: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:     %bound_method.loc15_9.1: <bound method> = bound_method %int_7, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.66b]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc15_9.2: <bound method> = bound_method %int_7, %specific_fn [concrete = constants.%bound_method.7ff]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_9.2(%int_7) [concrete = constants.%int_7.690]
// CHECK:STDOUT:     %.loc15_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_7.690]
// CHECK:STDOUT:     %.loc15_9.2: %i32 = converted %int_7, %.loc15_9.1 [concrete = constants.%int_7.690]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_7.690) [concrete = constants.%C.b0a]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.0e1: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.1 = fn_decl @C.as.ImplicitAs.impl.Convert.loc8 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.84d = value_param_pattern [concrete = constants.%self.param_patt.98b]
// CHECK:STDOUT:     %self.patt: %pattern_type.84d = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.975]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc8_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.771 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.0e1.%C [concrete = constants.%C.771]
// CHECK:STDOUT:     %self: %C.771 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.0e1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.4c1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.f6a: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.2 = fn_decl @C.as.ImplicitAs.impl.Convert.loc9 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.91a = value_param_pattern [concrete = constants.%self.param_patt.327]
// CHECK:STDOUT:     %self.patt: %pattern_type.91a = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.533]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc9_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.83b = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.f6a.%C [concrete = constants.%C.83b]
// CHECK:STDOUT:     %self: %C.83b = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.f6a [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.c46]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.aab: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.3 = fn_decl @C.as.ImplicitAs.impl.Convert.loc10 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7dd = value_param_pattern [concrete = constants.%self.param_patt.e14]
// CHECK:STDOUT:     %self.patt: %pattern_type.7dd = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.426]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc10_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.5ce = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.aab.%C [concrete = constants.%C.5ce]
// CHECK:STDOUT:     %self: %C.5ce = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.aab [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.430]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.203: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.4 = fn_decl @C.as.ImplicitAs.impl.Convert.loc11 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.26a = value_param_pattern [concrete = constants.%self.param_patt.0fc]
// CHECK:STDOUT:     %self.patt: %pattern_type.26a = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.450]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc11_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.b3e = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.203.%C [concrete = constants.%C.b3e]
// CHECK:STDOUT:     %self: %C.b3e = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.203 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.b6d]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.371: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.5 = fn_decl @C.as.ImplicitAs.impl.Convert.loc12 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.387 = value_param_pattern [concrete = constants.%self.param_patt.826]
// CHECK:STDOUT:     %self.patt: %pattern_type.387 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.401]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc12_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.8ee = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.371.%C [concrete = constants.%C.8ee]
// CHECK:STDOUT:     %self: %C.8ee = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.371 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.c85]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.c73: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.6 = fn_decl @C.as.ImplicitAs.impl.Convert.loc13 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.05e = value_param_pattern [concrete = constants.%self.param_patt.f7b]
// CHECK:STDOUT:     %self.patt: %pattern_type.05e = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.283]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc13_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.5c1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.c73.%C [concrete = constants.%C.5c1]
// CHECK:STDOUT:     %self: %C.5c1 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.c73 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.1af]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.2f3: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.7 = fn_decl @C.as.ImplicitAs.impl.Convert.loc14 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.4b0 = value_param_pattern [concrete = constants.%self.param_patt.e33]
// CHECK:STDOUT:     %self.patt: %pattern_type.4b0 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.7b4]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc14_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.57b = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.2f3.%C [concrete = constants.%C.57b]
// CHECK:STDOUT:     %self: %C.57b = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.2f3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.c9c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.28a: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.8 = fn_decl @C.as.ImplicitAs.impl.Convert.loc15 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.bc2 = value_param_pattern [concrete = constants.%self.param_patt.21a]
// CHECK:STDOUT:     %self.patt: %pattern_type.bc2 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.94c]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc15_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.b0a = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.28a.%C [concrete = constants.%C.b0a]
// CHECK:STDOUT:     %self: %C.b0a = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.28a [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.642]
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
// CHECK:STDOUT:   %N.patt.loc4_10.2: %pattern_type.6b6 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_10.2 (constants.%N.patt.38a)]
// CHECK:STDOUT:   %N.loc4_10.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_10.1 (constants.%N.37f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.bf4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc5_16: %D.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %.loc5_28: %D.elem = field_decl m, element1 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n.m.fa1 [concrete = constants.%complete_type.237]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .n = %.loc5_16
// CHECK:STDOUT:   .m = %.loc5_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc6_31: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_0.loc6_39: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc6_40.1: %struct_type.n.m.819 = struct_literal (%int_0.loc6_31, %int_0.loc6_39) [concrete = constants.%struct]
// CHECK:STDOUT:   %impl.elem0.loc6_40.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc6_40.1: <bound method> = bound_method %int_0.loc6_31, %impl.elem0.loc6_40.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4]
// CHECK:STDOUT:   %specific_fn.loc6_40.1: <specific function> = specific_function %impl.elem0.loc6_40.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.2: <bound method> = bound_method %int_0.loc6_31, %specific_fn.loc6_40.1 [concrete = constants.%bound_method.6e0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.1: init %i32 = call %bound_method.loc6_40.2(%int_0.loc6_31) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc6_40.2: init %i32 = converted %int_0.loc6_31, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc6_40.3: ref %i32 = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc6_40.4: init %i32 to %.loc6_40.3 = in_place_init %.loc6_40.2 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %impl.elem0.loc6_40.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc6_40.3: <bound method> = bound_method %int_0.loc6_39, %impl.elem0.loc6_40.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4]
// CHECK:STDOUT:   %specific_fn.loc6_40.2: <specific function> = specific_function %impl.elem0.loc6_40.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.4: <bound method> = bound_method %int_0.loc6_39, %specific_fn.loc6_40.2 [concrete = constants.%bound_method.6e0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.2: init %i32 = call %bound_method.loc6_40.4(%int_0.loc6_39) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc6_40.5: init %i32 = converted %int_0.loc6_39, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.2 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc6_40.6: ref %i32 = class_element_access %return.param, element1
// CHECK:STDOUT:   %.loc6_40.7: init %i32 to %.loc6_40.6 = in_place_init %.loc6_40.5 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc6_40.8: init %D to %return.param = class_init (%.loc6_40.4, %.loc6_40.7) [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc6_41: init %D = converted %.loc6_40.1, %.loc6_40.8 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc6_41 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc8(%self.param: %C.771) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc8_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc8_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc9(%self.param: %C.83b) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc9_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc9_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc10(%self.param: %C.5ce) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc10_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc10_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc11(%self.param: %C.b3e) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc11_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc11_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc12(%self.param: %C.8ee) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc12_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc12_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc13(%self.param: %C.5c1) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc13_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc14(%self.param: %C.57b) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc14_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc15(%self.param: %C.b0a) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc15_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc15_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.37f) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.38a
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%N.37f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.3c0) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.38a
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_0.3c0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.0c6) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.38a
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_1.0c6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.295) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.38a
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_2.295
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.410) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.38a
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_3.410
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.4eb) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.38a
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_4.4eb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.eef) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.38a
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_5.eef
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.439) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.38a
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_6.439
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.690) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.38a
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_7.690
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %n.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.6b6 = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %struct_type.n.m: type = struct_type {.n: %i32, .m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.237: <witness> = complete_type_witness %struct_type.n.m [concrete]
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
// CHECK:STDOUT:   %N.37f: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.38a: %pattern_type.6b6 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9f1: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.9f1) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet.9f1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.6e0: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %C.771: type = class_type @C, @C(%int_0.3c0) [concrete]
// CHECK:STDOUT:   %C.val.d9b: %C.771 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.24d: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.83b: type = class_type @C, @C(%int_1.0c6) [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.5ce: type = class_type @C, @C(%int_2.295) [concrete]
// CHECK:STDOUT:   %int_3.410: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.b3e: type = class_type @C, @C(%int_3.410) [concrete]
// CHECK:STDOUT:   %int_4.4eb: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %C.8ee: type = class_type @C, @C(%int_4.4eb) [concrete]
// CHECK:STDOUT:   %int_5.eef: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %C.5c1: type = class_type @C, @C(%int_5.eef) [concrete]
// CHECK:STDOUT:   %int_6.439: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %C.57b: type = class_type @C, @C(%int_6.439) [concrete]
// CHECK:STDOUT:   %int_7.690: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %C.b0a: type = class_type @C, @C(%int_7.690) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.39f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f81 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.93a: %ImplicitAs.type.24d = facet_value %C.771, (%ImplicitAs.impl_witness.39f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c58: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.93a) [concrete]
// CHECK:STDOUT:   %.f28: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c58, %ImplicitAs.facet.93a [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.1: type = fn_type @C.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.1: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.d30: <bound method> = bound_method %C.val.d9b, %C.as.ImplicitAs.impl.Convert.3cefb0.1 [concrete]
// CHECK:STDOUT:   %.de3: ref %C.771 = temporary invalid, %C.val.d9b [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc7_26.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.234: <bound method> = bound_method %.de3, %Destroy.Op.1a2547.2 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.399: %C.83b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d9f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.032 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.da3: %ImplicitAs.type.24d = facet_value %C.83b, (%ImplicitAs.impl_witness.d9f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.0dc: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.da3) [concrete]
// CHECK:STDOUT:   %.a7f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.0dc, %ImplicitAs.facet.da3 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.2: type = fn_type @C.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.2: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.b85: <bound method> = bound_method %C.val.399, %C.as.ImplicitAs.impl.Convert.3cefb0.2 [concrete]
// CHECK:STDOUT:   %.027: ref %C.83b = temporary invalid, %C.val.399 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.d6e: <bound method> = bound_method %.027, %Destroy.Op.1a2547.3 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.040: %C.5ce = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.409: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.93f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.bb4: %ImplicitAs.type.24d = facet_value %C.5ce, (%ImplicitAs.impl_witness.409) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.2a1: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.bb4) [concrete]
// CHECK:STDOUT:   %.8f7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.2a1, %ImplicitAs.facet.bb4 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.3: type = fn_type @C.as.ImplicitAs.impl.Convert.3 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.3: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.3 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.468: <bound method> = bound_method %C.val.040, %C.as.ImplicitAs.impl.Convert.3cefb0.3 [concrete]
// CHECK:STDOUT:   %.ed7: ref %C.5ce = temporary invalid, %C.val.040 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.458: <bound method> = bound_method %.ed7, %Destroy.Op.1a2547.4 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.763: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.cf1: %C.b3e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.69a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.62c [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.7a4: %ImplicitAs.type.24d = facet_value %C.b3e, (%ImplicitAs.impl_witness.69a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dfc: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.7a4) [concrete]
// CHECK:STDOUT:   %.8eb: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.dfc, %ImplicitAs.facet.7a4 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.4: type = fn_type @C.as.ImplicitAs.impl.Convert.4 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.4: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.4 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.891: <bound method> = bound_method %C.val.cf1, %C.as.ImplicitAs.impl.Convert.3cefb0.4 [concrete]
// CHECK:STDOUT:   %.dc8: ref %C.b3e = temporary invalid, %C.val.cf1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.5: type = fn_type @Destroy.Op.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.5: %Destroy.Op.type.1d8f74.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.5e3: <bound method> = bound_method %.dc8, %Destroy.Op.1a2547.5 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aad: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.b7a: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.19e: %C.8ee = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.af6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.3ac [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.007: %ImplicitAs.type.24d = facet_value %C.8ee, (%ImplicitAs.impl_witness.af6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.7d2: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.007) [concrete]
// CHECK:STDOUT:   %.9db: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.7d2, %ImplicitAs.facet.007 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.5: type = fn_type @C.as.ImplicitAs.impl.Convert.5 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.5: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.5 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.65e: <bound method> = bound_method %C.val.19e, %C.as.ImplicitAs.impl.Convert.3cefb0.5 [concrete]
// CHECK:STDOUT:   %.054: ref %C.8ee = temporary invalid, %C.val.19e [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.6: type = fn_type @Destroy.Op.loc11 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.6: %Destroy.Op.type.1d8f74.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.670: <bound method> = bound_method %.054, %Destroy.Op.1a2547.6 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f6c: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.c9d: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.6ca: %C.5c1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.b0e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a96 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.930: %ImplicitAs.type.24d = facet_value %C.5c1, (%ImplicitAs.impl_witness.b0e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.950: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.930) [concrete]
// CHECK:STDOUT:   %.cf1: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.950, %ImplicitAs.facet.930 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.6: type = fn_type @C.as.ImplicitAs.impl.Convert.6 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.6: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.6 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.c06: <bound method> = bound_method %C.val.6ca, %C.as.ImplicitAs.impl.Convert.3cefb0.6 [concrete]
// CHECK:STDOUT:   %.6f6: ref %C.5c1 = temporary invalid, %C.val.6ca [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc12 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.443: <bound method> = bound_method %.6f6, %Destroy.Op.1a2547.7 [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.13d: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.be9: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.225: %C.57b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4cc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.91e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9e1: %ImplicitAs.type.24d = facet_value %C.57b, (%ImplicitAs.impl_witness.4cc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c1e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.9e1) [concrete]
// CHECK:STDOUT:   %.68f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c1e, %ImplicitAs.facet.9e1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.7: type = fn_type @C.as.ImplicitAs.impl.Convert.7 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.7: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.7 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.edd: <bound method> = bound_method %C.val.225, %C.as.ImplicitAs.impl.Convert.3cefb0.7 [concrete]
// CHECK:STDOUT:   %.fa8: ref %C.57b = temporary invalid, %C.val.225 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.loc13 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.742: <bound method> = bound_method %.fa8, %Destroy.Op.1a2547.8 [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.66b: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.7ff: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.79c: %C.b0a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.82a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.990 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b94: %ImplicitAs.type.24d = facet_value %C.b0a, (%ImplicitAs.impl_witness.82a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ff6: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.b94) [concrete]
// CHECK:STDOUT:   %.eb8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ff6, %ImplicitAs.facet.b94 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.8: type = fn_type @C.as.ImplicitAs.impl.Convert.8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.8: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.8 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.a06: <bound method> = bound_method %C.val.79c, %C.as.ImplicitAs.impl.Convert.3cefb0.8 [concrete]
// CHECK:STDOUT:   %.a0a: ref %C.b0a = temporary invalid, %C.val.79c [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.9: type = fn_type @Destroy.Op.loc14 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.9: %Destroy.Op.type.1d8f74.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.e4a: <bound method> = bound_method %.a0a, %Destroy.Op.1a2547.9 [concrete]
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
// CHECK:STDOUT:   %P.import_ref.6bc: <witness> = import_ref P//library, loc5_35, loaded [concrete = constants.%complete_type.237]
// CHECK:STDOUT:   %P.import_ref.737 = import_ref P//library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.9f8 = import_ref P//library, loc5_16, unloaded
// CHECK:STDOUT:   %P.import_ref.0cb = import_ref P//library, loc5_28, unloaded
// CHECK:STDOUT:   %P.C: %C.type = import_ref P//library, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %P.import_ref.8f2: <witness> = import_ref P//library, loc4_19, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %P.import_ref.70e = import_ref P//library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.d2a: %i32 = import_ref P//library, loc4_10, loaded [symbolic = @C.%N (constants.%N.37f)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %P.import_ref.b19: <witness> = import_ref P//library, loc8_33, loaded [concrete = constants.%ImplicitAs.impl_witness.39f]
// CHECK:STDOUT:   %P.import_ref.5cd: type = import_ref P//library, loc8_9, loaded [concrete = constants.%C.771]
// CHECK:STDOUT:   %P.import_ref.4efc12.1: type = import_ref P//library, loc8_31, loaded [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   %P.import_ref.7c1: <witness> = import_ref P//library, loc9_33, loaded [concrete = constants.%ImplicitAs.impl_witness.d9f]
// CHECK:STDOUT:   %P.import_ref.25e: type = import_ref P//library, loc9_9, loaded [concrete = constants.%C.83b]
// CHECK:STDOUT:   %P.import_ref.4efc12.2: type = import_ref P//library, loc9_31, loaded [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   %P.import_ref.ad3: <witness> = import_ref P//library, loc10_33, loaded [concrete = constants.%ImplicitAs.impl_witness.409]
// CHECK:STDOUT:   %P.import_ref.30e: type = import_ref P//library, loc10_9, loaded [concrete = constants.%C.5ce]
// CHECK:STDOUT:   %P.import_ref.4efc12.3: type = import_ref P//library, loc10_31, loaded [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   %P.import_ref.beb: <witness> = import_ref P//library, loc11_33, loaded [concrete = constants.%ImplicitAs.impl_witness.69a]
// CHECK:STDOUT:   %P.import_ref.edf: type = import_ref P//library, loc11_9, loaded [concrete = constants.%C.b3e]
// CHECK:STDOUT:   %P.import_ref.4efc12.4: type = import_ref P//library, loc11_31, loaded [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   %P.import_ref.681: <witness> = import_ref P//library, loc12_33, loaded [concrete = constants.%ImplicitAs.impl_witness.af6]
// CHECK:STDOUT:   %P.import_ref.e77: type = import_ref P//library, loc12_9, loaded [concrete = constants.%C.8ee]
// CHECK:STDOUT:   %P.import_ref.4efc12.5: type = import_ref P//library, loc12_31, loaded [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   %P.import_ref.cea: <witness> = import_ref P//library, loc13_33, loaded [concrete = constants.%ImplicitAs.impl_witness.b0e]
// CHECK:STDOUT:   %P.import_ref.c8c: type = import_ref P//library, loc13_9, loaded [concrete = constants.%C.5c1]
// CHECK:STDOUT:   %P.import_ref.4efc12.6: type = import_ref P//library, loc13_31, loaded [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   %P.import_ref.cdb: <witness> = import_ref P//library, loc14_33, loaded [concrete = constants.%ImplicitAs.impl_witness.4cc]
// CHECK:STDOUT:   %P.import_ref.847: type = import_ref P//library, loc14_9, loaded [concrete = constants.%C.57b]
// CHECK:STDOUT:   %P.import_ref.4efc12.7: type = import_ref P//library, loc14_31, loaded [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   %P.import_ref.2b7: <witness> = import_ref P//library, loc15_33, loaded [concrete = constants.%ImplicitAs.impl_witness.82a]
// CHECK:STDOUT:   %P.import_ref.584: type = import_ref P//library, loc15_9, loaded [concrete = constants.%C.b0a]
// CHECK:STDOUT:   %P.import_ref.4efc12.8: type = import_ref P//library, loc15_31, loaded [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   %P.import_ref.b89e2b.1: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = import_ref P//library, loc8_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f81 = impl_witness_table (%P.import_ref.b89e2b.1), @C.as.ImplicitAs.impl.fa9 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %P.import_ref.b89e2b.2: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = import_ref P//library, loc9_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.032 = impl_witness_table (%P.import_ref.b89e2b.2), @C.as.ImplicitAs.impl.c89 [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.3: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.3 = import_ref P//library, loc10_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.3]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.93f = impl_witness_table (%P.import_ref.b89e2b.3), @C.as.ImplicitAs.impl.cc3 [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.4: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.4 = import_ref P//library, loc11_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.4]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.62c = impl_witness_table (%P.import_ref.b89e2b.4), @C.as.ImplicitAs.impl.2a2 [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.5: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.5 = import_ref P//library, loc12_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.5]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.3ac = impl_witness_table (%P.import_ref.b89e2b.5), @C.as.ImplicitAs.impl.862 [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.6: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.6 = import_ref P//library, loc13_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.6]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a96 = impl_witness_table (%P.import_ref.b89e2b.6), @C.as.ImplicitAs.impl.6ea [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.7: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.7 = import_ref P//library, loc14_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.7]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.91e = impl_witness_table (%P.import_ref.b89e2b.7), @C.as.ImplicitAs.impl.36b [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.8: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.8 = import_ref P//library, loc15_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.8]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.990 = impl_witness_table (%P.import_ref.b89e2b.8), @C.as.ImplicitAs.impl.550 [concrete]
// CHECK:STDOUT:   %P.Make: %Make.type = import_ref P//library, Make, loaded [concrete = constants.%Make]
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
// CHECK:STDOUT:     %n.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%n.param_patt]
// CHECK:STDOUT:     %n.patt: %pattern_type.6b6 = at_binding_pattern n, %n.param_patt [concrete = constants.%n.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %P.ref.loc6: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%P.D [concrete = constants.%D]
// CHECK:STDOUT:     %.loc6_26.10: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.fa9: imports.%P.import_ref.5cd as imports.%P.import_ref.4efc12.1 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.b19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.c89: imports.%P.import_ref.25e as imports.%P.import_ref.4efc12.2 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.7c1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.cc3: imports.%P.import_ref.30e as imports.%P.import_ref.4efc12.3 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.ad3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.2a2: imports.%P.import_ref.edf as imports.%P.import_ref.4efc12.4 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.beb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.862: imports.%P.import_ref.e77 as imports.%P.import_ref.4efc12.5 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.681
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.6ea: imports.%P.import_ref.c8c as imports.%P.import_ref.4efc12.6 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.cea
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.36b: imports.%P.import_ref.847 as imports.%P.import_ref.4efc12.7 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.cdb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.550: imports.%P.import_ref.584 as imports.%P.import_ref.4efc12.8 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.2b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "library.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%P.import_ref.6bc
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%P.import_ref.737
// CHECK:STDOUT:   .n = imports.%P.import_ref.9f8
// CHECK:STDOUT:   .m = imports.%P.import_ref.0cb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%P.import_ref.d2a: %i32) [from "library.carbon"] {
// CHECK:STDOUT:   %N.patt: %pattern_type.6b6 = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.38a)]
// CHECK:STDOUT:   %N: %i32 = symbolic_binding N, 0 [symbolic = %N (constants.%N.37f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%P.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%P.import_ref.70e
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
// CHECK:STDOUT:   %impl.elem0.loc7_34: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc7_34.1: <bound method> = bound_method %int_0, %impl.elem0.loc7_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7b4]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_34.2: <bound method> = bound_method %int_0, %specific_fn.loc7 [concrete = constants.%bound_method.6e0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_34.2(%int_0) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc7_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc7_34.2: %i32 = converted %int_0, %.loc7_34.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %C.loc7: type = class_type @C, @C(constants.%int_0.3c0) [concrete = constants.%C.771]
// CHECK:STDOUT:   %.loc7_24.2: ref %C.771 = temporary_storage
// CHECK:STDOUT:   %.loc7_24.3: init %C.771 to %.loc7_24.2 = class_init () [concrete = constants.%C.val.d9b]
// CHECK:STDOUT:   %.loc7_26.1: init %C.771 = converted %.loc7_24.1, %.loc7_24.3 [concrete = constants.%C.val.d9b]
// CHECK:STDOUT:   %impl.elem0.loc7_35: %.f28 = impl_witness_access constants.%ImplicitAs.impl_witness.39f, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %bound_method.loc7_35: <bound method> = bound_method %.loc7_26.1, %impl.elem0.loc7_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.d30]
// CHECK:STDOUT:   %.loc6_26.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc7_26.2: ref %C.771 = temporary %.loc7_24.2, %.loc7_26.1 [concrete = constants.%.de3]
// CHECK:STDOUT:   %.loc7_26.3: %C.771 = acquire_value %.loc7_26.2 [concrete = constants.%C.val.d9b]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc7: init %D to %.loc6_26.1 = call %bound_method.loc7_35(%.loc7_26.3)
// CHECK:STDOUT:   %.loc7_35: init %D = converted %.loc7_26.1, %C.as.ImplicitAs.impl.Convert.call.loc7
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.1: init %empty_tuple.type = call constants.%Destroy.Op.bound.234(constants.%.de3)
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
// CHECK:STDOUT:   %impl.elem0.loc8_34: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_1, %specific_fn.loc8 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_34.2(%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_1, %.loc8_34.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %C.loc8: type = class_type @C, @C(constants.%int_1.0c6) [concrete = constants.%C.83b]
// CHECK:STDOUT:   %.loc8_24.2: ref %C.83b = temporary_storage
// CHECK:STDOUT:   %.loc8_24.3: init %C.83b to %.loc8_24.2 = class_init () [concrete = constants.%C.val.399]
// CHECK:STDOUT:   %.loc8_26.1: init %C.83b = converted %.loc8_24.1, %.loc8_24.3 [concrete = constants.%C.val.399]
// CHECK:STDOUT:   %impl.elem0.loc8_35: %.a7f = impl_witness_access constants.%ImplicitAs.impl_witness.d9f, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:   %bound_method.loc8_35: <bound method> = bound_method %.loc8_26.1, %impl.elem0.loc8_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.b85]
// CHECK:STDOUT:   %.loc6_26.2: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc8_26.2: ref %C.83b = temporary %.loc8_24.2, %.loc8_26.1 [concrete = constants.%.027]
// CHECK:STDOUT:   %.loc8_26.3: %C.83b = acquire_value %.loc8_26.2 [concrete = constants.%C.val.399]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc8: init %D to %.loc6_26.2 = call %bound_method.loc8_35(%.loc8_26.3)
// CHECK:STDOUT:   %.loc8_35: init %D = converted %.loc8_26.1, %C.as.ImplicitAs.impl.Convert.call.loc8
// CHECK:STDOUT:   %Destroy.Op.call.loc8_26.1: init %empty_tuple.type = call constants.%Destroy.Op.bound.d6e(constants.%.027)
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.2: init %empty_tuple.type = call constants.%Destroy.Op.bound.234(constants.%.de3)
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
// CHECK:STDOUT:   %impl.elem0.loc9_34: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %int_2, %impl.elem0.loc9_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %int_2, %specific_fn.loc9 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_34.2(%int_2) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc9_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc9_34.2: %i32 = converted %int_2, %.loc9_34.1 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %C.loc9: type = class_type @C, @C(constants.%int_2.295) [concrete = constants.%C.5ce]
// CHECK:STDOUT:   %.loc9_24.2: ref %C.5ce = temporary_storage
// CHECK:STDOUT:   %.loc9_24.3: init %C.5ce to %.loc9_24.2 = class_init () [concrete = constants.%C.val.040]
// CHECK:STDOUT:   %.loc9_26.1: init %C.5ce = converted %.loc9_24.1, %.loc9_24.3 [concrete = constants.%C.val.040]
// CHECK:STDOUT:   %impl.elem0.loc9_35: %.8f7 = impl_witness_access constants.%ImplicitAs.impl_witness.409, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.3]
// CHECK:STDOUT:   %bound_method.loc9_35: <bound method> = bound_method %.loc9_26.1, %impl.elem0.loc9_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.468]
// CHECK:STDOUT:   %.loc6_26.3: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc9_26.2: ref %C.5ce = temporary %.loc9_24.2, %.loc9_26.1 [concrete = constants.%.ed7]
// CHECK:STDOUT:   %.loc9_26.3: %C.5ce = acquire_value %.loc9_26.2 [concrete = constants.%C.val.040]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc9: init %D to %.loc6_26.3 = call %bound_method.loc9_35(%.loc9_26.3)
// CHECK:STDOUT:   %.loc9_35: init %D = converted %.loc9_26.1, %C.as.ImplicitAs.impl.Convert.call.loc9
// CHECK:STDOUT:   %Destroy.Op.call.loc9_26.1: init %empty_tuple.type = call constants.%Destroy.Op.bound.458(constants.%.ed7)
// CHECK:STDOUT:   %Destroy.Op.call.loc8_26.2: init %empty_tuple.type = call constants.%Destroy.Op.bound.d6e(constants.%.027)
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.3: init %empty_tuple.type = call constants.%Destroy.Op.bound.234(constants.%.de3)
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
// CHECK:STDOUT:   %impl.elem0.loc10_34: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc10_34.1: <bound method> = bound_method %int_3, %impl.elem0.loc10_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_34.2: <bound method> = bound_method %int_3, %specific_fn.loc10 [concrete = constants.%bound_method.763]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %i32 = call %bound_method.loc10_34.2(%int_3) [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc10_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc10_34.2: %i32 = converted %int_3, %.loc10_34.1 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %C.loc10: type = class_type @C, @C(constants.%int_3.410) [concrete = constants.%C.b3e]
// CHECK:STDOUT:   %.loc10_24.2: ref %C.b3e = temporary_storage
// CHECK:STDOUT:   %.loc10_24.3: init %C.b3e to %.loc10_24.2 = class_init () [concrete = constants.%C.val.cf1]
// CHECK:STDOUT:   %.loc10_26.1: init %C.b3e = converted %.loc10_24.1, %.loc10_24.3 [concrete = constants.%C.val.cf1]
// CHECK:STDOUT:   %impl.elem0.loc10_35: %.8eb = impl_witness_access constants.%ImplicitAs.impl_witness.69a, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.4]
// CHECK:STDOUT:   %bound_method.loc10_35: <bound method> = bound_method %.loc10_26.1, %impl.elem0.loc10_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.891]
// CHECK:STDOUT:   %.loc6_26.4: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc10_26.2: ref %C.b3e = temporary %.loc10_24.2, %.loc10_26.1 [concrete = constants.%.dc8]
// CHECK:STDOUT:   %.loc10_26.3: %C.b3e = acquire_value %.loc10_26.2 [concrete = constants.%C.val.cf1]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc10: init %D to %.loc6_26.4 = call %bound_method.loc10_35(%.loc10_26.3)
// CHECK:STDOUT:   %.loc10_35: init %D = converted %.loc10_26.1, %C.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %Destroy.Op.call.loc10_26.1: init %empty_tuple.type = call constants.%Destroy.Op.bound.5e3(constants.%.dc8)
// CHECK:STDOUT:   %Destroy.Op.call.loc9_26.2: init %empty_tuple.type = call constants.%Destroy.Op.bound.458(constants.%.ed7)
// CHECK:STDOUT:   %Destroy.Op.call.loc8_26.3: init %empty_tuple.type = call constants.%Destroy.Op.bound.d6e(constants.%.027)
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.4: init %empty_tuple.type = call constants.%Destroy.Op.bound.234(constants.%.de3)
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
// CHECK:STDOUT:   %impl.elem0.loc11_34: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc11_34.1: <bound method> = bound_method %int_4, %impl.elem0.loc11_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.aad]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_34.2: <bound method> = bound_method %int_4, %specific_fn.loc11 [concrete = constants.%bound_method.b7a]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %i32 = call %bound_method.loc11_34.2(%int_4) [concrete = constants.%int_4.4eb]
// CHECK:STDOUT:   %.loc11_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_4.4eb]
// CHECK:STDOUT:   %.loc11_34.2: %i32 = converted %int_4, %.loc11_34.1 [concrete = constants.%int_4.4eb]
// CHECK:STDOUT:   %C.loc11: type = class_type @C, @C(constants.%int_4.4eb) [concrete = constants.%C.8ee]
// CHECK:STDOUT:   %.loc11_24.2: ref %C.8ee = temporary_storage
// CHECK:STDOUT:   %.loc11_24.3: init %C.8ee to %.loc11_24.2 = class_init () [concrete = constants.%C.val.19e]
// CHECK:STDOUT:   %.loc11_26.1: init %C.8ee = converted %.loc11_24.1, %.loc11_24.3 [concrete = constants.%C.val.19e]
// CHECK:STDOUT:   %impl.elem0.loc11_35: %.9db = impl_witness_access constants.%ImplicitAs.impl_witness.af6, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.5]
// CHECK:STDOUT:   %bound_method.loc11_35: <bound method> = bound_method %.loc11_26.1, %impl.elem0.loc11_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.65e]
// CHECK:STDOUT:   %.loc6_26.5: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc11_26.2: ref %C.8ee = temporary %.loc11_24.2, %.loc11_26.1 [concrete = constants.%.054]
// CHECK:STDOUT:   %.loc11_26.3: %C.8ee = acquire_value %.loc11_26.2 [concrete = constants.%C.val.19e]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc11: init %D to %.loc6_26.5 = call %bound_method.loc11_35(%.loc11_26.3)
// CHECK:STDOUT:   %.loc11_35: init %D = converted %.loc11_26.1, %C.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %Destroy.Op.call.loc11_26.1: init %empty_tuple.type = call constants.%Destroy.Op.bound.670(constants.%.054)
// CHECK:STDOUT:   %Destroy.Op.call.loc10_26.2: init %empty_tuple.type = call constants.%Destroy.Op.bound.5e3(constants.%.dc8)
// CHECK:STDOUT:   %Destroy.Op.call.loc9_26.3: init %empty_tuple.type = call constants.%Destroy.Op.bound.458(constants.%.ed7)
// CHECK:STDOUT:   %Destroy.Op.call.loc8_26.4: init %empty_tuple.type = call constants.%Destroy.Op.bound.d6e(constants.%.027)
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.5: init %empty_tuple.type = call constants.%Destroy.Op.bound.234(constants.%.de3)
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
// CHECK:STDOUT:   %impl.elem0.loc12_34: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc12_34.1: <bound method> = bound_method %int_5, %impl.elem0.loc12_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f6c]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_34.2: <bound method> = bound_method %int_5, %specific_fn.loc12 [concrete = constants.%bound_method.c9d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %i32 = call %bound_method.loc12_34.2(%int_5) [concrete = constants.%int_5.eef]
// CHECK:STDOUT:   %.loc12_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_5.eef]
// CHECK:STDOUT:   %.loc12_34.2: %i32 = converted %int_5, %.loc12_34.1 [concrete = constants.%int_5.eef]
// CHECK:STDOUT:   %C.loc12: type = class_type @C, @C(constants.%int_5.eef) [concrete = constants.%C.5c1]
// CHECK:STDOUT:   %.loc12_24.2: ref %C.5c1 = temporary_storage
// CHECK:STDOUT:   %.loc12_24.3: init %C.5c1 to %.loc12_24.2 = class_init () [concrete = constants.%C.val.6ca]
// CHECK:STDOUT:   %.loc12_26.1: init %C.5c1 = converted %.loc12_24.1, %.loc12_24.3 [concrete = constants.%C.val.6ca]
// CHECK:STDOUT:   %impl.elem0.loc12_35: %.cf1 = impl_witness_access constants.%ImplicitAs.impl_witness.b0e, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.6]
// CHECK:STDOUT:   %bound_method.loc12_35: <bound method> = bound_method %.loc12_26.1, %impl.elem0.loc12_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.c06]
// CHECK:STDOUT:   %.loc6_26.6: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc12_26.2: ref %C.5c1 = temporary %.loc12_24.2, %.loc12_26.1 [concrete = constants.%.6f6]
// CHECK:STDOUT:   %.loc12_26.3: %C.5c1 = acquire_value %.loc12_26.2 [concrete = constants.%C.val.6ca]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc12: init %D to %.loc6_26.6 = call %bound_method.loc12_35(%.loc12_26.3)
// CHECK:STDOUT:   %.loc12_35: init %D = converted %.loc12_26.1, %C.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %Destroy.Op.call.loc12_26.1: init %empty_tuple.type = call constants.%Destroy.Op.bound.443(constants.%.6f6)
// CHECK:STDOUT:   %Destroy.Op.call.loc11_26.2: init %empty_tuple.type = call constants.%Destroy.Op.bound.670(constants.%.054)
// CHECK:STDOUT:   %Destroy.Op.call.loc10_26.3: init %empty_tuple.type = call constants.%Destroy.Op.bound.5e3(constants.%.dc8)
// CHECK:STDOUT:   %Destroy.Op.call.loc9_26.4: init %empty_tuple.type = call constants.%Destroy.Op.bound.458(constants.%.ed7)
// CHECK:STDOUT:   %Destroy.Op.call.loc8_26.5: init %empty_tuple.type = call constants.%Destroy.Op.bound.d6e(constants.%.027)
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.6: init %empty_tuple.type = call constants.%Destroy.Op.bound.234(constants.%.de3)
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
// CHECK:STDOUT:   %impl.elem0.loc13_34: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc13_34.1: <bound method> = bound_method %int_6, %impl.elem0.loc13_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.13d]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_34.2: <bound method> = bound_method %int_6, %specific_fn.loc13 [concrete = constants.%bound_method.be9]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %i32 = call %bound_method.loc13_34.2(%int_6) [concrete = constants.%int_6.439]
// CHECK:STDOUT:   %.loc13_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%int_6.439]
// CHECK:STDOUT:   %.loc13_34.2: %i32 = converted %int_6, %.loc13_34.1 [concrete = constants.%int_6.439]
// CHECK:STDOUT:   %C.loc13: type = class_type @C, @C(constants.%int_6.439) [concrete = constants.%C.57b]
// CHECK:STDOUT:   %.loc13_24.2: ref %C.57b = temporary_storage
// CHECK:STDOUT:   %.loc13_24.3: init %C.57b to %.loc13_24.2 = class_init () [concrete = constants.%C.val.225]
// CHECK:STDOUT:   %.loc13_26.1: init %C.57b = converted %.loc13_24.1, %.loc13_24.3 [concrete = constants.%C.val.225]
// CHECK:STDOUT:   %impl.elem0.loc13_35: %.68f = impl_witness_access constants.%ImplicitAs.impl_witness.4cc, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.7]
// CHECK:STDOUT:   %bound_method.loc13_35: <bound method> = bound_method %.loc13_26.1, %impl.elem0.loc13_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.edd]
// CHECK:STDOUT:   %.loc6_26.7: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc13_26.2: ref %C.57b = temporary %.loc13_24.2, %.loc13_26.1 [concrete = constants.%.fa8]
// CHECK:STDOUT:   %.loc13_26.3: %C.57b = acquire_value %.loc13_26.2 [concrete = constants.%C.val.225]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc13: init %D to %.loc6_26.7 = call %bound_method.loc13_35(%.loc13_26.3)
// CHECK:STDOUT:   %.loc13_35: init %D = converted %.loc13_26.1, %C.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %Destroy.Op.call.loc13_26.1: init %empty_tuple.type = call constants.%Destroy.Op.bound.742(constants.%.fa8)
// CHECK:STDOUT:   %Destroy.Op.call.loc12_26.2: init %empty_tuple.type = call constants.%Destroy.Op.bound.443(constants.%.6f6)
// CHECK:STDOUT:   %Destroy.Op.call.loc11_26.3: init %empty_tuple.type = call constants.%Destroy.Op.bound.670(constants.%.054)
// CHECK:STDOUT:   %Destroy.Op.call.loc10_26.4: init %empty_tuple.type = call constants.%Destroy.Op.bound.5e3(constants.%.dc8)
// CHECK:STDOUT:   %Destroy.Op.call.loc9_26.5: init %empty_tuple.type = call constants.%Destroy.Op.bound.458(constants.%.ed7)
// CHECK:STDOUT:   %Destroy.Op.call.loc8_26.6: init %empty_tuple.type = call constants.%Destroy.Op.bound.d6e(constants.%.027)
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.7: init %empty_tuple.type = call constants.%Destroy.Op.bound.234(constants.%.de3)
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
// CHECK:STDOUT:   %impl.elem0.loc14_34: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc14_34.1: <bound method> = bound_method %int_7, %impl.elem0.loc14_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.66b]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.2: <bound method> = bound_method %int_7, %specific_fn.loc14 [concrete = constants.%bound_method.7ff]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14: init %i32 = call %bound_method.loc14_34.2(%int_7) [concrete = constants.%int_7.690]
// CHECK:STDOUT:   %.loc14_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14 [concrete = constants.%int_7.690]
// CHECK:STDOUT:   %.loc14_34.2: %i32 = converted %int_7, %.loc14_34.1 [concrete = constants.%int_7.690]
// CHECK:STDOUT:   %C.loc14: type = class_type @C, @C(constants.%int_7.690) [concrete = constants.%C.b0a]
// CHECK:STDOUT:   %.loc14_24.2: ref %C.b0a = temporary_storage
// CHECK:STDOUT:   %.loc14_24.3: init %C.b0a to %.loc14_24.2 = class_init () [concrete = constants.%C.val.79c]
// CHECK:STDOUT:   %.loc14_26.1: init %C.b0a = converted %.loc14_24.1, %.loc14_24.3 [concrete = constants.%C.val.79c]
// CHECK:STDOUT:   %impl.elem0.loc14_35: %.eb8 = impl_witness_access constants.%ImplicitAs.impl_witness.82a, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.8]
// CHECK:STDOUT:   %bound_method.loc14_35: <bound method> = bound_method %.loc14_26.1, %impl.elem0.loc14_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.a06]
// CHECK:STDOUT:   %.loc6_26.8: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc14_26.2: ref %C.b0a = temporary %.loc14_24.2, %.loc14_26.1 [concrete = constants.%.a0a]
// CHECK:STDOUT:   %.loc14_26.3: %C.b0a = acquire_value %.loc14_26.2 [concrete = constants.%C.val.79c]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc14: init %D to %.loc6_26.8 = call %bound_method.loc14_35(%.loc14_26.3)
// CHECK:STDOUT:   %.loc14_35: init %D = converted %.loc14_26.1, %C.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %Destroy.Op.call.loc14_26.1: init %empty_tuple.type = call constants.%Destroy.Op.bound.e4a(constants.%.a0a)
// CHECK:STDOUT:   %Destroy.Op.call.loc13_26.2: init %empty_tuple.type = call constants.%Destroy.Op.bound.742(constants.%.fa8)
// CHECK:STDOUT:   %Destroy.Op.call.loc12_26.3: init %empty_tuple.type = call constants.%Destroy.Op.bound.443(constants.%.6f6)
// CHECK:STDOUT:   %Destroy.Op.call.loc11_26.4: init %empty_tuple.type = call constants.%Destroy.Op.bound.670(constants.%.054)
// CHECK:STDOUT:   %Destroy.Op.call.loc10_26.5: init %empty_tuple.type = call constants.%Destroy.Op.bound.5e3(constants.%.dc8)
// CHECK:STDOUT:   %Destroy.Op.call.loc9_26.6: init %empty_tuple.type = call constants.%Destroy.Op.bound.458(constants.%.ed7)
// CHECK:STDOUT:   %Destroy.Op.call.loc8_26.7: init %empty_tuple.type = call constants.%Destroy.Op.bound.d6e(constants.%.027)
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.8: init %empty_tuple.type = call constants.%Destroy.Op.bound.234(constants.%.de3)
// CHECK:STDOUT:   return %.loc14_35 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc14:
// CHECK:STDOUT:   %P.ref.loc15: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%P.Make [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc6_26.9: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc6_26.9 = call %Make.ref()
// CHECK:STDOUT:   %Destroy.Op.call.loc14_26.2: init %empty_tuple.type = call constants.%Destroy.Op.bound.e4a(constants.%.a0a)
// CHECK:STDOUT:   %Destroy.Op.call.loc13_26.3: init %empty_tuple.type = call constants.%Destroy.Op.bound.742(constants.%.fa8)
// CHECK:STDOUT:   %Destroy.Op.call.loc12_26.4: init %empty_tuple.type = call constants.%Destroy.Op.bound.443(constants.%.6f6)
// CHECK:STDOUT:   %Destroy.Op.call.loc11_26.5: init %empty_tuple.type = call constants.%Destroy.Op.bound.670(constants.%.054)
// CHECK:STDOUT:   %Destroy.Op.call.loc10_26.6: init %empty_tuple.type = call constants.%Destroy.Op.bound.5e3(constants.%.dc8)
// CHECK:STDOUT:   %Destroy.Op.call.loc9_26.7: init %empty_tuple.type = call constants.%Destroy.Op.bound.458(constants.%.ed7)
// CHECK:STDOUT:   %Destroy.Op.call.loc8_26.8: init %empty_tuple.type = call constants.%Destroy.Op.bound.d6e(constants.%.027)
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.9: init %empty_tuple.type = call constants.%Destroy.Op.bound.234(constants.%.de3)
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.1 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_26.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_26.2(%self.param: ref %C.771) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.2 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8(%self.param: ref %C.83b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.3 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9(%self.param: ref %C.5ce) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.4 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10(%self.param: ref %C.b3e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.5 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11(%self.param: ref %C.8ee) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.6 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12(%self.param: ref %C.5c1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.7 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13(%self.param: ref %C.57b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.8 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14(%self.param: ref %C.b0a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.37f) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.38a
// CHECK:STDOUT:   %N => constants.%N.37f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.3c0) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.38a
// CHECK:STDOUT:   %N => constants.%int_0.3c0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.0c6) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.38a
// CHECK:STDOUT:   %N => constants.%int_1.0c6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.295) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.38a
// CHECK:STDOUT:   %N => constants.%int_2.295
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.410) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.38a
// CHECK:STDOUT:   %N => constants.%int_3.410
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.4eb) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.38a
// CHECK:STDOUT:   %N => constants.%int_4.4eb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.eef) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.38a
// CHECK:STDOUT:   %N => constants.%int_5.eef
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.439) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.38a
// CHECK:STDOUT:   %N => constants.%int_6.439
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.690) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.38a
// CHECK:STDOUT:   %N => constants.%int_7.690
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
