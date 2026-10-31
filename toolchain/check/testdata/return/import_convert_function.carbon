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
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.d64: %pattern_type.d94 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.c34: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.bb3: type = class_type @C, @C(%N.c34) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.484: type = struct_type {.n: %i32, .m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.28b: <witness> = complete_type_witness %struct_type.n.m.484 [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fc6: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.fc6) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet.fc6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.762: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.036: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_0.753, %int_0.753) [concrete]
// CHECK:STDOUT:   %C.920: type = class_type @C, @C(%int_0.753) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.164: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %.131: <witness> = impl_self_witness %C.920, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2a1: <witness> = impl_witness @C.as.ImplicitAs.impl.c5b.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.0a5: type = pattern_type %C.920 [concrete]
// CHECK:STDOUT:   %self.param_patt.20a: %pattern_type.0a5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e33: %pattern_type.0a5 = wrapper_binding_pattern self, %self.param_patt.20a [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.1: type = fn_type @C.as.ImplicitAs.impl.Convert.loc8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.1: %C.as.ImplicitAs.impl.Convert.type.860b0a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.4bc: type = class_type @C, @C(%int_1.68a) [concrete]
// CHECK:STDOUT:   %.80b: <witness> = impl_self_witness %C.4bc, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f7c: <witness> = impl_witness @C.as.ImplicitAs.impl.491.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.ac7: type = pattern_type %C.4bc [concrete]
// CHECK:STDOUT:   %self.param_patt.a28: %pattern_type.ac7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.315: %pattern_type.ac7 = wrapper_binding_pattern self, %self.param_patt.a28 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.2: type = fn_type @C.as.ImplicitAs.impl.Convert.loc9 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.2: %C.as.ImplicitAs.impl.Convert.type.860b0a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a55: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.2e0: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.6d6: type = class_type @C, @C(%int_2.2e0) [concrete]
// CHECK:STDOUT:   %.230: <witness> = impl_self_witness %C.6d6, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d1e: <witness> = impl_witness @C.as.ImplicitAs.impl.c74.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.624: type = pattern_type %C.6d6 [concrete]
// CHECK:STDOUT:   %self.param_patt.c35: %pattern_type.624 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.8ce: %pattern_type.624 = wrapper_binding_pattern self, %self.param_patt.c35 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.3: type = fn_type @C.as.ImplicitAs.impl.Convert.loc10 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.3: %C.as.ImplicitAs.impl.Convert.type.860b0a.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.fb0: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.9f3: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.917: type = class_type @C, @C(%int_3.9f3) [concrete]
// CHECK:STDOUT:   %.15d: <witness> = impl_self_witness %C.917, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.179: <witness> = impl_witness @C.as.ImplicitAs.impl.65f.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.c15: type = pattern_type %C.917 [concrete]
// CHECK:STDOUT:   %self.param_patt.754: %pattern_type.c15 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.51d: %pattern_type.c15 = wrapper_binding_pattern self, %self.param_patt.754 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.4: type = fn_type @C.as.ImplicitAs.impl.Convert.loc11 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.4: %C.as.ImplicitAs.impl.Convert.type.860b0a.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.776: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.7fa: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.6bf: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %C.20d: type = class_type @C, @C(%int_4.6bf) [concrete]
// CHECK:STDOUT:   %.c03: <witness> = impl_self_witness %C.20d, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.db0: <witness> = impl_witness @C.as.ImplicitAs.impl.7f4.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.dc6: type = pattern_type %C.20d [concrete]
// CHECK:STDOUT:   %self.param_patt.acc: %pattern_type.dc6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.3b1: %pattern_type.dc6 = wrapper_binding_pattern self, %self.param_patt.acc [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.5: type = fn_type @C.as.ImplicitAs.impl.Convert.loc12 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.5: %C.as.ImplicitAs.impl.Convert.type.860b0a.5 = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f65: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.ab2: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.708: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %C.c2b: type = class_type @C, @C(%int_5.708) [concrete]
// CHECK:STDOUT:   %.9ac: <witness> = impl_self_witness %C.c2b, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.619: <witness> = impl_witness @C.as.ImplicitAs.impl.e67.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.a63: type = pattern_type %C.c2b [concrete]
// CHECK:STDOUT:   %self.param_patt.737: %pattern_type.a63 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1fe: %pattern_type.a63 = wrapper_binding_pattern self, %self.param_patt.737 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.6: type = fn_type @C.as.ImplicitAs.impl.Convert.loc13 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.6: %C.as.ImplicitAs.impl.Convert.type.860b0a.6 = struct_value () [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dcf: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.c9a: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.b33: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %C.09c: type = class_type @C, @C(%int_6.b33) [concrete]
// CHECK:STDOUT:   %.7b3: <witness> = impl_self_witness %C.09c, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.66c: <witness> = impl_witness @C.as.ImplicitAs.impl.291.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.1a9: type = pattern_type %C.09c [concrete]
// CHECK:STDOUT:   %self.param_patt.223: %pattern_type.1a9 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.fc7: %pattern_type.1a9 = wrapper_binding_pattern self, %self.param_patt.223 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.7: type = fn_type @C.as.ImplicitAs.impl.Convert.loc14 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.7: %C.as.ImplicitAs.impl.Convert.type.860b0a.7 = struct_value () [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5f1: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a5e: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.a13: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %C.284: type = class_type @C, @C(%int_7.a13) [concrete]
// CHECK:STDOUT:   %.c90: <witness> = impl_self_witness %C.284, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9ea: <witness> = impl_witness @C.as.ImplicitAs.impl.bad.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.d80: type = pattern_type %C.284 [concrete]
// CHECK:STDOUT:   %self.param_patt.906: %pattern_type.d80 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.5e3: %pattern_type.d80 = wrapper_binding_pattern self, %self.param_patt.906 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %N.patt.loc4_10.1: %pattern_type.d94 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_10.2 (constants.%N.patt.d64)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_10.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_10.1 (constants.%N.c34)]
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
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.c5b [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:     %bound_method.loc8_9.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.762]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_9.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.036]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_9.2(%int_0) [concrete = constants.%int_0.753]
// CHECK:STDOUT:     %.loc8_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.753]
// CHECK:STDOUT:     %.loc8_9.2: %i32 = converted %int_0, %.loc8_9.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_0.753) [concrete = constants.%C.920]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = impl_self_witness @C.as.ImplicitAs.impl.c5b.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.131]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.491 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:     %bound_method.loc9_9.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_9.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_9.2(%int_1) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:     %.loc9_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.68a]
// CHECK:STDOUT:     %.loc9_9.2: %i32 = converted %int_1, %.loc9_9.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_1.68a) [concrete = constants.%C.4bc]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: <witness> = impl_self_witness @C.as.ImplicitAs.impl.491.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.80b]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.c74 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:     %bound_method.loc10_9.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_9.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc10_9.2(%int_2) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:     %.loc10_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:     %.loc10_9.2: %i32 = converted %int_2, %.loc10_9.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_2.2e0) [concrete = constants.%C.6d6]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <witness> = impl_self_witness @C.as.ImplicitAs.impl.c74.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.230]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.65f [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:     %bound_method.loc11_9.1: <bound method> = bound_method %int_3, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_9.2: <bound method> = bound_method %int_3, %specific_fn [concrete = constants.%bound_method.fb0]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_9.2(%int_3) [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:     %.loc11_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:     %.loc11_9.2: %i32 = converted %int_3, %.loc11_9.1 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_3.9f3) [concrete = constants.%C.917]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = impl_self_witness @C.as.ImplicitAs.impl.65f.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.15d]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.7f4 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:     %bound_method.loc12_9.1: <bound method> = bound_method %int_4, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.776]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc12_9.2: <bound method> = bound_method %int_4, %specific_fn [concrete = constants.%bound_method.7fa]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_9.2(%int_4) [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:     %.loc12_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:     %.loc12_9.2: %i32 = converted %int_4, %.loc12_9.1 [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_4.6bf) [concrete = constants.%C.20d]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: <witness> = impl_self_witness @C.as.ImplicitAs.impl.7f4.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.c03]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.e67 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:     %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:     %bound_method.loc13_9.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f65]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc13_9.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method.ab2]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_9.2(%int_5) [concrete = constants.%int_5.708]
// CHECK:STDOUT:     %.loc13_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.708]
// CHECK:STDOUT:     %.loc13_9.2: %i32 = converted %int_5, %.loc13_9.1 [concrete = constants.%int_5.708]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_5.708) [concrete = constants.%C.c2b]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = impl_self_witness @C.as.ImplicitAs.impl.e67.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.9ac]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.291 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:     %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %int_6, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dcf]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %int_6, %specific_fn [concrete = constants.%bound_method.c9a]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc14_9.2(%int_6) [concrete = constants.%int_6.b33]
// CHECK:STDOUT:     %.loc14_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_6.b33]
// CHECK:STDOUT:     %.loc14_9.2: %i32 = converted %int_6, %.loc14_9.1 [concrete = constants.%int_6.b33]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_6.b33) [concrete = constants.%C.09c]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: <witness> = impl_self_witness @C.as.ImplicitAs.impl.291.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.7b3]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.bad [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:     %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:     %bound_method.loc15_9.1: <bound method> = bound_method %int_7, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5f1]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc15_9.2: <bound method> = bound_method %int_7, %specific_fn [concrete = constants.%bound_method.a5e]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_9.2(%int_7) [concrete = constants.%int_7.a13]
// CHECK:STDOUT:     %.loc15_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_7.a13]
// CHECK:STDOUT:     %.loc15_9.2: %i32 = converted %int_7, %.loc15_9.1 [concrete = constants.%int_7.a13]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_7.a13) [concrete = constants.%C.284]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: <witness> = impl_self_witness @C.as.ImplicitAs.impl.bad.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.c90]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.c5b: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.1 = fn_decl @C.as.ImplicitAs.impl.Convert.loc8 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0a5 = value_param_pattern [concrete = constants.%self.param_patt.20a]
// CHECK:STDOUT:     %self.patt: %pattern_type.0a5 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e33]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc8_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.920 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.c5b.%C [concrete = constants.%C.920]
// CHECK:STDOUT:     %self: %C.920 = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.c5b [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.2a1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.491: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.2 = fn_decl @C.as.ImplicitAs.impl.Convert.loc9 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ac7 = value_param_pattern [concrete = constants.%self.param_patt.a28]
// CHECK:STDOUT:     %self.patt: %pattern_type.ac7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.315]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc9_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.4bc = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.491.%C [concrete = constants.%C.4bc]
// CHECK:STDOUT:     %self: %C.4bc = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.491 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.f7c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.c74: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.3 = fn_decl @C.as.ImplicitAs.impl.Convert.loc10 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.624 = value_param_pattern [concrete = constants.%self.param_patt.c35]
// CHECK:STDOUT:     %self.patt: %pattern_type.624 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.8ce]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc10_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.6d6 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.c74.%C [concrete = constants.%C.6d6]
// CHECK:STDOUT:     %self: %C.6d6 = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.c74 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.d1e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.65f: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.4 = fn_decl @C.as.ImplicitAs.impl.Convert.loc11 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c15 = value_param_pattern [concrete = constants.%self.param_patt.754]
// CHECK:STDOUT:     %self.patt: %pattern_type.c15 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.51d]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc11_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.917 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.65f.%C [concrete = constants.%C.917]
// CHECK:STDOUT:     %self: %C.917 = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.65f [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.179]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.7f4: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.5 = fn_decl @C.as.ImplicitAs.impl.Convert.loc12 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dc6 = value_param_pattern [concrete = constants.%self.param_patt.acc]
// CHECK:STDOUT:     %self.patt: %pattern_type.dc6 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.3b1]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc12_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.20d = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.7f4.%C [concrete = constants.%C.20d]
// CHECK:STDOUT:     %self: %C.20d = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.7f4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.db0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.e67: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.6 = fn_decl @C.as.ImplicitAs.impl.Convert.loc13 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a63 = value_param_pattern [concrete = constants.%self.param_patt.737]
// CHECK:STDOUT:     %self.patt: %pattern_type.a63 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1fe]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc13_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.c2b = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.e67.%C [concrete = constants.%C.c2b]
// CHECK:STDOUT:     %self: %C.c2b = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.e67 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.619]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.291: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.7 = fn_decl @C.as.ImplicitAs.impl.Convert.loc14 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1a9 = value_param_pattern [concrete = constants.%self.param_patt.223]
// CHECK:STDOUT:     %self.patt: %pattern_type.1a9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.fc7]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc14_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.09c = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.291.%C [concrete = constants.%C.09c]
// CHECK:STDOUT:     %self: %C.09c = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.291 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.66c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.bad: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.8 = fn_decl @C.as.ImplicitAs.impl.Convert.loc15 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d80 = value_param_pattern [concrete = constants.%self.param_patt.906]
// CHECK:STDOUT:     %self.patt: %pattern_type.d80 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5e3]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc15_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.284 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.bad.%C [concrete = constants.%C.284]
// CHECK:STDOUT:     %self: %C.284 = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.bad [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.9ea]
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
// CHECK:STDOUT:   %N.patt.loc4_10.2: %pattern_type.d94 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_10.2 (constants.%N.patt.d64)]
// CHECK:STDOUT:   %N.loc4_10.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_10.1 (constants.%N.c34)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.bb3
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n.m.484 [concrete = constants.%complete_type.28b]
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
// CHECK:STDOUT:   %impl.elem0.loc6_40.1: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc6_40.1: <bound method> = bound_method %int_0.loc6_31, %impl.elem0.loc6_40.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.762]
// CHECK:STDOUT:   %specific_fn.loc6_40.1: <specific function> = specific_function %impl.elem0.loc6_40.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.2: <bound method> = bound_method %int_0.loc6_31, %specific_fn.loc6_40.1 [concrete = constants.%bound_method.036]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.1: init %i32 = call %bound_method.loc6_40.2(%int_0.loc6_31) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc6_40.2: init %i32 = converted %int_0.loc6_31, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc6_40.3: ref %i32 = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc6_40.4: init %i32 to %.loc6_40.3 = in_place_init %.loc6_40.2 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %impl.elem0.loc6_40.2: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc6_40.3: <bound method> = bound_method %int_0.loc6_39, %impl.elem0.loc6_40.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.762]
// CHECK:STDOUT:   %specific_fn.loc6_40.2: <specific function> = specific_function %impl.elem0.loc6_40.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.4: <bound method> = bound_method %int_0.loc6_39, %specific_fn.loc6_40.2 [concrete = constants.%bound_method.036]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.2: init %i32 = call %bound_method.loc6_40.4(%int_0.loc6_39) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc6_40.5: init %i32 = converted %int_0.loc6_39, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.2 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc6_40.6: ref %i32 = class_element_access %return.param, element1
// CHECK:STDOUT:   %.loc6_40.7: init %i32 to %.loc6_40.6 = in_place_init %.loc6_40.5 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc6_40.8: init %D to %return.param = class_init (%.loc6_40.4, %.loc6_40.7) [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc6_41: init %D = converted %.loc6_40.1, %.loc6_40.8 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc6_41 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc8(%self.param: %C.920) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc8_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc8_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc9(%self.param: %C.4bc) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc9_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc9_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc10(%self.param: %C.6d6) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc10_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc10_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc11(%self.param: %C.917) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc11_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc11_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc12(%self.param: %C.20d) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc12_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc12_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc13(%self.param: %C.c2b) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc13_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc14(%self.param: %C.09c) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc14_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc15(%self.param: %C.284) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc15_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc15_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.c34) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%N.c34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.753) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_0.753
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.68a) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_1.68a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.2e0) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_2.2e0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.9f3) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_3.9f3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.6bf) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_4.6bf
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.708) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_5.708
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.b33) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_6.b33
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.a13) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_7.a13
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %n.param_patt: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.d94 = wrapper_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %struct_type.n.m: type = struct_type {.n: %i32, .m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.28b: <witness> = complete_type_witness %struct_type.n.m [concrete]
// CHECK:STDOUT:   %.128d: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %pattern_type.b22: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %return.param_patt.dca: %pattern_type.b22 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.904: %pattern_type.b22 = return_slot_pattern %return.param_patt.dca, %D [concrete]
// CHECK:STDOUT:   %F0.type: type = fn_type @F0 [concrete]
// CHECK:STDOUT:   %F0: %F0.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %N.c34: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.d64: %pattern_type.d94 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fc6: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f0: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.fc6) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f0, %ImplicitAs.facet.fc6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.762: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.036: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %C.920: type = class_type @C, @C(%int_0.753) [concrete]
// CHECK:STDOUT:   %C.val.95f: %C.920 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.164: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.4bc: type = class_type @C, @C(%int_1.68a) [concrete]
// CHECK:STDOUT:   %int_2.2e0: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.6d6: type = class_type @C, @C(%int_2.2e0) [concrete]
// CHECK:STDOUT:   %int_3.9f3: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.917: type = class_type @C, @C(%int_3.9f3) [concrete]
// CHECK:STDOUT:   %int_4.6bf: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %C.20d: type = class_type @C, @C(%int_4.6bf) [concrete]
// CHECK:STDOUT:   %int_5.708: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %C.c2b: type = class_type @C, @C(%int_5.708) [concrete]
// CHECK:STDOUT:   %int_6.b33: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %C.09c: type = class_type @C, @C(%int_6.b33) [concrete]
// CHECK:STDOUT:   %int_7.a13: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %C.284: type = class_type @C, @C(%int_7.a13) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0ca: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.32b [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3ac: %ImplicitAs.type.164 = facet_value %C.920, (%ImplicitAs.impl_witness.0ca) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ad7: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.3ac) [concrete]
// CHECK:STDOUT:   %.5c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ad7, %ImplicitAs.facet.3ac [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.1: type = fn_type @C.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.1: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.2ac: <bound method> = bound_method %C.val.95f, %C.as.ImplicitAs.impl.Convert.3cefb0.1 [concrete]
// CHECK:STDOUT:   %.2eb: ref %C.920 = temporary invalid, %C.val.95f [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_26.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.0e8: %SubobjectDestroy.type = facet_value %C.920, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.094: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.0e8) [concrete]
// CHECK:STDOUT:   %Destroy.facet.bcb: %Destroy.type = facet_value %C.920, (%DestroyImpl.impl_witness.094, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2c2: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.bcb) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.190: %C.4bc = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.812: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.2fe [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c44: %ImplicitAs.type.164 = facet_value %C.4bc, (%ImplicitAs.impl_witness.812) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c0d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.c44) [concrete]
// CHECK:STDOUT:   %.1d8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c0d, %ImplicitAs.facet.c44 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.2: type = fn_type @C.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.2: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.9fe: <bound method> = bound_method %C.val.190, %C.as.ImplicitAs.impl.Convert.3cefb0.2 [concrete]
// CHECK:STDOUT:   %.3e0: ref %C.4bc = temporary invalid, %C.val.190 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.571: %SubobjectDestroy.type = facet_value %C.4bc, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a1e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.571) [concrete]
// CHECK:STDOUT:   %Destroy.facet.cf6: %Destroy.type = facet_value %C.4bc, (%DestroyImpl.impl_witness.a1e, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.036: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.cf6) [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a55: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.bfb: %C.6d6 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.25a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.179 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.dc7: %ImplicitAs.type.164 = facet_value %C.6d6, (%ImplicitAs.impl_witness.25a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.daf: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.dc7) [concrete]
// CHECK:STDOUT:   %.1282: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.daf, %ImplicitAs.facet.dc7 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.3: type = fn_type @C.as.ImplicitAs.impl.Convert.3 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.3: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.3 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.06f: <bound method> = bound_method %C.val.bfb, %C.as.ImplicitAs.impl.Convert.3cefb0.3 [concrete]
// CHECK:STDOUT:   %.88b: ref %C.6d6 = temporary invalid, %C.val.bfb [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.2e4: %SubobjectDestroy.type = facet_value %C.6d6, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a17: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.2e4) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b72: %Destroy.type = facet_value %C.6d6, (%DestroyImpl.impl_witness.a17, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.81c: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.b72) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.fb0: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.b62: %C.917 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c39: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d4e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.7d6: %ImplicitAs.type.164 = facet_value %C.917, (%ImplicitAs.impl_witness.c39) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f96: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.7d6) [concrete]
// CHECK:STDOUT:   %.104: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f96, %ImplicitAs.facet.7d6 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.4: type = fn_type @C.as.ImplicitAs.impl.Convert.4 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.4: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.4 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.e85: <bound method> = bound_method %C.val.b62, %C.as.ImplicitAs.impl.Convert.3cefb0.4 [concrete]
// CHECK:STDOUT:   %.295: ref %C.917 = temporary invalid, %C.val.b62 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c9c: %SubobjectDestroy.type = facet_value %C.917, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.70f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c9c) [concrete]
// CHECK:STDOUT:   %Destroy.facet.a9e: %Destroy.type = facet_value %C.917, (%DestroyImpl.impl_witness.70f, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.941: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.a9e) [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.776: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.7fa: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.c9e: %C.20d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9b1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e02 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.170: %ImplicitAs.type.164 = facet_value %C.20d, (%ImplicitAs.impl_witness.9b1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ff2: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.170) [concrete]
// CHECK:STDOUT:   %.ebf: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ff2, %ImplicitAs.facet.170 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.5: type = fn_type @C.as.ImplicitAs.impl.Convert.5 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.5: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.5 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.c6b: <bound method> = bound_method %C.val.c9e, %C.as.ImplicitAs.impl.Convert.3cefb0.5 [concrete]
// CHECK:STDOUT:   %.85b: ref %C.20d = temporary invalid, %C.val.c9e [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.69a: %SubobjectDestroy.type = facet_value %C.20d, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.194: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.69a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.1a2: %Destroy.type = facet_value %C.20d, (%DestroyImpl.impl_witness.194, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.73c: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.1a2) [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f65: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.ab2: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.7c2: %C.c2b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a1c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.3c3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.40a: %ImplicitAs.type.164 = facet_value %C.c2b, (%ImplicitAs.impl_witness.a1c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f4c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.40a) [concrete]
// CHECK:STDOUT:   %.d9e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f4c, %ImplicitAs.facet.40a [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.6: type = fn_type @C.as.ImplicitAs.impl.Convert.6 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.6: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.6 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.c22: <bound method> = bound_method %C.val.7c2, %C.as.ImplicitAs.impl.Convert.3cefb0.6 [concrete]
// CHECK:STDOUT:   %.441: ref %C.c2b = temporary invalid, %C.val.7c2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc12 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d99: %SubobjectDestroy.type = facet_value %C.c2b, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.279: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d99) [concrete]
// CHECK:STDOUT:   %Destroy.facet.abc: %Destroy.type = facet_value %C.c2b, (%DestroyImpl.impl_witness.279, %custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.c3a: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.abc) [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dcf: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.c9a: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.d14: %C.09c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ffb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9dd [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.0d8: %ImplicitAs.type.164 = facet_value %C.09c, (%ImplicitAs.impl_witness.ffb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f2: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.0d8) [concrete]
// CHECK:STDOUT:   %.a38: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f2, %ImplicitAs.facet.0d8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.7: type = fn_type @C.as.ImplicitAs.impl.Convert.7 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.7: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.7 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.569: <bound method> = bound_method %C.val.d14, %C.as.ImplicitAs.impl.Convert.3cefb0.7 [concrete]
// CHECK:STDOUT:   %.e0f: ref %C.09c = temporary invalid, %C.val.d14 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.298: %SubobjectDestroy.type = facet_value %C.09c, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.d65: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.298) [concrete]
// CHECK:STDOUT:   %Destroy.facet.d15: %Destroy.type = facet_value %C.09c, (%DestroyImpl.impl_witness.d65, %custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.019: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.d15) [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5f1: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a5e: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.015: %C.284 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0f6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e8f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.09a: %ImplicitAs.type.164 = facet_value %C.284, (%ImplicitAs.impl_witness.0f6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c50: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.09a) [concrete]
// CHECK:STDOUT:   %.583: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c50, %ImplicitAs.facet.09a [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.8: type = fn_type @C.as.ImplicitAs.impl.Convert.8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.8: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.8 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.076: <bound method> = bound_method %C.val.015, %C.as.ImplicitAs.impl.Convert.3cefb0.8 [concrete]
// CHECK:STDOUT:   %.5ca: ref %C.284 = temporary invalid, %C.val.015 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.641: %SubobjectDestroy.type = facet_value %C.284, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4b1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.641) [concrete]
// CHECK:STDOUT:   %Destroy.facet.97c: %Destroy.type = facet_value %C.284, (%DestroyImpl.impl_witness.4b1, %custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.b91: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.97c) [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
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
// CHECK:STDOUT:   %P.import_ref.f9a: <witness> = import_ref P//library, loc5_35, loaded [concrete = constants.%complete_type.28b]
// CHECK:STDOUT:   %P.import_ref.737 = import_ref P//library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.9ce = import_ref P//library, loc5_16, unloaded
// CHECK:STDOUT:   %P.import_ref.4cbb = import_ref P//library, loc5_28, unloaded
// CHECK:STDOUT:   %P.C: %C.type = import_ref P//library, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %P.import_ref.8f2: <witness> = import_ref P//library, loc4_18, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %P.import_ref.013 = import_ref P//library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.a68: %i32 = import_ref P//library, loc4_10, loaded [symbolic = @C.%N (constants.%N.c34)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %P.import_ref.66e: <witness> = import_ref P//library, loc8_33, loaded [concrete = constants.%ImplicitAs.impl_witness.0ca]
// CHECK:STDOUT:   %P.import_ref.58e: type = import_ref P//library, loc8_9, loaded [concrete = constants.%C.920]
// CHECK:STDOUT:   %P.import_ref.0d5cf9.1: type = import_ref P//library, loc8_31, loaded [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   %P.import_ref.41d: <witness> = import_ref P//library, loc9_33, loaded [concrete = constants.%ImplicitAs.impl_witness.812]
// CHECK:STDOUT:   %P.import_ref.e4b: type = import_ref P//library, loc9_9, loaded [concrete = constants.%C.4bc]
// CHECK:STDOUT:   %P.import_ref.0d5cf9.2: type = import_ref P//library, loc9_31, loaded [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   %P.import_ref.371: <witness> = import_ref P//library, loc10_33, loaded [concrete = constants.%ImplicitAs.impl_witness.25a]
// CHECK:STDOUT:   %P.import_ref.90b: type = import_ref P//library, loc10_9, loaded [concrete = constants.%C.6d6]
// CHECK:STDOUT:   %P.import_ref.0d5cf9.3: type = import_ref P//library, loc10_31, loaded [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   %P.import_ref.222: <witness> = import_ref P//library, loc11_33, loaded [concrete = constants.%ImplicitAs.impl_witness.c39]
// CHECK:STDOUT:   %P.import_ref.aff: type = import_ref P//library, loc11_9, loaded [concrete = constants.%C.917]
// CHECK:STDOUT:   %P.import_ref.0d5cf9.4: type = import_ref P//library, loc11_31, loaded [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   %P.import_ref.2b2: <witness> = import_ref P//library, loc12_33, loaded [concrete = constants.%ImplicitAs.impl_witness.9b1]
// CHECK:STDOUT:   %P.import_ref.d7f: type = import_ref P//library, loc12_9, loaded [concrete = constants.%C.20d]
// CHECK:STDOUT:   %P.import_ref.0d5cf9.5: type = import_ref P//library, loc12_31, loaded [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   %P.import_ref.e18: <witness> = import_ref P//library, loc13_33, loaded [concrete = constants.%ImplicitAs.impl_witness.a1c]
// CHECK:STDOUT:   %P.import_ref.7f0: type = import_ref P//library, loc13_9, loaded [concrete = constants.%C.c2b]
// CHECK:STDOUT:   %P.import_ref.0d5cf9.6: type = import_ref P//library, loc13_31, loaded [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   %P.import_ref.007: <witness> = import_ref P//library, loc14_33, loaded [concrete = constants.%ImplicitAs.impl_witness.ffb]
// CHECK:STDOUT:   %P.import_ref.5c9: type = import_ref P//library, loc14_9, loaded [concrete = constants.%C.09c]
// CHECK:STDOUT:   %P.import_ref.0d5cf9.7: type = import_ref P//library, loc14_31, loaded [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   %P.import_ref.ede: <witness> = import_ref P//library, loc15_33, loaded [concrete = constants.%ImplicitAs.impl_witness.0f6]
// CHECK:STDOUT:   %P.import_ref.9a7: type = import_ref P//library, loc15_9, loaded [concrete = constants.%C.284]
// CHECK:STDOUT:   %P.import_ref.0d5cf9.8: type = import_ref P//library, loc15_31, loaded [concrete = constants.%ImplicitAs.type.164]
// CHECK:STDOUT:   %P.import_ref.b89e2b.1: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = import_ref P//library, loc8_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.32b = impl_witness_table (%P.import_ref.b89e2b.1), @C.as.ImplicitAs.impl.e04 [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.2: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = import_ref P//library, loc9_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.2fe = impl_witness_table (%P.import_ref.b89e2b.2), @C.as.ImplicitAs.impl.c09 [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.3: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.3 = import_ref P//library, loc10_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.3]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.179 = impl_witness_table (%P.import_ref.b89e2b.3), @C.as.ImplicitAs.impl.cfd [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.4: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.4 = import_ref P//library, loc11_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.4]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d4e = impl_witness_table (%P.import_ref.b89e2b.4), @C.as.ImplicitAs.impl.fb3 [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.5: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.5 = import_ref P//library, loc12_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.5]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e02 = impl_witness_table (%P.import_ref.b89e2b.5), @C.as.ImplicitAs.impl.bc7 [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.6: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.6 = import_ref P//library, loc13_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.6]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.3c3 = impl_witness_table (%P.import_ref.b89e2b.6), @C.as.ImplicitAs.impl.fd5 [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.7: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.7 = import_ref P//library, loc14_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.7]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9dd = impl_witness_table (%P.import_ref.b89e2b.7), @C.as.ImplicitAs.impl.fc6 [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.8: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.8 = import_ref P//library, loc15_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.8]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e8f = impl_witness_table (%P.import_ref.b89e2b.8), @C.as.ImplicitAs.impl.f8b [concrete]
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
// CHECK:STDOUT:     %n.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%n.param_patt]
// CHECK:STDOUT:     %n.patt: %pattern_type.d94 = wrapper_binding_pattern n, %n.param_patt [concrete = constants.%n.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %P.ref.loc6: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%P.D [concrete = constants.%D]
// CHECK:STDOUT:     %.loc6_26.10: Core.Form = init_form %D.ref [concrete = constants.%.128d]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = wrapper_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.e04: imports.%P.import_ref.58e as imports.%P.import_ref.0d5cf9.1 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.66e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.c09: imports.%P.import_ref.e4b as imports.%P.import_ref.0d5cf9.2 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.41d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.cfd: imports.%P.import_ref.90b as imports.%P.import_ref.0d5cf9.3 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.371
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.fb3: imports.%P.import_ref.aff as imports.%P.import_ref.0d5cf9.4 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.222
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.bc7: imports.%P.import_ref.d7f as imports.%P.import_ref.0d5cf9.5 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.2b2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.fd5: imports.%P.import_ref.7f0 as imports.%P.import_ref.0d5cf9.6 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.e18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.fc6: imports.%P.import_ref.5c9 as imports.%P.import_ref.0d5cf9.7 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.007
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.f8b: imports.%P.import_ref.9a7 as imports.%P.import_ref.0d5cf9.8 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.ede
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "library.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%P.import_ref.f9a
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%P.import_ref.737
// CHECK:STDOUT:   .n = imports.%P.import_ref.9ce
// CHECK:STDOUT:   .m = imports.%P.import_ref.4cbb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%P.import_ref.a68: %i32) [from "library.carbon"] {
// CHECK:STDOUT:   %N.patt: %pattern_type.d94 = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.d64)]
// CHECK:STDOUT:   %N: %i32 = symbolic_binding N, 0 [symbolic = %N (constants.%N.c34)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%P.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%P.import_ref.013
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
// CHECK:STDOUT:   %impl.elem0.loc7_34: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc7_34.1: <bound method> = bound_method %int_0, %impl.elem0.loc7_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.762]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_34.2: <bound method> = bound_method %int_0, %specific_fn.loc7 [concrete = constants.%bound_method.036]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_34.2(%int_0) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc7_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc7_34.2: %i32 = converted %int_0, %.loc7_34.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %C.loc7: type = class_type @C, @C(constants.%int_0.753) [concrete = constants.%C.920]
// CHECK:STDOUT:   %.loc7_24.2: ref %C.920 = temporary_storage
// CHECK:STDOUT:   %.loc7_24.3: init %C.920 to %.loc7_24.2 = class_init () [concrete = constants.%C.val.95f]
// CHECK:STDOUT:   %.loc7_26.1: init %C.920 = converted %.loc7_24.1, %.loc7_24.3 [concrete = constants.%C.val.95f]
// CHECK:STDOUT:   %impl.elem0.loc7_35: %.5c5 = impl_witness_access constants.%ImplicitAs.impl_witness.0ca, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %bound_method.loc7_35: <bound method> = bound_method %.loc7_26.1, %impl.elem0.loc7_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.2ac]
// CHECK:STDOUT:   %.loc6_26.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc7_26.2: ref %C.920 = temporary %.loc7_24.2, %.loc7_26.1 [concrete = constants.%.2eb]
// CHECK:STDOUT:   %.loc7_26.3: %C.920 = acquire_value %.loc7_26.2 [concrete = constants.%C.val.95f]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc7: init %D to %.loc6_26.1 = call %bound_method.loc7_35(%.loc7_26.3)
// CHECK:STDOUT:   %.loc7_35: init %D = converted %.loc7_26.1, %C.as.ImplicitAs.impl.Convert.call.loc7
// CHECK:STDOUT:   %Destroy.facet.loc7_26.1: %Destroy.type = facet_value constants.%C.920, (constants.%DestroyImpl.impl_witness.094, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.bcb]
// CHECK:STDOUT:   %.loc7_26.4: %Destroy.type = converted constants.%C.920, %Destroy.facet.loc7_26.1 [concrete = constants.%Destroy.facet.bcb]
// CHECK:STDOUT:   %Destroy.facet.loc7_26.2: %Destroy.type = facet_value constants.%C.920, (constants.%DestroyImpl.impl_witness.094, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.bcb]
// CHECK:STDOUT:   %.loc7_26.5: %Destroy.type = converted constants.%C.920, %Destroy.facet.loc7_26.2 [concrete = constants.%Destroy.facet.bcb]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.bcb) [concrete = constants.%SelfDestruct.specific_fn.2c2]
// CHECK:STDOUT:   %SelfDestruct.call.loc7: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc7_26.2)
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
// CHECK:STDOUT:   %impl.elem0.loc8_34: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_1, %specific_fn.loc8 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_34.2(%int_1) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_1, %.loc8_34.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %C.loc8: type = class_type @C, @C(constants.%int_1.68a) [concrete = constants.%C.4bc]
// CHECK:STDOUT:   %.loc8_24.2: ref %C.4bc = temporary_storage
// CHECK:STDOUT:   %.loc8_24.3: init %C.4bc to %.loc8_24.2 = class_init () [concrete = constants.%C.val.190]
// CHECK:STDOUT:   %.loc8_26.1: init %C.4bc = converted %.loc8_24.1, %.loc8_24.3 [concrete = constants.%C.val.190]
// CHECK:STDOUT:   %impl.elem0.loc8_35: %.1d8 = impl_witness_access constants.%ImplicitAs.impl_witness.812, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:   %bound_method.loc8_35: <bound method> = bound_method %.loc8_26.1, %impl.elem0.loc8_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.9fe]
// CHECK:STDOUT:   %.loc6_26.2: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc8_26.2: ref %C.4bc = temporary %.loc8_24.2, %.loc8_26.1 [concrete = constants.%.3e0]
// CHECK:STDOUT:   %.loc8_26.3: %C.4bc = acquire_value %.loc8_26.2 [concrete = constants.%C.val.190]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc8: init %D to %.loc6_26.2 = call %bound_method.loc8_35(%.loc8_26.3)
// CHECK:STDOUT:   %.loc8_35: init %D = converted %.loc8_26.1, %C.as.ImplicitAs.impl.Convert.call.loc8
// CHECK:STDOUT:   %Destroy.facet.loc8_26.1: %Destroy.type = facet_value constants.%C.4bc, (constants.%DestroyImpl.impl_witness.a1e, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.cf6]
// CHECK:STDOUT:   %.loc8_26.4: %Destroy.type = converted constants.%C.4bc, %Destroy.facet.loc8_26.1 [concrete = constants.%Destroy.facet.cf6]
// CHECK:STDOUT:   %Destroy.facet.loc8_26.2: %Destroy.type = facet_value constants.%C.4bc, (constants.%DestroyImpl.impl_witness.a1e, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.cf6]
// CHECK:STDOUT:   %.loc8_26.5: %Destroy.type = converted constants.%C.4bc, %Destroy.facet.loc8_26.2 [concrete = constants.%Destroy.facet.cf6]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.cf6) [concrete = constants.%SelfDestruct.specific_fn.036]
// CHECK:STDOUT:   %SelfDestruct.call.loc8: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc8_26.2)
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
// CHECK:STDOUT:   %impl.elem0.loc9_34: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %int_2, %impl.elem0.loc9_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %int_2, %specific_fn.loc9 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_34.2(%int_2) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc9_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc9_34.2: %i32 = converted %int_2, %.loc9_34.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %C.loc9: type = class_type @C, @C(constants.%int_2.2e0) [concrete = constants.%C.6d6]
// CHECK:STDOUT:   %.loc9_24.2: ref %C.6d6 = temporary_storage
// CHECK:STDOUT:   %.loc9_24.3: init %C.6d6 to %.loc9_24.2 = class_init () [concrete = constants.%C.val.bfb]
// CHECK:STDOUT:   %.loc9_26.1: init %C.6d6 = converted %.loc9_24.1, %.loc9_24.3 [concrete = constants.%C.val.bfb]
// CHECK:STDOUT:   %impl.elem0.loc9_35: %.1282 = impl_witness_access constants.%ImplicitAs.impl_witness.25a, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.3]
// CHECK:STDOUT:   %bound_method.loc9_35: <bound method> = bound_method %.loc9_26.1, %impl.elem0.loc9_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.06f]
// CHECK:STDOUT:   %.loc6_26.3: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc9_26.2: ref %C.6d6 = temporary %.loc9_24.2, %.loc9_26.1 [concrete = constants.%.88b]
// CHECK:STDOUT:   %.loc9_26.3: %C.6d6 = acquire_value %.loc9_26.2 [concrete = constants.%C.val.bfb]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc9: init %D to %.loc6_26.3 = call %bound_method.loc9_35(%.loc9_26.3)
// CHECK:STDOUT:   %.loc9_35: init %D = converted %.loc9_26.1, %C.as.ImplicitAs.impl.Convert.call.loc9
// CHECK:STDOUT:   %Destroy.facet.loc9_26.1: %Destroy.type = facet_value constants.%C.6d6, (constants.%DestroyImpl.impl_witness.a17, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.b72]
// CHECK:STDOUT:   %.loc9_26.4: %Destroy.type = converted constants.%C.6d6, %Destroy.facet.loc9_26.1 [concrete = constants.%Destroy.facet.b72]
// CHECK:STDOUT:   %Destroy.facet.loc9_26.2: %Destroy.type = facet_value constants.%C.6d6, (constants.%DestroyImpl.impl_witness.a17, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.b72]
// CHECK:STDOUT:   %.loc9_26.5: %Destroy.type = converted constants.%C.6d6, %Destroy.facet.loc9_26.2 [concrete = constants.%Destroy.facet.b72]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.b72) [concrete = constants.%SelfDestruct.specific_fn.81c]
// CHECK:STDOUT:   %SelfDestruct.call.loc9: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc9_26.2)
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
// CHECK:STDOUT:   %impl.elem0.loc10_34: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc10_34.1: <bound method> = bound_method %int_3, %impl.elem0.loc10_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_34.2: <bound method> = bound_method %int_3, %specific_fn.loc10 [concrete = constants.%bound_method.fb0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %i32 = call %bound_method.loc10_34.2(%int_3) [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc10_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc10_34.2: %i32 = converted %int_3, %.loc10_34.1 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %C.loc10: type = class_type @C, @C(constants.%int_3.9f3) [concrete = constants.%C.917]
// CHECK:STDOUT:   %.loc10_24.2: ref %C.917 = temporary_storage
// CHECK:STDOUT:   %.loc10_24.3: init %C.917 to %.loc10_24.2 = class_init () [concrete = constants.%C.val.b62]
// CHECK:STDOUT:   %.loc10_26.1: init %C.917 = converted %.loc10_24.1, %.loc10_24.3 [concrete = constants.%C.val.b62]
// CHECK:STDOUT:   %impl.elem0.loc10_35: %.104 = impl_witness_access constants.%ImplicitAs.impl_witness.c39, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.4]
// CHECK:STDOUT:   %bound_method.loc10_35: <bound method> = bound_method %.loc10_26.1, %impl.elem0.loc10_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.e85]
// CHECK:STDOUT:   %.loc6_26.4: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc10_26.2: ref %C.917 = temporary %.loc10_24.2, %.loc10_26.1 [concrete = constants.%.295]
// CHECK:STDOUT:   %.loc10_26.3: %C.917 = acquire_value %.loc10_26.2 [concrete = constants.%C.val.b62]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc10: init %D to %.loc6_26.4 = call %bound_method.loc10_35(%.loc10_26.3)
// CHECK:STDOUT:   %.loc10_35: init %D = converted %.loc10_26.1, %C.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %Destroy.facet.loc10_26.1: %Destroy.type = facet_value constants.%C.917, (constants.%DestroyImpl.impl_witness.70f, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.a9e]
// CHECK:STDOUT:   %.loc10_26.4: %Destroy.type = converted constants.%C.917, %Destroy.facet.loc10_26.1 [concrete = constants.%Destroy.facet.a9e]
// CHECK:STDOUT:   %Destroy.facet.loc10_26.2: %Destroy.type = facet_value constants.%C.917, (constants.%DestroyImpl.impl_witness.70f, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.a9e]
// CHECK:STDOUT:   %.loc10_26.5: %Destroy.type = converted constants.%C.917, %Destroy.facet.loc10_26.2 [concrete = constants.%Destroy.facet.a9e]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.4: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.a9e) [concrete = constants.%SelfDestruct.specific_fn.941]
// CHECK:STDOUT:   %SelfDestruct.call.loc10: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc10_26.2)
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
// CHECK:STDOUT:   %impl.elem0.loc11_34: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc11_34.1: <bound method> = bound_method %int_4, %impl.elem0.loc11_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.776]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_34.2: <bound method> = bound_method %int_4, %specific_fn.loc11 [concrete = constants.%bound_method.7fa]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %i32 = call %bound_method.loc11_34.2(%int_4) [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:   %.loc11_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:   %.loc11_34.2: %i32 = converted %int_4, %.loc11_34.1 [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:   %C.loc11: type = class_type @C, @C(constants.%int_4.6bf) [concrete = constants.%C.20d]
// CHECK:STDOUT:   %.loc11_24.2: ref %C.20d = temporary_storage
// CHECK:STDOUT:   %.loc11_24.3: init %C.20d to %.loc11_24.2 = class_init () [concrete = constants.%C.val.c9e]
// CHECK:STDOUT:   %.loc11_26.1: init %C.20d = converted %.loc11_24.1, %.loc11_24.3 [concrete = constants.%C.val.c9e]
// CHECK:STDOUT:   %impl.elem0.loc11_35: %.ebf = impl_witness_access constants.%ImplicitAs.impl_witness.9b1, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.5]
// CHECK:STDOUT:   %bound_method.loc11_35: <bound method> = bound_method %.loc11_26.1, %impl.elem0.loc11_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.c6b]
// CHECK:STDOUT:   %.loc6_26.5: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc11_26.2: ref %C.20d = temporary %.loc11_24.2, %.loc11_26.1 [concrete = constants.%.85b]
// CHECK:STDOUT:   %.loc11_26.3: %C.20d = acquire_value %.loc11_26.2 [concrete = constants.%C.val.c9e]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc11: init %D to %.loc6_26.5 = call %bound_method.loc11_35(%.loc11_26.3)
// CHECK:STDOUT:   %.loc11_35: init %D = converted %.loc11_26.1, %C.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %Destroy.facet.loc11_26.1: %Destroy.type = facet_value constants.%C.20d, (constants.%DestroyImpl.impl_witness.194, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.1a2]
// CHECK:STDOUT:   %.loc11_26.4: %Destroy.type = converted constants.%C.20d, %Destroy.facet.loc11_26.1 [concrete = constants.%Destroy.facet.1a2]
// CHECK:STDOUT:   %Destroy.facet.loc11_26.2: %Destroy.type = facet_value constants.%C.20d, (constants.%DestroyImpl.impl_witness.194, constants.%custom_witness.b80ef6.6) [concrete = constants.%Destroy.facet.1a2]
// CHECK:STDOUT:   %.loc11_26.5: %Destroy.type = converted constants.%C.20d, %Destroy.facet.loc11_26.2 [concrete = constants.%Destroy.facet.1a2]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.1a2) [concrete = constants.%SelfDestruct.specific_fn.73c]
// CHECK:STDOUT:   %SelfDestruct.call.loc11: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc11_26.2)
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
// CHECK:STDOUT:   %impl.elem0.loc12_34: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc12_34.1: <bound method> = bound_method %int_5, %impl.elem0.loc12_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f65]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_34.2: <bound method> = bound_method %int_5, %specific_fn.loc12 [concrete = constants.%bound_method.ab2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %i32 = call %bound_method.loc12_34.2(%int_5) [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %.loc12_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %.loc12_34.2: %i32 = converted %int_5, %.loc12_34.1 [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %C.loc12: type = class_type @C, @C(constants.%int_5.708) [concrete = constants.%C.c2b]
// CHECK:STDOUT:   %.loc12_24.2: ref %C.c2b = temporary_storage
// CHECK:STDOUT:   %.loc12_24.3: init %C.c2b to %.loc12_24.2 = class_init () [concrete = constants.%C.val.7c2]
// CHECK:STDOUT:   %.loc12_26.1: init %C.c2b = converted %.loc12_24.1, %.loc12_24.3 [concrete = constants.%C.val.7c2]
// CHECK:STDOUT:   %impl.elem0.loc12_35: %.d9e = impl_witness_access constants.%ImplicitAs.impl_witness.a1c, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.6]
// CHECK:STDOUT:   %bound_method.loc12_35: <bound method> = bound_method %.loc12_26.1, %impl.elem0.loc12_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.c22]
// CHECK:STDOUT:   %.loc6_26.6: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc12_26.2: ref %C.c2b = temporary %.loc12_24.2, %.loc12_26.1 [concrete = constants.%.441]
// CHECK:STDOUT:   %.loc12_26.3: %C.c2b = acquire_value %.loc12_26.2 [concrete = constants.%C.val.7c2]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc12: init %D to %.loc6_26.6 = call %bound_method.loc12_35(%.loc12_26.3)
// CHECK:STDOUT:   %.loc12_35: init %D = converted %.loc12_26.1, %C.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %Destroy.facet.loc12_26.1: %Destroy.type = facet_value constants.%C.c2b, (constants.%DestroyImpl.impl_witness.279, constants.%custom_witness.b80ef6.7) [concrete = constants.%Destroy.facet.abc]
// CHECK:STDOUT:   %.loc12_26.4: %Destroy.type = converted constants.%C.c2b, %Destroy.facet.loc12_26.1 [concrete = constants.%Destroy.facet.abc]
// CHECK:STDOUT:   %Destroy.facet.loc12_26.2: %Destroy.type = facet_value constants.%C.c2b, (constants.%DestroyImpl.impl_witness.279, constants.%custom_witness.b80ef6.7) [concrete = constants.%Destroy.facet.abc]
// CHECK:STDOUT:   %.loc12_26.5: %Destroy.type = converted constants.%C.c2b, %Destroy.facet.loc12_26.2 [concrete = constants.%Destroy.facet.abc]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.abc) [concrete = constants.%SelfDestruct.specific_fn.c3a]
// CHECK:STDOUT:   %SelfDestruct.call.loc12: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%.loc12_26.2)
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
// CHECK:STDOUT:   %impl.elem0.loc13_34: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc13_34.1: <bound method> = bound_method %int_6, %impl.elem0.loc13_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dcf]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_34.2: <bound method> = bound_method %int_6, %specific_fn.loc13 [concrete = constants.%bound_method.c9a]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %i32 = call %bound_method.loc13_34.2(%int_6) [concrete = constants.%int_6.b33]
// CHECK:STDOUT:   %.loc13_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%int_6.b33]
// CHECK:STDOUT:   %.loc13_34.2: %i32 = converted %int_6, %.loc13_34.1 [concrete = constants.%int_6.b33]
// CHECK:STDOUT:   %C.loc13: type = class_type @C, @C(constants.%int_6.b33) [concrete = constants.%C.09c]
// CHECK:STDOUT:   %.loc13_24.2: ref %C.09c = temporary_storage
// CHECK:STDOUT:   %.loc13_24.3: init %C.09c to %.loc13_24.2 = class_init () [concrete = constants.%C.val.d14]
// CHECK:STDOUT:   %.loc13_26.1: init %C.09c = converted %.loc13_24.1, %.loc13_24.3 [concrete = constants.%C.val.d14]
// CHECK:STDOUT:   %impl.elem0.loc13_35: %.a38 = impl_witness_access constants.%ImplicitAs.impl_witness.ffb, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.7]
// CHECK:STDOUT:   %bound_method.loc13_35: <bound method> = bound_method %.loc13_26.1, %impl.elem0.loc13_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.569]
// CHECK:STDOUT:   %.loc6_26.7: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc13_26.2: ref %C.09c = temporary %.loc13_24.2, %.loc13_26.1 [concrete = constants.%.e0f]
// CHECK:STDOUT:   %.loc13_26.3: %C.09c = acquire_value %.loc13_26.2 [concrete = constants.%C.val.d14]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc13: init %D to %.loc6_26.7 = call %bound_method.loc13_35(%.loc13_26.3)
// CHECK:STDOUT:   %.loc13_35: init %D = converted %.loc13_26.1, %C.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %Destroy.facet.loc13_26.1: %Destroy.type = facet_value constants.%C.09c, (constants.%DestroyImpl.impl_witness.d65, constants.%custom_witness.b80ef6.8) [concrete = constants.%Destroy.facet.d15]
// CHECK:STDOUT:   %.loc13_26.4: %Destroy.type = converted constants.%C.09c, %Destroy.facet.loc13_26.1 [concrete = constants.%Destroy.facet.d15]
// CHECK:STDOUT:   %Destroy.facet.loc13_26.2: %Destroy.type = facet_value constants.%C.09c, (constants.%DestroyImpl.impl_witness.d65, constants.%custom_witness.b80ef6.8) [concrete = constants.%Destroy.facet.d15]
// CHECK:STDOUT:   %.loc13_26.5: %Destroy.type = converted constants.%C.09c, %Destroy.facet.loc13_26.2 [concrete = constants.%Destroy.facet.d15]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.7: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.d15) [concrete = constants.%SelfDestruct.specific_fn.019]
// CHECK:STDOUT:   %SelfDestruct.call.loc13: init %empty_tuple.type = call %SelfDestruct.specific_fn.7(%.loc13_26.2)
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
// CHECK:STDOUT:   %impl.elem0.loc14_34: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc14_34.1: <bound method> = bound_method %int_7, %impl.elem0.loc14_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5f1]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.2: <bound method> = bound_method %int_7, %specific_fn.loc14 [concrete = constants.%bound_method.a5e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14: init %i32 = call %bound_method.loc14_34.2(%int_7) [concrete = constants.%int_7.a13]
// CHECK:STDOUT:   %.loc14_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14 [concrete = constants.%int_7.a13]
// CHECK:STDOUT:   %.loc14_34.2: %i32 = converted %int_7, %.loc14_34.1 [concrete = constants.%int_7.a13]
// CHECK:STDOUT:   %C.loc14: type = class_type @C, @C(constants.%int_7.a13) [concrete = constants.%C.284]
// CHECK:STDOUT:   %.loc14_24.2: ref %C.284 = temporary_storage
// CHECK:STDOUT:   %.loc14_24.3: init %C.284 to %.loc14_24.2 = class_init () [concrete = constants.%C.val.015]
// CHECK:STDOUT:   %.loc14_26.1: init %C.284 = converted %.loc14_24.1, %.loc14_24.3 [concrete = constants.%C.val.015]
// CHECK:STDOUT:   %impl.elem0.loc14_35: %.583 = impl_witness_access constants.%ImplicitAs.impl_witness.0f6, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.8]
// CHECK:STDOUT:   %bound_method.loc14_35: <bound method> = bound_method %.loc14_26.1, %impl.elem0.loc14_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.076]
// CHECK:STDOUT:   %.loc6_26.8: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc14_26.2: ref %C.284 = temporary %.loc14_24.2, %.loc14_26.1 [concrete = constants.%.5ca]
// CHECK:STDOUT:   %.loc14_26.3: %C.284 = acquire_value %.loc14_26.2 [concrete = constants.%C.val.015]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc14: init %D to %.loc6_26.8 = call %bound_method.loc14_35(%.loc14_26.3)
// CHECK:STDOUT:   %.loc14_35: init %D = converted %.loc14_26.1, %C.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %Destroy.facet.loc14_26.1: %Destroy.type = facet_value constants.%C.284, (constants.%DestroyImpl.impl_witness.4b1, constants.%custom_witness.b80ef6.9) [concrete = constants.%Destroy.facet.97c]
// CHECK:STDOUT:   %.loc14_26.4: %Destroy.type = converted constants.%C.284, %Destroy.facet.loc14_26.1 [concrete = constants.%Destroy.facet.97c]
// CHECK:STDOUT:   %Destroy.facet.loc14_26.2: %Destroy.type = facet_value constants.%C.284, (constants.%DestroyImpl.impl_witness.4b1, constants.%custom_witness.b80ef6.9) [concrete = constants.%Destroy.facet.97c]
// CHECK:STDOUT:   %.loc14_26.5: %Destroy.type = converted constants.%C.284, %Destroy.facet.loc14_26.2 [concrete = constants.%Destroy.facet.97c]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.8: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet.97c) [concrete = constants.%SelfDestruct.specific_fn.b91]
// CHECK:STDOUT:   %SelfDestruct.call.loc14: init %empty_tuple.type = call %SelfDestruct.specific_fn.8(%.loc14_26.2)
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_26.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_26.2(%self.param: ref %C.920) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.2 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %C.4bc) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.3 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref %C.6d6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.4 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %C.917) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.5 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11(%self.param: ref %C.20d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.6 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc12(%self.param: ref %C.c2b) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.7 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13(%self.param: ref %C.09c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.8 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14(%self.param: ref %C.284) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.c34) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.d64
// CHECK:STDOUT:   %N => constants.%N.c34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.753) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.d64
// CHECK:STDOUT:   %N => constants.%int_0.753
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.68a) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.d64
// CHECK:STDOUT:   %N => constants.%int_1.68a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.2e0) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.d64
// CHECK:STDOUT:   %N => constants.%int_2.2e0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.9f3) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.d64
// CHECK:STDOUT:   %N => constants.%int_3.9f3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.6bf) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.d64
// CHECK:STDOUT:   %N => constants.%int_4.6bf
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.708) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.d64
// CHECK:STDOUT:   %N => constants.%int_5.708
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.b33) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.d64
// CHECK:STDOUT:   %N => constants.%int_6.b33
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.a13) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.d64
// CHECK:STDOUT:   %N => constants.%int_7.a13
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
