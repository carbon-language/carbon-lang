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
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.282: %pattern_type.df9 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.06f: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.67b: type = class_type @C, @C(%N.06f) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.5e7: type = struct_type {.n: %i32, .m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.8d0: <witness> = complete_type_witness %struct_type.n.m.5e7 [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e87: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.186: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.604: %ImplicitAs.type.cb7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e87) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.604) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac8, %ImplicitAs.facet.604 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1da: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.be3: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.1b1: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_0.1b1, %int_0.1b1) [concrete]
// CHECK:STDOUT:   %C.7ad: type = class_type @C, @C(%int_0.1b1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.24d: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %.df1: <witness> = impl_self_witness %C.7ad, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0ff: <witness> = impl_witness @C.as.ImplicitAs.impl.dc4.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.f86: type = pattern_type %C.7ad [concrete]
// CHECK:STDOUT:   %self.param_patt.95c: %pattern_type.f86 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.bab: %pattern_type.f86 = wrapper_binding_pattern self, %self.param_patt.95c [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.1: type = fn_type @C.as.ImplicitAs.impl.Convert.loc8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.1: %C.as.ImplicitAs.impl.Convert.type.860b0a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a5f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.67d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.874: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.dca: type = class_type @C, @C(%int_1.874) [concrete]
// CHECK:STDOUT:   %.879: <witness> = impl_self_witness %C.dca, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f24: <witness> = impl_witness @C.as.ImplicitAs.impl.b38.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.0aa: type = pattern_type %C.dca [concrete]
// CHECK:STDOUT:   %self.param_patt.ba5: %pattern_type.0aa = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.656: %pattern_type.0aa = wrapper_binding_pattern self, %self.param_patt.ba5 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.2: type = fn_type @C.as.ImplicitAs.impl.Convert.loc9 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.2: %C.as.ImplicitAs.impl.Convert.type.860b0a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.ca1: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.92b: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.e54: type = class_type @C, @C(%int_2.92b) [concrete]
// CHECK:STDOUT:   %.7fa: <witness> = impl_self_witness %C.e54, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c05: <witness> = impl_witness @C.as.ImplicitAs.impl.510.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9e8: type = pattern_type %C.e54 [concrete]
// CHECK:STDOUT:   %self.param_patt.e13: %pattern_type.9e8 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.db8: %pattern_type.9e8 = wrapper_binding_pattern self, %self.param_patt.e13 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.3: type = fn_type @C.as.ImplicitAs.impl.Convert.loc10 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.3: %C.as.ImplicitAs.impl.Convert.type.860b0a.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d7b: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.690: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.0aa: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.f6f: type = class_type @C, @C(%int_3.0aa) [concrete]
// CHECK:STDOUT:   %.e54: <witness> = impl_self_witness %C.f6f, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.14e: <witness> = impl_witness @C.as.ImplicitAs.impl.7ca.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.83d: type = pattern_type %C.f6f [concrete]
// CHECK:STDOUT:   %self.param_patt.57d: %pattern_type.83d = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.f9c: %pattern_type.83d = wrapper_binding_pattern self, %self.param_patt.57d [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.4: type = fn_type @C.as.ImplicitAs.impl.Convert.loc11 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.4: %C.as.ImplicitAs.impl.Convert.type.860b0a.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7be: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.a39: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.bdd: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %C.77a: type = class_type @C, @C(%int_4.bdd) [concrete]
// CHECK:STDOUT:   %.b19: <witness> = impl_self_witness %C.77a, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.727: <witness> = impl_witness @C.as.ImplicitAs.impl.cc6.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.557: type = pattern_type %C.77a [concrete]
// CHECK:STDOUT:   %self.param_patt.792: %pattern_type.557 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.5ed: %pattern_type.557 = wrapper_binding_pattern self, %self.param_patt.792 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.5: type = fn_type @C.as.ImplicitAs.impl.Convert.loc12 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.5: %C.as.ImplicitAs.impl.Convert.type.860b0a.5 = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a32: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.954: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.ff2: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %C.bd8: type = class_type @C, @C(%int_5.ff2) [concrete]
// CHECK:STDOUT:   %.00b: <witness> = impl_self_witness %C.bd8, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.648: <witness> = impl_witness @C.as.ImplicitAs.impl.def.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.5ad: type = pattern_type %C.bd8 [concrete]
// CHECK:STDOUT:   %self.param_patt.b99: %pattern_type.5ad = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.fff: %pattern_type.5ad = wrapper_binding_pattern self, %self.param_patt.b99 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.6: type = fn_type @C.as.ImplicitAs.impl.Convert.loc13 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.6: %C.as.ImplicitAs.impl.Convert.type.860b0a.6 = struct_value () [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.552: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.a13: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.47a: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %C.a98: type = class_type @C, @C(%int_6.47a) [concrete]
// CHECK:STDOUT:   %.f3f: <witness> = impl_self_witness %C.a98, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.490: <witness> = impl_witness @C.as.ImplicitAs.impl.c5d.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.e13: type = pattern_type %C.a98 [concrete]
// CHECK:STDOUT:   %self.param_patt.187: %pattern_type.e13 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.8b8: %pattern_type.e13 = wrapper_binding_pattern self, %self.param_patt.187 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.860b0a.7: type = fn_type @C.as.ImplicitAs.impl.Convert.loc14 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.54620a.7: %C.as.ImplicitAs.impl.Convert.type.860b0a.7 = struct_value () [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.828: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.aaa: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.367: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %C.bc5: type = class_type @C, @C(%int_7.367) [concrete]
// CHECK:STDOUT:   %.126: <witness> = impl_self_witness %C.bc5, @ImplicitAs, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.4ef: <witness> = impl_witness @C.as.ImplicitAs.impl.2e9.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.98d: type = pattern_type %C.bc5 [concrete]
// CHECK:STDOUT:   %self.param_patt.fca: %pattern_type.98d = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.ca9: %pattern_type.98d = wrapper_binding_pattern self, %self.param_patt.fca [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %N.patt.loc4_10.1: %pattern_type.df9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_10.2 (constants.%N.patt.282)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_10.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_10.1 (constants.%N.06f)]
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
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.dc4 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:     %bound_method.loc8_9.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1da]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_9.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.be3]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_9.2(%int_0) [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:     %.loc8_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:     %.loc8_9.2: %i32 = converted %int_0, %.loc8_9.1 [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_0.1b1) [concrete = constants.%C.7ad]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = impl_self_witness @C.as.ImplicitAs.impl.dc4.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.df1]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.b38 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %impl.elem0: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:     %bound_method.loc9_9.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a5f]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_9.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method.67d]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_9.2(%int_1) [concrete = constants.%int_1.874]
// CHECK:STDOUT:     %.loc9_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.874]
// CHECK:STDOUT:     %.loc9_9.2: %i32 = converted %int_1, %.loc9_9.1 [concrete = constants.%int_1.874]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_1.874) [concrete = constants.%C.dca]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: <witness> = impl_self_witness @C.as.ImplicitAs.impl.b38.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.879]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.510 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:     %bound_method.loc10_9.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_9.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.ca1]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc10_9.2(%int_2) [concrete = constants.%int_2.92b]
// CHECK:STDOUT:     %.loc10_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.92b]
// CHECK:STDOUT:     %.loc10_9.2: %i32 = converted %int_2, %.loc10_9.1 [concrete = constants.%int_2.92b]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_2.92b) [concrete = constants.%C.e54]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <witness> = impl_self_witness @C.as.ImplicitAs.impl.510.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.7fa]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.7ca [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:     %bound_method.loc11_9.1: <bound method> = bound_method %int_3, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d7b]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_9.2: <bound method> = bound_method %int_3, %specific_fn [concrete = constants.%bound_method.690]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_9.2(%int_3) [concrete = constants.%int_3.0aa]
// CHECK:STDOUT:     %.loc11_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.0aa]
// CHECK:STDOUT:     %.loc11_9.2: %i32 = converted %int_3, %.loc11_9.1 [concrete = constants.%int_3.0aa]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_3.0aa) [concrete = constants.%C.f6f]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <witness> = impl_self_witness @C.as.ImplicitAs.impl.7ca.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.e54]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.cc6 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %impl.elem0: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:     %bound_method.loc12_9.1: <bound method> = bound_method %int_4, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7be]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc12_9.2: <bound method> = bound_method %int_4, %specific_fn [concrete = constants.%bound_method.a39]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_9.2(%int_4) [concrete = constants.%int_4.bdd]
// CHECK:STDOUT:     %.loc12_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_4.bdd]
// CHECK:STDOUT:     %.loc12_9.2: %i32 = converted %int_4, %.loc12_9.1 [concrete = constants.%int_4.bdd]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_4.bdd) [concrete = constants.%C.77a]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: <witness> = impl_self_witness @C.as.ImplicitAs.impl.cc6.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.b19]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.def [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:     %impl.elem0: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:     %bound_method.loc13_9.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a32]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc13_9.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method.954]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_9.2(%int_5) [concrete = constants.%int_5.ff2]
// CHECK:STDOUT:     %.loc13_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.ff2]
// CHECK:STDOUT:     %.loc13_9.2: %i32 = converted %int_5, %.loc13_9.1 [concrete = constants.%int_5.ff2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_5.ff2) [concrete = constants.%C.bd8]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = impl_self_witness @C.as.ImplicitAs.impl.def.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.00b]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.c5d [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:     %impl.elem0: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %int_6, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.552]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %int_6, %specific_fn [concrete = constants.%bound_method.a13]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc14_9.2(%int_6) [concrete = constants.%int_6.47a]
// CHECK:STDOUT:     %.loc14_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_6.47a]
// CHECK:STDOUT:     %.loc14_9.2: %i32 = converted %int_6, %.loc14_9.1 [concrete = constants.%int_6.47a]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_6.47a) [concrete = constants.%C.a98]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14: <witness> = impl_self_witness @C.as.ImplicitAs.impl.c5d.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.f3f]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.2e9 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:     %impl.elem0: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:     %bound_method.loc15_9.1: <bound method> = bound_method %int_7, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.828]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc15_9.2: <bound method> = bound_method %int_7, %specific_fn [concrete = constants.%bound_method.aaa]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_9.2(%int_7) [concrete = constants.%int_7.367]
// CHECK:STDOUT:     %.loc15_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_7.367]
// CHECK:STDOUT:     %.loc15_9.2: %i32 = converted %int_7, %.loc15_9.1 [concrete = constants.%int_7.367]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_7.367) [concrete = constants.%C.bc5]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: <witness> = impl_self_witness @C.as.ImplicitAs.impl.2e9.%C, @ImplicitAs, @ImplicitAs(constants.%D) [concrete = constants.%.126]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.dc4: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.1 = fn_decl @C.as.ImplicitAs.impl.Convert.loc8 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f86 = value_param_pattern [concrete = constants.%self.param_patt.95c]
// CHECK:STDOUT:     %self.patt: %pattern_type.f86 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bab]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc8_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.7ad = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.dc4.%C [concrete = constants.%C.7ad]
// CHECK:STDOUT:     %self: %C.7ad = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.dc4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.0ff]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.b38: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.2 = fn_decl @C.as.ImplicitAs.impl.Convert.loc9 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0aa = value_param_pattern [concrete = constants.%self.param_patt.ba5]
// CHECK:STDOUT:     %self.patt: %pattern_type.0aa = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.656]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc9_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.dca = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.b38.%C [concrete = constants.%C.dca]
// CHECK:STDOUT:     %self: %C.dca = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.b38 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.f24]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.510: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.3 = fn_decl @C.as.ImplicitAs.impl.Convert.loc10 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9e8 = value_param_pattern [concrete = constants.%self.param_patt.e13]
// CHECK:STDOUT:     %self.patt: %pattern_type.9e8 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.db8]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc10_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.e54 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.510.%C [concrete = constants.%C.e54]
// CHECK:STDOUT:     %self: %C.e54 = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.510 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.c05]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.7ca: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.4 = fn_decl @C.as.ImplicitAs.impl.Convert.loc11 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.83d = value_param_pattern [concrete = constants.%self.param_patt.57d]
// CHECK:STDOUT:     %self.patt: %pattern_type.83d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f9c]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc11_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.f6f = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.7ca.%C [concrete = constants.%C.f6f]
// CHECK:STDOUT:     %self: %C.f6f = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.7ca [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.14e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.cc6: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.5 = fn_decl @C.as.ImplicitAs.impl.Convert.loc12 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.557 = value_param_pattern [concrete = constants.%self.param_patt.792]
// CHECK:STDOUT:     %self.patt: %pattern_type.557 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5ed]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc12_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.77a = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.cc6.%C [concrete = constants.%C.77a]
// CHECK:STDOUT:     %self: %C.77a = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.cc6 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.727]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.def: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.6 = fn_decl @C.as.ImplicitAs.impl.Convert.loc13 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5ad = value_param_pattern [concrete = constants.%self.param_patt.b99]
// CHECK:STDOUT:     %self.patt: %pattern_type.5ad = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.fff]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc13_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.bd8 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.def.%C [concrete = constants.%C.bd8]
// CHECK:STDOUT:     %self: %C.bd8 = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.def [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.648]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.c5d: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.7 = fn_decl @C.as.ImplicitAs.impl.Convert.loc14 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e13 = value_param_pattern [concrete = constants.%self.param_patt.187]
// CHECK:STDOUT:     %self.patt: %pattern_type.e13 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.8b8]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc14_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.a98 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.c5d.%C [concrete = constants.%C.a98]
// CHECK:STDOUT:     %self: %C.a98 = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.c5d [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.490]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.2e9: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.860b0a.8 = fn_decl @C.as.ImplicitAs.impl.Convert.loc15 [concrete = constants.%C.as.ImplicitAs.impl.Convert.54620a.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98d = value_param_pattern [concrete = constants.%self.param_patt.fca]
// CHECK:STDOUT:     %self.patt: %pattern_type.98d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.ca9]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b22 = out_param_pattern [concrete = constants.%return.param_patt.dca]
// CHECK:STDOUT:     %return.patt: %pattern_type.b22 = return_slot_pattern %return.param_patt, %D.ref [concrete = constants.%return.patt.904]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc15_62.2: Core.Form = init_form %D.ref [concrete = constants.%.128]
// CHECK:STDOUT:     %self.param: %C.bc5 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.2e9.%C [concrete = constants.%C.bc5]
// CHECK:STDOUT:     %self: %C.bc5 = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.2e9 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.4ef]
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
// CHECK:STDOUT:   %N.patt.loc4_10.2: %pattern_type.df9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_10.2 (constants.%N.patt.282)]
// CHECK:STDOUT:   %N.loc4_10.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_10.1 (constants.%N.06f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.67b
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n.m.5e7 [concrete = constants.%complete_type.8d0]
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
// CHECK:STDOUT:   %impl.elem0.loc6_40.1: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc6_40.1: <bound method> = bound_method %int_0.loc6_31, %impl.elem0.loc6_40.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1da]
// CHECK:STDOUT:   %specific_fn.loc6_40.1: <specific function> = specific_function %impl.elem0.loc6_40.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.2: <bound method> = bound_method %int_0.loc6_31, %specific_fn.loc6_40.1 [concrete = constants.%bound_method.be3]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.1: init %i32 = call %bound_method.loc6_40.2(%int_0.loc6_31) [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc6_40.2: init %i32 = converted %int_0.loc6_31, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.1 [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc6_40.3: ref %i32 = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc6_40.4: init %i32 to %.loc6_40.3 = in_place_init %.loc6_40.2 [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %impl.elem0.loc6_40.2: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc6_40.3: <bound method> = bound_method %int_0.loc6_39, %impl.elem0.loc6_40.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1da]
// CHECK:STDOUT:   %specific_fn.loc6_40.2: <specific function> = specific_function %impl.elem0.loc6_40.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.4: <bound method> = bound_method %int_0.loc6_39, %specific_fn.loc6_40.2 [concrete = constants.%bound_method.be3]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.2: init %i32 = call %bound_method.loc6_40.4(%int_0.loc6_39) [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc6_40.5: init %i32 = converted %int_0.loc6_39, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.2 [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc6_40.6: ref %i32 = class_element_access %return.param, element1
// CHECK:STDOUT:   %.loc6_40.7: init %i32 to %.loc6_40.6 = in_place_init %.loc6_40.5 [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc6_40.8: init %D to %return.param = class_init (%.loc6_40.4, %.loc6_40.7) [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc6_41: init %D = converted %.loc6_40.1, %.loc6_40.8 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc6_41 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc8(%self.param: %C.7ad) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc8_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc8_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc9(%self.param: %C.dca) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc9_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc9_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc10(%self.param: %C.e54) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc10_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc10_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc11(%self.param: %C.f6f) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc11_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc11_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc12(%self.param: %C.77a) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc12_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc12_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc13(%self.param: %C.bd8) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc13_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc14(%self.param: %C.a98) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc14_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc15(%self.param: %C.bc5) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc15_62.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc15_62.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.06f) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.282
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%N.06f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.1b1) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.282
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_0.1b1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.874) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.282
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_1.874
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.92b) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.282
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_2.92b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.0aa) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.282
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_3.0aa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.bdd) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.282
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_4.bdd
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.ff2) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.282
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_5.ff2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.47a) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.282
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_6.47a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.367) {
// CHECK:STDOUT:   %N.patt.loc4_10.2 => constants.%N.patt.282
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_7.367
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
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %n.param_patt: %pattern_type.df9 = value_param_pattern [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.df9 = wrapper_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %struct_type.n.m: type = struct_type {.n: %i32, .m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.8d0: <witness> = complete_type_witness %struct_type.n.m [concrete]
// CHECK:STDOUT:   %.128: Core.Form = init_form %D [concrete]
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
// CHECK:STDOUT:   %N.06f: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.282: %pattern_type.df9 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e87: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.186: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.604: %ImplicitAs.type.cb7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e87) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.604) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac8, %ImplicitAs.facet.604 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1da: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.be3: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.1b1: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %C.7ad: type = class_type @C, @C(%int_0.1b1) [concrete]
// CHECK:STDOUT:   %C.val.a2a: %C.7ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.24d: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %int_1.874: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.dca: type = class_type @C, @C(%int_1.874) [concrete]
// CHECK:STDOUT:   %int_2.92b: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.e54: type = class_type @C, @C(%int_2.92b) [concrete]
// CHECK:STDOUT:   %int_3.0aa: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.f6f: type = class_type @C, @C(%int_3.0aa) [concrete]
// CHECK:STDOUT:   %int_4.bdd: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %C.77a: type = class_type @C, @C(%int_4.bdd) [concrete]
// CHECK:STDOUT:   %int_5.ff2: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %C.bd8: type = class_type @C, @C(%int_5.ff2) [concrete]
// CHECK:STDOUT:   %int_6.47a: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %C.a98: type = class_type @C, @C(%int_6.47a) [concrete]
// CHECK:STDOUT:   %int_7.367: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %C.bc5: type = class_type @C, @C(%int_7.367) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.041: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1e4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9b5: %ImplicitAs.type.24d = facet_value %C.7ad, (%ImplicitAs.impl_witness.041) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.481: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.9b5) [concrete]
// CHECK:STDOUT:   %.557: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.481, %ImplicitAs.facet.9b5 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.1: type = fn_type @C.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.1: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.f86: type = pattern_type %C.7ad [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.47a: <bound method> = bound_method %C.val.a2a, %C.as.ImplicitAs.impl.Convert.3cefb0.1 [concrete]
// CHECK:STDOUT:   %.e49: ref %C.7ad = temporary invalid, %C.val.a2a [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc7_26.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc7_26.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.a20: %pattern_type.f86 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.703: %pattern_type.f86 = wrapper_binding_pattern self, %self.param_patt.a20 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc7_26.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc7_26.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc7_26.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.bc5: <bound method> = bound_method %.e49, %Destroy.WithSelf.SelfDestruct.db3fdb.2 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a5f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.67d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.054: %C.dca = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.488: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.80c [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f5f: %ImplicitAs.type.24d = facet_value %C.dca, (%ImplicitAs.impl_witness.488) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.63b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.f5f) [concrete]
// CHECK:STDOUT:   %.a18: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.63b, %ImplicitAs.facet.f5f [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.2: type = fn_type @C.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.2: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.0aa: type = pattern_type %C.dca [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.7a7: <bound method> = bound_method %C.val.054, %C.as.ImplicitAs.impl.Convert.3cefb0.2 [concrete]
// CHECK:STDOUT:   %.e68: ref %C.dca = temporary invalid, %C.val.054 [concrete]
// CHECK:STDOUT:   %self.param_patt.55e: %pattern_type.0aa = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.7fb: %pattern_type.0aa = wrapper_binding_pattern self, %self.param_patt.55e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.fc2: <bound method> = bound_method %.e68, %Destroy.WithSelf.SelfDestruct.db3fdb.3 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.ca1: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.70c: %C.e54 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.23c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.2e4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b5e: %ImplicitAs.type.24d = facet_value %C.e54, (%ImplicitAs.impl_witness.23c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.2cc: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.b5e) [concrete]
// CHECK:STDOUT:   %.0cd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.2cc, %ImplicitAs.facet.b5e [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.3: type = fn_type @C.as.ImplicitAs.impl.Convert.3 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.3: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9e8: type = pattern_type %C.e54 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.a7a: <bound method> = bound_method %C.val.70c, %C.as.ImplicitAs.impl.Convert.3cefb0.3 [concrete]
// CHECK:STDOUT:   %.08a: ref %C.e54 = temporary invalid, %C.val.70c [concrete]
// CHECK:STDOUT:   %self.param_patt.cf6: %pattern_type.9e8 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.ce3: %pattern_type.9e8 = wrapper_binding_pattern self, %self.param_patt.cf6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.439: <bound method> = bound_method %.08a, %Destroy.WithSelf.SelfDestruct.db3fdb.4 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d7b: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.690: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.eda: %C.f6f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.127: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.07c [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.804: %ImplicitAs.type.24d = facet_value %C.f6f, (%ImplicitAs.impl_witness.127) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ce1: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.804) [concrete]
// CHECK:STDOUT:   %.3c7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ce1, %ImplicitAs.facet.804 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.4: type = fn_type @C.as.ImplicitAs.impl.Convert.4 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.4: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.4 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.83d: type = pattern_type %C.f6f [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.5d0: <bound method> = bound_method %C.val.eda, %C.as.ImplicitAs.impl.Convert.3cefb0.4 [concrete]
// CHECK:STDOUT:   %.023: ref %C.f6f = temporary invalid, %C.val.eda [concrete]
// CHECK:STDOUT:   %self.param_patt.5d3: %pattern_type.83d = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.750: %pattern_type.83d = wrapper_binding_pattern self, %self.param_patt.5d3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.686: <bound method> = bound_method %.023, %Destroy.WithSelf.SelfDestruct.db3fdb.5 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7be: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.a39: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.937: %C.77a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc5: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8cf [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.081: %ImplicitAs.type.24d = facet_value %C.77a, (%ImplicitAs.impl_witness.bc5) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.192: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.081) [concrete]
// CHECK:STDOUT:   %.37c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.192, %ImplicitAs.facet.081 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.5: type = fn_type @C.as.ImplicitAs.impl.Convert.5 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.5: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.5 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.557: type = pattern_type %C.77a [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.299: <bound method> = bound_method %C.val.937, %C.as.ImplicitAs.impl.Convert.3cefb0.5 [concrete]
// CHECK:STDOUT:   %.a13: ref %C.77a = temporary invalid, %C.val.937 [concrete]
// CHECK:STDOUT:   %self.param_patt.762: %pattern_type.557 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.2c5: %pattern_type.557 = wrapper_binding_pattern self, %self.param_patt.762 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc11 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc11 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.6: type = fn_type @Destroy.WithSelf.SelfDestruct.loc11 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.6: %Destroy.WithSelf.SelfDestruct.type.fbceb5.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.c94: <bound method> = bound_method %.a13, %Destroy.WithSelf.SelfDestruct.db3fdb.6 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a32: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.954: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.5e8: %C.bd8 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.866: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ff4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d81: %ImplicitAs.type.24d = facet_value %C.bd8, (%ImplicitAs.impl_witness.866) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.223: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.d81) [concrete]
// CHECK:STDOUT:   %.e4f7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.223, %ImplicitAs.facet.d81 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.6: type = fn_type @C.as.ImplicitAs.impl.Convert.6 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.6: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.5ad: type = pattern_type %C.bd8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.d4d: <bound method> = bound_method %C.val.5e8, %C.as.ImplicitAs.impl.Convert.3cefb0.6 [concrete]
// CHECK:STDOUT:   %.163: ref %C.bd8 = temporary invalid, %C.val.5e8 [concrete]
// CHECK:STDOUT:   %self.param_patt.321: %pattern_type.5ad = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1ee: %pattern_type.5ad = wrapper_binding_pattern self, %self.param_patt.321 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc12 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc12 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.7: type = fn_type @Destroy.WithSelf.SelfDestruct.loc12 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.7: %Destroy.WithSelf.SelfDestruct.type.fbceb5.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.124: <bound method> = bound_method %.163, %Destroy.WithSelf.SelfDestruct.db3fdb.7 [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.552: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.a13: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.207: %C.a98 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d45: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5d9 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.4e8: %ImplicitAs.type.24d = facet_value %C.a98, (%ImplicitAs.impl_witness.d45) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e79: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.4e8) [concrete]
// CHECK:STDOUT:   %.16e: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e79, %ImplicitAs.facet.4e8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.7: type = fn_type @C.as.ImplicitAs.impl.Convert.7 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.7: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.7 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.e13: type = pattern_type %C.a98 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.3f5: <bound method> = bound_method %C.val.207, %C.as.ImplicitAs.impl.Convert.3cefb0.7 [concrete]
// CHECK:STDOUT:   %.fe5: ref %C.a98 = temporary invalid, %C.val.207 [concrete]
// CHECK:STDOUT:   %self.param_patt.1f4: %pattern_type.e13 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.a44: %pattern_type.e13 = wrapper_binding_pattern self, %self.param_patt.1f4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc13 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc13 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc13 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.fd0: <bound method> = bound_method %.fe5, %Destroy.WithSelf.SelfDestruct.db3fdb.8 [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.828: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %bound_method.aaa: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.9ed: %C.bc5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.59e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cd2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.62b: %ImplicitAs.type.24d = facet_value %C.bc5, (%ImplicitAs.impl_witness.59e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a67: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.62b) [concrete]
// CHECK:STDOUT:   %.60f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a67, %ImplicitAs.facet.62b [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae6f4c.8: type = fn_type @C.as.ImplicitAs.impl.Convert.8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.3cefb0.8: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.8 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98d: type = pattern_type %C.bc5 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.a2c: <bound method> = bound_method %C.val.9ed, %C.as.ImplicitAs.impl.Convert.3cefb0.8 [concrete]
// CHECK:STDOUT:   %.ac8: ref %C.bc5 = temporary invalid, %C.val.9ed [concrete]
// CHECK:STDOUT:   %self.param_patt.593: %pattern_type.98d = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e01: %pattern_type.98d = wrapper_binding_pattern self, %self.param_patt.593 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc14 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.9: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.loc14 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.loc14 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.c02: <bound method> = bound_method %.ac8, %Destroy.WithSelf.SelfDestruct.db3fdb.9 [concrete]
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
// CHECK:STDOUT:   %P.import_ref.f04: <witness> = import_ref P//library, loc5_35, loaded [concrete = constants.%complete_type.8d0]
// CHECK:STDOUT:   %P.import_ref.737 = import_ref P//library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.a6d = import_ref P//library, loc5_16, unloaded
// CHECK:STDOUT:   %P.import_ref.904 = import_ref P//library, loc5_28, unloaded
// CHECK:STDOUT:   %P.C: %C.type = import_ref P//library, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %P.import_ref.8f2: <witness> = import_ref P//library, loc4_18, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %P.import_ref.ebe = import_ref P//library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.340: %i32 = import_ref P//library, loc4_10, loaded [symbolic = @C.%N (constants.%N.06f)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %P.import_ref.814: <witness> = import_ref P//library, loc8_33, loaded [concrete = constants.%ImplicitAs.impl_witness.041]
// CHECK:STDOUT:   %P.import_ref.6a6: type = import_ref P//library, loc8_9, loaded [concrete = constants.%C.7ad]
// CHECK:STDOUT:   %P.import_ref.4efc12.1: type = import_ref P//library, loc8_31, loaded [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   %P.import_ref.37a: <witness> = import_ref P//library, loc9_33, loaded [concrete = constants.%ImplicitAs.impl_witness.488]
// CHECK:STDOUT:   %P.import_ref.1cb: type = import_ref P//library, loc9_9, loaded [concrete = constants.%C.dca]
// CHECK:STDOUT:   %P.import_ref.4efc12.2: type = import_ref P//library, loc9_31, loaded [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   %P.import_ref.3f6: <witness> = import_ref P//library, loc10_33, loaded [concrete = constants.%ImplicitAs.impl_witness.23c]
// CHECK:STDOUT:   %P.import_ref.757: type = import_ref P//library, loc10_9, loaded [concrete = constants.%C.e54]
// CHECK:STDOUT:   %P.import_ref.4efc12.3: type = import_ref P//library, loc10_31, loaded [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   %P.import_ref.32c: <witness> = import_ref P//library, loc11_33, loaded [concrete = constants.%ImplicitAs.impl_witness.127]
// CHECK:STDOUT:   %P.import_ref.dce3: type = import_ref P//library, loc11_9, loaded [concrete = constants.%C.f6f]
// CHECK:STDOUT:   %P.import_ref.4efc12.4: type = import_ref P//library, loc11_31, loaded [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   %P.import_ref.3de: <witness> = import_ref P//library, loc12_33, loaded [concrete = constants.%ImplicitAs.impl_witness.bc5]
// CHECK:STDOUT:   %P.import_ref.3ef: type = import_ref P//library, loc12_9, loaded [concrete = constants.%C.77a]
// CHECK:STDOUT:   %P.import_ref.4efc12.5: type = import_ref P//library, loc12_31, loaded [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   %P.import_ref.fc4: <witness> = import_ref P//library, loc13_33, loaded [concrete = constants.%ImplicitAs.impl_witness.866]
// CHECK:STDOUT:   %P.import_ref.df5: type = import_ref P//library, loc13_9, loaded [concrete = constants.%C.bd8]
// CHECK:STDOUT:   %P.import_ref.4efc12.6: type = import_ref P//library, loc13_31, loaded [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   %P.import_ref.504: <witness> = import_ref P//library, loc14_33, loaded [concrete = constants.%ImplicitAs.impl_witness.d45]
// CHECK:STDOUT:   %P.import_ref.e54: type = import_ref P//library, loc14_9, loaded [concrete = constants.%C.a98]
// CHECK:STDOUT:   %P.import_ref.4efc12.7: type = import_ref P//library, loc14_31, loaded [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   %P.import_ref.efb: <witness> = import_ref P//library, loc15_33, loaded [concrete = constants.%ImplicitAs.impl_witness.59e]
// CHECK:STDOUT:   %P.import_ref.6e0: type = import_ref P//library, loc15_9, loaded [concrete = constants.%C.bc5]
// CHECK:STDOUT:   %P.import_ref.4efc12.8: type = import_ref P//library, loc15_31, loaded [concrete = constants.%ImplicitAs.type.24d]
// CHECK:STDOUT:   %P.import_ref.b89e2b.1: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.1 = import_ref P//library, loc8_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1e4 = impl_witness_table (%P.import_ref.b89e2b.1), @C.as.ImplicitAs.impl.4f7 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %P.import_ref.b89e2b.2: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.2 = import_ref P//library, loc9_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.80c = impl_witness_table (%P.import_ref.b89e2b.2), @C.as.ImplicitAs.impl.855 [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.3: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.3 = import_ref P//library, loc10_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.3]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.2e4 = impl_witness_table (%P.import_ref.b89e2b.3), @C.as.ImplicitAs.impl.459 [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.4: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.4 = import_ref P//library, loc11_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.4]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.07c = impl_witness_table (%P.import_ref.b89e2b.4), @C.as.ImplicitAs.impl.c5c [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.5: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.5 = import_ref P//library, loc12_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.5]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8cf = impl_witness_table (%P.import_ref.b89e2b.5), @C.as.ImplicitAs.impl.68f [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.6: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.6 = import_ref P//library, loc13_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.6]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ff4 = impl_witness_table (%P.import_ref.b89e2b.6), @C.as.ImplicitAs.impl.edd [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.7: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.7 = import_ref P//library, loc14_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.7]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5d9 = impl_witness_table (%P.import_ref.b89e2b.7), @C.as.ImplicitAs.impl.0f0 [concrete]
// CHECK:STDOUT:   %P.import_ref.b89e2b.8: %C.as.ImplicitAs.impl.Convert.type.ae6f4c.8 = import_ref P//library, loc15_64, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.8]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cd2 = impl_witness_table (%P.import_ref.b89e2b.8), @C.as.ImplicitAs.impl.93b [concrete]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f86 = ref_param_pattern [concrete = constants.%self.param_patt.a20]
// CHECK:STDOUT:     %self.patt: %pattern_type.f86 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.703]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.7ad = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.7ad = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc7_26.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f86 = ref_param_pattern [concrete = constants.%self.param_patt.a20]
// CHECK:STDOUT:     %self.patt: %pattern_type.f86 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.703]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.7ad = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.7ad = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0aa = ref_param_pattern [concrete = constants.%self.param_patt.55e]
// CHECK:STDOUT:     %self.patt: %pattern_type.0aa = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.7fb]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.dca = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.dca = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0aa = ref_param_pattern [concrete = constants.%self.param_patt.55e]
// CHECK:STDOUT:     %self.patt: %pattern_type.0aa = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.7fb]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.dca = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.dca = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9e8 = ref_param_pattern [concrete = constants.%self.param_patt.cf6]
// CHECK:STDOUT:     %self.patt: %pattern_type.9e8 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.ce3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.e54 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.e54 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc9 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9e8 = ref_param_pattern [concrete = constants.%self.param_patt.cf6]
// CHECK:STDOUT:     %self.patt: %pattern_type.9e8 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.ce3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.e54 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.e54 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc10 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.83d = ref_param_pattern [concrete = constants.%self.param_patt.5d3]
// CHECK:STDOUT:     %self.patt: %pattern_type.83d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.750]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.f6f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.f6f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc10 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.83d = ref_param_pattern [concrete = constants.%self.param_patt.5d3]
// CHECK:STDOUT:     %self.patt: %pattern_type.83d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.750]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.f6f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.f6f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc11 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.557 = ref_param_pattern [concrete = constants.%self.param_patt.762]
// CHECK:STDOUT:     %self.patt: %pattern_type.557 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.2c5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.77a = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.77a = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc11 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.557 = ref_param_pattern [concrete = constants.%self.param_patt.762]
// CHECK:STDOUT:     %self.patt: %pattern_type.557 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.2c5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.77a = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.77a = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc12 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5ad = ref_param_pattern [concrete = constants.%self.param_patt.321]
// CHECK:STDOUT:     %self.patt: %pattern_type.5ad = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ee]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.bd8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.bd8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc12 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5ad = ref_param_pattern [concrete = constants.%self.param_patt.321]
// CHECK:STDOUT:     %self.patt: %pattern_type.5ad = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ee]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.bd8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.bd8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc13 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e13 = ref_param_pattern [concrete = constants.%self.param_patt.1f4]
// CHECK:STDOUT:     %self.patt: %pattern_type.e13 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a44]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.a98 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.a98 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.8: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc13 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e13 = ref_param_pattern [concrete = constants.%self.param_patt.1f4]
// CHECK:STDOUT:     %self.patt: %pattern_type.e13 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a44]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.a98 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.a98 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.9: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc14 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.9] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98d = ref_param_pattern [concrete = constants.%self.param_patt.593]
// CHECK:STDOUT:     %self.patt: %pattern_type.98d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e01]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.bc5 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.bc5 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.9: %Destroy.WithSelf.Op.type.ef016f.9 = fn_decl @Destroy.WithSelf.Op.loc14 [concrete = constants.%Destroy.WithSelf.Op.403171.9] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98d = ref_param_pattern [concrete = constants.%self.param_patt.593]
// CHECK:STDOUT:     %self.patt: %pattern_type.98d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e01]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.bc5 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.bc5 = wrapper_binding self, %self.param
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
// CHECK:STDOUT:     %n.param_patt: %pattern_type.df9 = value_param_pattern [concrete = constants.%n.param_patt]
// CHECK:STDOUT:     %n.patt: %pattern_type.df9 = wrapper_binding_pattern n, %n.param_patt [concrete = constants.%n.patt]
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
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.4f7: imports.%P.import_ref.6a6 as imports.%P.import_ref.4efc12.1 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.814
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.855: imports.%P.import_ref.1cb as imports.%P.import_ref.4efc12.2 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.37a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.459: imports.%P.import_ref.757 as imports.%P.import_ref.4efc12.3 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.3f6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.c5c: imports.%P.import_ref.dce3 as imports.%P.import_ref.4efc12.4 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.32c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.68f: imports.%P.import_ref.3ef as imports.%P.import_ref.4efc12.5 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.3de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.edd: imports.%P.import_ref.df5 as imports.%P.import_ref.4efc12.6 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.fc4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.0f0: imports.%P.import_ref.e54 as imports.%P.import_ref.4efc12.7 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.504
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.93b: imports.%P.import_ref.6e0 as imports.%P.import_ref.4efc12.8 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.efb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "library.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%P.import_ref.f04
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%P.import_ref.737
// CHECK:STDOUT:   .n = imports.%P.import_ref.a6d
// CHECK:STDOUT:   .m = imports.%P.import_ref.904
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%P.import_ref.340: %i32) [from "library.carbon"] {
// CHECK:STDOUT:   %N.patt: %pattern_type.df9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt.282)]
// CHECK:STDOUT:   %N: %i32 = symbolic_binding N, 0 [symbolic = %N (constants.%N.06f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%P.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%P.import_ref.ebe
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
// CHECK:STDOUT:   %impl.elem0.loc7_34: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc7_34.1: <bound method> = bound_method %int_0, %impl.elem0.loc7_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1da]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_34.2: <bound method> = bound_method %int_0, %specific_fn.loc7 [concrete = constants.%bound_method.be3]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_34.2(%int_0) [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc7_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc7_34.2: %i32 = converted %int_0, %.loc7_34.1 [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %C.loc7: type = class_type @C, @C(constants.%int_0.1b1) [concrete = constants.%C.7ad]
// CHECK:STDOUT:   %.loc7_24.2: ref %C.7ad = temporary_storage
// CHECK:STDOUT:   %.loc7_24.3: init %C.7ad to %.loc7_24.2 = class_init () [concrete = constants.%C.val.a2a]
// CHECK:STDOUT:   %.loc7_26.1: init %C.7ad = converted %.loc7_24.1, %.loc7_24.3 [concrete = constants.%C.val.a2a]
// CHECK:STDOUT:   %impl.elem0.loc7_35: %.557 = impl_witness_access constants.%ImplicitAs.impl_witness.041, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.1]
// CHECK:STDOUT:   %bound_method.loc7_35: <bound method> = bound_method %.loc7_26.1, %impl.elem0.loc7_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.47a]
// CHECK:STDOUT:   %.loc6_26.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc7_26.2: ref %C.7ad = temporary %.loc7_24.2, %.loc7_26.1 [concrete = constants.%.e49]
// CHECK:STDOUT:   %.loc7_26.3: %C.7ad = acquire_value %.loc7_26.2 [concrete = constants.%C.val.a2a]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc7: init %D to %.loc6_26.1 = call %bound_method.loc7_35(%.loc7_26.3)
// CHECK:STDOUT:   %.loc7_35: init %D = converted %.loc7_26.1, %C.as.ImplicitAs.impl.Convert.call.loc7
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc7: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.bc5(constants.%.e49)
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
// CHECK:STDOUT:   %impl.elem0.loc8_34: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a5f]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_1, %specific_fn.loc8 [concrete = constants.%bound_method.67d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_34.2(%int_1) [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_1, %.loc8_34.1 [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %C.loc8: type = class_type @C, @C(constants.%int_1.874) [concrete = constants.%C.dca]
// CHECK:STDOUT:   %.loc8_24.2: ref %C.dca = temporary_storage
// CHECK:STDOUT:   %.loc8_24.3: init %C.dca to %.loc8_24.2 = class_init () [concrete = constants.%C.val.054]
// CHECK:STDOUT:   %.loc8_26.1: init %C.dca = converted %.loc8_24.1, %.loc8_24.3 [concrete = constants.%C.val.054]
// CHECK:STDOUT:   %impl.elem0.loc8_35: %.a18 = impl_witness_access constants.%ImplicitAs.impl_witness.488, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.2]
// CHECK:STDOUT:   %bound_method.loc8_35: <bound method> = bound_method %.loc8_26.1, %impl.elem0.loc8_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.7a7]
// CHECK:STDOUT:   %.loc6_26.2: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc8_26.2: ref %C.dca = temporary %.loc8_24.2, %.loc8_26.1 [concrete = constants.%.e68]
// CHECK:STDOUT:   %.loc8_26.3: %C.dca = acquire_value %.loc8_26.2 [concrete = constants.%C.val.054]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc8: init %D to %.loc6_26.2 = call %bound_method.loc8_35(%.loc8_26.3)
// CHECK:STDOUT:   %.loc8_35: init %D = converted %.loc8_26.1, %C.as.ImplicitAs.impl.Convert.call.loc8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc8: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.fc2(constants.%.e68)
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
// CHECK:STDOUT:   %impl.elem0.loc9_34: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %int_2, %impl.elem0.loc9_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %int_2, %specific_fn.loc9 [concrete = constants.%bound_method.ca1]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_34.2(%int_2) [concrete = constants.%int_2.92b]
// CHECK:STDOUT:   %.loc9_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_2.92b]
// CHECK:STDOUT:   %.loc9_34.2: %i32 = converted %int_2, %.loc9_34.1 [concrete = constants.%int_2.92b]
// CHECK:STDOUT:   %C.loc9: type = class_type @C, @C(constants.%int_2.92b) [concrete = constants.%C.e54]
// CHECK:STDOUT:   %.loc9_24.2: ref %C.e54 = temporary_storage
// CHECK:STDOUT:   %.loc9_24.3: init %C.e54 to %.loc9_24.2 = class_init () [concrete = constants.%C.val.70c]
// CHECK:STDOUT:   %.loc9_26.1: init %C.e54 = converted %.loc9_24.1, %.loc9_24.3 [concrete = constants.%C.val.70c]
// CHECK:STDOUT:   %impl.elem0.loc9_35: %.0cd = impl_witness_access constants.%ImplicitAs.impl_witness.23c, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.3]
// CHECK:STDOUT:   %bound_method.loc9_35: <bound method> = bound_method %.loc9_26.1, %impl.elem0.loc9_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.a7a]
// CHECK:STDOUT:   %.loc6_26.3: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc9_26.2: ref %C.e54 = temporary %.loc9_24.2, %.loc9_26.1 [concrete = constants.%.08a]
// CHECK:STDOUT:   %.loc9_26.3: %C.e54 = acquire_value %.loc9_26.2 [concrete = constants.%C.val.70c]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc9: init %D to %.loc6_26.3 = call %bound_method.loc9_35(%.loc9_26.3)
// CHECK:STDOUT:   %.loc9_35: init %D = converted %.loc9_26.1, %C.as.ImplicitAs.impl.Convert.call.loc9
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.439(constants.%.08a)
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
// CHECK:STDOUT:   %impl.elem0.loc10_34: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc10_34.1: <bound method> = bound_method %int_3, %impl.elem0.loc10_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d7b]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_34.2: <bound method> = bound_method %int_3, %specific_fn.loc10 [concrete = constants.%bound_method.690]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %i32 = call %bound_method.loc10_34.2(%int_3) [concrete = constants.%int_3.0aa]
// CHECK:STDOUT:   %.loc10_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_3.0aa]
// CHECK:STDOUT:   %.loc10_34.2: %i32 = converted %int_3, %.loc10_34.1 [concrete = constants.%int_3.0aa]
// CHECK:STDOUT:   %C.loc10: type = class_type @C, @C(constants.%int_3.0aa) [concrete = constants.%C.f6f]
// CHECK:STDOUT:   %.loc10_24.2: ref %C.f6f = temporary_storage
// CHECK:STDOUT:   %.loc10_24.3: init %C.f6f to %.loc10_24.2 = class_init () [concrete = constants.%C.val.eda]
// CHECK:STDOUT:   %.loc10_26.1: init %C.f6f = converted %.loc10_24.1, %.loc10_24.3 [concrete = constants.%C.val.eda]
// CHECK:STDOUT:   %impl.elem0.loc10_35: %.3c7 = impl_witness_access constants.%ImplicitAs.impl_witness.127, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.4]
// CHECK:STDOUT:   %bound_method.loc10_35: <bound method> = bound_method %.loc10_26.1, %impl.elem0.loc10_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.5d0]
// CHECK:STDOUT:   %.loc6_26.4: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc10_26.2: ref %C.f6f = temporary %.loc10_24.2, %.loc10_26.1 [concrete = constants.%.023]
// CHECK:STDOUT:   %.loc10_26.3: %C.f6f = acquire_value %.loc10_26.2 [concrete = constants.%C.val.eda]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc10: init %D to %.loc6_26.4 = call %bound_method.loc10_35(%.loc10_26.3)
// CHECK:STDOUT:   %.loc10_35: init %D = converted %.loc10_26.1, %C.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc10: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.686(constants.%.023)
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
// CHECK:STDOUT:   %impl.elem0.loc11_34: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc11_34.1: <bound method> = bound_method %int_4, %impl.elem0.loc11_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7be]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_34.2: <bound method> = bound_method %int_4, %specific_fn.loc11 [concrete = constants.%bound_method.a39]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %i32 = call %bound_method.loc11_34.2(%int_4) [concrete = constants.%int_4.bdd]
// CHECK:STDOUT:   %.loc11_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_4.bdd]
// CHECK:STDOUT:   %.loc11_34.2: %i32 = converted %int_4, %.loc11_34.1 [concrete = constants.%int_4.bdd]
// CHECK:STDOUT:   %C.loc11: type = class_type @C, @C(constants.%int_4.bdd) [concrete = constants.%C.77a]
// CHECK:STDOUT:   %.loc11_24.2: ref %C.77a = temporary_storage
// CHECK:STDOUT:   %.loc11_24.3: init %C.77a to %.loc11_24.2 = class_init () [concrete = constants.%C.val.937]
// CHECK:STDOUT:   %.loc11_26.1: init %C.77a = converted %.loc11_24.1, %.loc11_24.3 [concrete = constants.%C.val.937]
// CHECK:STDOUT:   %impl.elem0.loc11_35: %.37c = impl_witness_access constants.%ImplicitAs.impl_witness.bc5, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.5]
// CHECK:STDOUT:   %bound_method.loc11_35: <bound method> = bound_method %.loc11_26.1, %impl.elem0.loc11_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.299]
// CHECK:STDOUT:   %.loc6_26.5: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc11_26.2: ref %C.77a = temporary %.loc11_24.2, %.loc11_26.1 [concrete = constants.%.a13]
// CHECK:STDOUT:   %.loc11_26.3: %C.77a = acquire_value %.loc11_26.2 [concrete = constants.%C.val.937]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc11: init %D to %.loc6_26.5 = call %bound_method.loc11_35(%.loc11_26.3)
// CHECK:STDOUT:   %.loc11_35: init %D = converted %.loc11_26.1, %C.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc11: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.c94(constants.%.a13)
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
// CHECK:STDOUT:   %impl.elem0.loc12_34: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc12_34.1: <bound method> = bound_method %int_5, %impl.elem0.loc12_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a32]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_34.2: <bound method> = bound_method %int_5, %specific_fn.loc12 [concrete = constants.%bound_method.954]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %i32 = call %bound_method.loc12_34.2(%int_5) [concrete = constants.%int_5.ff2]
// CHECK:STDOUT:   %.loc12_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_5.ff2]
// CHECK:STDOUT:   %.loc12_34.2: %i32 = converted %int_5, %.loc12_34.1 [concrete = constants.%int_5.ff2]
// CHECK:STDOUT:   %C.loc12: type = class_type @C, @C(constants.%int_5.ff2) [concrete = constants.%C.bd8]
// CHECK:STDOUT:   %.loc12_24.2: ref %C.bd8 = temporary_storage
// CHECK:STDOUT:   %.loc12_24.3: init %C.bd8 to %.loc12_24.2 = class_init () [concrete = constants.%C.val.5e8]
// CHECK:STDOUT:   %.loc12_26.1: init %C.bd8 = converted %.loc12_24.1, %.loc12_24.3 [concrete = constants.%C.val.5e8]
// CHECK:STDOUT:   %impl.elem0.loc12_35: %.e4f7 = impl_witness_access constants.%ImplicitAs.impl_witness.866, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.6]
// CHECK:STDOUT:   %bound_method.loc12_35: <bound method> = bound_method %.loc12_26.1, %impl.elem0.loc12_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.d4d]
// CHECK:STDOUT:   %.loc6_26.6: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc12_26.2: ref %C.bd8 = temporary %.loc12_24.2, %.loc12_26.1 [concrete = constants.%.163]
// CHECK:STDOUT:   %.loc12_26.3: %C.bd8 = acquire_value %.loc12_26.2 [concrete = constants.%C.val.5e8]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc12: init %D to %.loc6_26.6 = call %bound_method.loc12_35(%.loc12_26.3)
// CHECK:STDOUT:   %.loc12_35: init %D = converted %.loc12_26.1, %C.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc12: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.124(constants.%.163)
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
// CHECK:STDOUT:   %impl.elem0.loc13_34: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc13_34.1: <bound method> = bound_method %int_6, %impl.elem0.loc13_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.552]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_34.2: <bound method> = bound_method %int_6, %specific_fn.loc13 [concrete = constants.%bound_method.a13]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %i32 = call %bound_method.loc13_34.2(%int_6) [concrete = constants.%int_6.47a]
// CHECK:STDOUT:   %.loc13_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%int_6.47a]
// CHECK:STDOUT:   %.loc13_34.2: %i32 = converted %int_6, %.loc13_34.1 [concrete = constants.%int_6.47a]
// CHECK:STDOUT:   %C.loc13: type = class_type @C, @C(constants.%int_6.47a) [concrete = constants.%C.a98]
// CHECK:STDOUT:   %.loc13_24.2: ref %C.a98 = temporary_storage
// CHECK:STDOUT:   %.loc13_24.3: init %C.a98 to %.loc13_24.2 = class_init () [concrete = constants.%C.val.207]
// CHECK:STDOUT:   %.loc13_26.1: init %C.a98 = converted %.loc13_24.1, %.loc13_24.3 [concrete = constants.%C.val.207]
// CHECK:STDOUT:   %impl.elem0.loc13_35: %.16e = impl_witness_access constants.%ImplicitAs.impl_witness.d45, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.7]
// CHECK:STDOUT:   %bound_method.loc13_35: <bound method> = bound_method %.loc13_26.1, %impl.elem0.loc13_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.3f5]
// CHECK:STDOUT:   %.loc6_26.7: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc13_26.2: ref %C.a98 = temporary %.loc13_24.2, %.loc13_26.1 [concrete = constants.%.fe5]
// CHECK:STDOUT:   %.loc13_26.3: %C.a98 = acquire_value %.loc13_26.2 [concrete = constants.%C.val.207]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc13: init %D to %.loc6_26.7 = call %bound_method.loc13_35(%.loc13_26.3)
// CHECK:STDOUT:   %.loc13_35: init %D = converted %.loc13_26.1, %C.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc13: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.fd0(constants.%.fe5)
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
// CHECK:STDOUT:   %impl.elem0.loc14_34: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc14_34.1: <bound method> = bound_method %int_7, %impl.elem0.loc14_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.828]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.2: <bound method> = bound_method %int_7, %specific_fn.loc14 [concrete = constants.%bound_method.aaa]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14: init %i32 = call %bound_method.loc14_34.2(%int_7) [concrete = constants.%int_7.367]
// CHECK:STDOUT:   %.loc14_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14 [concrete = constants.%int_7.367]
// CHECK:STDOUT:   %.loc14_34.2: %i32 = converted %int_7, %.loc14_34.1 [concrete = constants.%int_7.367]
// CHECK:STDOUT:   %C.loc14: type = class_type @C, @C(constants.%int_7.367) [concrete = constants.%C.bc5]
// CHECK:STDOUT:   %.loc14_24.2: ref %C.bc5 = temporary_storage
// CHECK:STDOUT:   %.loc14_24.3: init %C.bc5 to %.loc14_24.2 = class_init () [concrete = constants.%C.val.9ed]
// CHECK:STDOUT:   %.loc14_26.1: init %C.bc5 = converted %.loc14_24.1, %.loc14_24.3 [concrete = constants.%C.val.9ed]
// CHECK:STDOUT:   %impl.elem0.loc14_35: %.60f = impl_witness_access constants.%ImplicitAs.impl_witness.59e, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.3cefb0.8]
// CHECK:STDOUT:   %bound_method.loc14_35: <bound method> = bound_method %.loc14_26.1, %impl.elem0.loc14_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.a2c]
// CHECK:STDOUT:   %.loc6_26.8: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc14_26.2: ref %C.bc5 = temporary %.loc14_24.2, %.loc14_26.1 [concrete = constants.%.ac8]
// CHECK:STDOUT:   %.loc14_26.3: %C.bc5 = acquire_value %.loc14_26.2 [concrete = constants.%C.val.9ed]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc14: init %D to %.loc6_26.8 = call %bound_method.loc14_35(%.loc14_26.3)
// CHECK:STDOUT:   %.loc14_35: init %D = converted %.loc14_26.1, %C.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc14: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.c02(constants.%.ac8)
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc7_26.2(%self.param: ref %C.7ad) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc7_26.2(%self.param: ref %C.7ad) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc7_26.2(%self.param: ref %C.7ad) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.2 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %C.dca) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %C.dca) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %C.dca) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.3 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9(%self.param: ref %C.e54) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9(%self.param: ref %C.e54) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9(%self.param: ref %C.e54) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.4 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc10(%self.param: ref %C.f6f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc10(%self.param: ref %C.f6f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc10(%self.param: ref %C.f6f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.5 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc11(%self.param: ref %C.77a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc11(%self.param: ref %C.77a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc11(%self.param: ref %C.77a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.6 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc12(%self.param: ref %C.bd8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc12(%self.param: ref %C.bd8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc12(%self.param: ref %C.bd8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.7(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.7(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.7 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc13(%self.param: ref %C.a98) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc13(%self.param: ref %C.a98) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc13(%self.param: ref %C.a98) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.8(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.8(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.8 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc14(%self.param: ref %C.bc5) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc14(%self.param: ref %C.bc5) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc14(%self.param: ref %C.bc5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.9(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.9(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.06f) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.282
// CHECK:STDOUT:   %N => constants.%N.06f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.1b1) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.282
// CHECK:STDOUT:   %N => constants.%int_0.1b1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.874) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.282
// CHECK:STDOUT:   %N => constants.%int_1.874
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.92b) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.282
// CHECK:STDOUT:   %N => constants.%int_2.92b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.0aa) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.282
// CHECK:STDOUT:   %N => constants.%int_3.0aa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.bdd) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.282
// CHECK:STDOUT:   %N => constants.%int_4.bdd
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.ff2) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.282
// CHECK:STDOUT:   %N => constants.%int_5.ff2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.47a) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.282
// CHECK:STDOUT:   %N => constants.%int_6.47a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.367) {
// CHECK:STDOUT:   %N.patt => constants.%N.patt.282
// CHECK:STDOUT:   %N => constants.%int_7.367
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
