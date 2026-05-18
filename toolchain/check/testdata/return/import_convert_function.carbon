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

impl C(0) as Core.ImplicitAs(D) { fn Convert[unused self: Self]() -> D { return Make(); } }
impl C(1) as Core.ImplicitAs(D) { fn Convert[unused self: Self]() -> D { return Make(); } }
impl C(2) as Core.ImplicitAs(D) { fn Convert[unused self: Self]() -> D { return Make(); } }
impl C(3) as Core.ImplicitAs(D) { fn Convert[unused self: Self]() -> D { return Make(); } }
impl C(4) as Core.ImplicitAs(D) { fn Convert[unused self: Self]() -> D { return Make(); } }
impl C(5) as Core.ImplicitAs(D) { fn Convert[unused self: Self]() -> D { return Make(); } }
impl C(6) as Core.ImplicitAs(D) { fn Convert[unused self: Self]() -> D { return Make(); } }
impl C(7) as Core.ImplicitAs(D) { fn Convert[unused self: Self]() -> D { return Make(); } }

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
// CHECK:STDOUT:   %pattern_type.17a: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.921: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.5f1: type = class_type @C, @C(%N.921) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.e40: type = struct_type {.n: %i32, .m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.6f9: <witness> = complete_type_witness %struct_type.n.m.e40 [concrete]
// CHECK:STDOUT:   %.169: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %pattern_type.d39: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.819: type = struct_type {.n: Core.IntLiteral, .m: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.n.m.819 = struct_value (%int_0.5c6, %int_0.5c6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b23: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.b23 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ef: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.abb: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f4c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e6c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ca: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ca = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3f1: %ImplicitAs.type.0ef = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f4c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.61c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.3f1) [concrete]
// CHECK:STDOUT:   %.cbf: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.61c, %ImplicitAs.facet.3f1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ce0: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e76: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.fed: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_0.fed, %int_0.fed) [concrete]
// CHECK:STDOUT:   %C.71f: type = class_type @C, @C(%int_0.fed) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e47: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.877: <witness> = impl_witness @C.as.ImplicitAs.impl.7b5.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.389: type = pattern_type %C.71f [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.2f1cce.1: type = fn_type @C.as.ImplicitAs.impl.Convert.loc8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.b44fbb.1: %C.as.ImplicitAs.impl.Convert.type.2f1cce.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.1c4: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d72: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.578: type = class_type @C, @C(%int_1.d72) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.cd6: <witness> = impl_witness @C.as.ImplicitAs.impl.3ac.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.67c: type = pattern_type %C.578 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.2f1cce.2: type = fn_type @C.as.ImplicitAs.impl.Convert.loc9 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.b44fbb.2: %C.as.ImplicitAs.impl.Convert.type.2f1cce.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.099: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.c74: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.cb7: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.e98: type = class_type @C, @C(%int_2.cb7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb3: <witness> = impl_witness @C.as.ImplicitAs.impl.98b.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.8d9: type = pattern_type %C.e98 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.2f1cce.3: type = fn_type @C.as.ImplicitAs.impl.Convert.loc10 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.b44fbb.3: %C.as.ImplicitAs.impl.Convert.type.2f1cce.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.eab: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.77a: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.781: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.4d1: type = class_type @C, @C(%int_3.781) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1be: <witness> = impl_witness @C.as.ImplicitAs.impl.b7c.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.e20: type = pattern_type %C.4d1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.2f1cce.4: type = fn_type @C.as.ImplicitAs.impl.Convert.loc11 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.b44fbb.4: %C.as.ImplicitAs.impl.Convert.type.2f1cce.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.e80: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.c2d: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.d29: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %C.b75: type = class_type @C, @C(%int_4.d29) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.868: <witness> = impl_witness @C.as.ImplicitAs.impl.eec.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.498: type = pattern_type %C.b75 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.2f1cce.5: type = fn_type @C.as.ImplicitAs.impl.Convert.loc12 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.b44fbb.5: %C.as.ImplicitAs.impl.Convert.type.2f1cce.5 = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.e8f: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.469: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.bb1: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %C.b5f: type = class_type @C, @C(%int_5.bb1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2b2: <witness> = impl_witness @C.as.ImplicitAs.impl.682.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.744: type = pattern_type %C.b5f [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.2f1cce.6: type = fn_type @C.as.ImplicitAs.impl.Convert.loc13 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.b44fbb.6: %C.as.ImplicitAs.impl.Convert.type.2f1cce.6 = struct_value () [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87f: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.700: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.c0f: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %C.91c: type = class_type @C, @C(%int_6.c0f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.27b: <witness> = impl_witness @C.as.ImplicitAs.impl.38b.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.417: type = pattern_type %C.91c [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.2f1cce.7: type = fn_type @C.as.ImplicitAs.impl.Convert.loc14 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.b44fbb.7: %C.as.ImplicitAs.impl.Convert.type.2f1cce.7 = struct_value () [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0bb: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.48c: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.075: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %C.41b: type = class_type @C, @C(%int_7.075) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.8c6: <witness> = impl_witness @C.as.ImplicitAs.impl.d9b.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.2b3: type = pattern_type %C.41b [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.2f1cce.8: type = fn_type @C.as.ImplicitAs.impl.Convert.loc15 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.b44fbb.8: %C.as.ImplicitAs.impl.Convert.type.2f1cce.8 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.b23 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.fa9: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.abb)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e6c = impl_witness_table (%Core.import_ref.fa9), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.17a = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_10.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_10.1 (constants.%N.921)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d39 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d39 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc6_14: Core.Form = init_form %D.ref [concrete = constants.%.169]
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.7b5 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:     %bound_method.loc8_9.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ce0]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_9.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.e76]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_9.2(%int_0) [concrete = constants.%int_0.fed]
// CHECK:STDOUT:     %.loc8_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.fed]
// CHECK:STDOUT:     %.loc8_9.2: %i32 = converted %int_0, %.loc8_9.1 [concrete = constants.%int_0.fed]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_0.fed) [concrete = constants.%C.71f]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.b23 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.e47]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.3ac [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %impl.elem0: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:     %bound_method.loc9_9.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b7]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_9.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method.1c4]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_9.2(%int_1) [concrete = constants.%int_1.d72]
// CHECK:STDOUT:     %.loc9_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.d72]
// CHECK:STDOUT:     %.loc9_9.2: %i32 = converted %int_1, %.loc9_9.1 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_1.d72) [concrete = constants.%C.578]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.b23 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.e47]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.98b [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:     %bound_method.loc10_9.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.099]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_9.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.c74]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc10_9.2(%int_2) [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:     %.loc10_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:     %.loc10_9.2: %i32 = converted %int_2, %.loc10_9.1 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_2.cb7) [concrete = constants.%C.e98]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.b23 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.e47]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.b7c [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:     %bound_method.loc11_9.1: <bound method> = bound_method %int_3, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.eab]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_9.2: <bound method> = bound_method %int_3, %specific_fn [concrete = constants.%bound_method.77a]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_9.2(%int_3) [concrete = constants.%int_3.781]
// CHECK:STDOUT:     %.loc11_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.781]
// CHECK:STDOUT:     %.loc11_9.2: %i32 = converted %int_3, %.loc11_9.1 [concrete = constants.%int_3.781]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_3.781) [concrete = constants.%C.4d1]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.b23 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.e47]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.eec [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %impl.elem0: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:     %bound_method.loc12_9.1: <bound method> = bound_method %int_4, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.e80]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc12_9.2: <bound method> = bound_method %int_4, %specific_fn [concrete = constants.%bound_method.c2d]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_9.2(%int_4) [concrete = constants.%int_4.d29]
// CHECK:STDOUT:     %.loc12_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_4.d29]
// CHECK:STDOUT:     %.loc12_9.2: %i32 = converted %int_4, %.loc12_9.1 [concrete = constants.%int_4.d29]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_4.d29) [concrete = constants.%C.b75]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.b23 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.e47]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.682 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:     %impl.elem0: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:     %bound_method.loc13_9.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.e8f]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc13_9.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method.469]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_9.2(%int_5) [concrete = constants.%int_5.bb1]
// CHECK:STDOUT:     %.loc13_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.bb1]
// CHECK:STDOUT:     %.loc13_9.2: %i32 = converted %int_5, %.loc13_9.1 [concrete = constants.%int_5.bb1]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_5.bb1) [concrete = constants.%C.b5f]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.b23 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.e47]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.38b [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:     %impl.elem0: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %int_6, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87f]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %int_6, %specific_fn [concrete = constants.%bound_method.700]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc14_9.2(%int_6) [concrete = constants.%int_6.c0f]
// CHECK:STDOUT:     %.loc14_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_6.c0f]
// CHECK:STDOUT:     %.loc14_9.2: %i32 = converted %int_6, %.loc14_9.1 [concrete = constants.%int_6.c0f]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_6.c0f) [concrete = constants.%C.91c]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.b23 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.e47]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.d9b [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:     %impl.elem0: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:     %bound_method.loc15_9.1: <bound method> = bound_method %int_7, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0bb]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc15_9.2: <bound method> = bound_method %int_7, %specific_fn [concrete = constants.%bound_method.48c]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_9.2(%int_7) [concrete = constants.%int_7.075]
// CHECK:STDOUT:     %.loc15_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_7.075]
// CHECK:STDOUT:     %.loc15_9.2: %i32 = converted %int_7, %.loc15_9.1 [concrete = constants.%int_7.075]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_7.075) [concrete = constants.%C.41b]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.b23 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.e47]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.7b5: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.2f1cce.1 = fn_decl @C.as.ImplicitAs.impl.Convert.loc8 [concrete = constants.%C.as.ImplicitAs.impl.Convert.b44fbb.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.389 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.389 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d39 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d39 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc8_70.2: Core.Form = init_form %D.ref [concrete = constants.%.169]
// CHECK:STDOUT:     %self.param: %C.71f = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.7b5.%C [concrete = constants.%C.71f]
// CHECK:STDOUT:     %self: %C.71f = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.7b5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.877]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.3ac: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.2f1cce.2 = fn_decl @C.as.ImplicitAs.impl.Convert.loc9 [concrete = constants.%C.as.ImplicitAs.impl.Convert.b44fbb.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.67c = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.67c = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d39 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d39 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc9_70.2: Core.Form = init_form %D.ref [concrete = constants.%.169]
// CHECK:STDOUT:     %self.param: %C.578 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.3ac.%C [concrete = constants.%C.578]
// CHECK:STDOUT:     %self: %C.578 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.3ac [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.cd6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.98b: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.2f1cce.3 = fn_decl @C.as.ImplicitAs.impl.Convert.loc10 [concrete = constants.%C.as.ImplicitAs.impl.Convert.b44fbb.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8d9 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.8d9 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d39 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d39 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc10_70.2: Core.Form = init_form %D.ref [concrete = constants.%.169]
// CHECK:STDOUT:     %self.param: %C.e98 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.98b.%C [concrete = constants.%C.e98]
// CHECK:STDOUT:     %self: %C.e98 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.98b [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.fb3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.b7c: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.2f1cce.4 = fn_decl @C.as.ImplicitAs.impl.Convert.loc11 [concrete = constants.%C.as.ImplicitAs.impl.Convert.b44fbb.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e20 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.e20 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d39 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d39 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc11_70.2: Core.Form = init_form %D.ref [concrete = constants.%.169]
// CHECK:STDOUT:     %self.param: %C.4d1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.b7c.%C [concrete = constants.%C.4d1]
// CHECK:STDOUT:     %self: %C.4d1 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.b7c [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.1be]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.eec: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.2f1cce.5 = fn_decl @C.as.ImplicitAs.impl.Convert.loc12 [concrete = constants.%C.as.ImplicitAs.impl.Convert.b44fbb.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.498 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.498 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d39 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d39 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc12_70.2: Core.Form = init_form %D.ref [concrete = constants.%.169]
// CHECK:STDOUT:     %self.param: %C.b75 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.eec.%C [concrete = constants.%C.b75]
// CHECK:STDOUT:     %self: %C.b75 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.eec [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.868]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.682: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.2f1cce.6 = fn_decl @C.as.ImplicitAs.impl.Convert.loc13 [concrete = constants.%C.as.ImplicitAs.impl.Convert.b44fbb.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.744 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.744 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d39 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d39 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc13_70.2: Core.Form = init_form %D.ref [concrete = constants.%.169]
// CHECK:STDOUT:     %self.param: %C.b5f = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.682.%C [concrete = constants.%C.b5f]
// CHECK:STDOUT:     %self: %C.b5f = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.682 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.2b2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.38b: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.2f1cce.7 = fn_decl @C.as.ImplicitAs.impl.Convert.loc14 [concrete = constants.%C.as.ImplicitAs.impl.Convert.b44fbb.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.417 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.417 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d39 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d39 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc14_70.2: Core.Form = init_form %D.ref [concrete = constants.%.169]
// CHECK:STDOUT:     %self.param: %C.91c = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.38b.%C [concrete = constants.%C.91c]
// CHECK:STDOUT:     %self: %C.91c = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.38b [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.27b]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.d9b: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.2f1cce.8 = fn_decl @C.as.ImplicitAs.impl.Convert.loc15 [concrete = constants.%C.as.ImplicitAs.impl.Convert.b44fbb.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2b3 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.2b3 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d39 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d39 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc15_70.2: Core.Form = init_form %D.ref [concrete = constants.%.169]
// CHECK:STDOUT:     %self.param: %C.41b = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.d9b.%C [concrete = constants.%C.41b]
// CHECK:STDOUT:     %self: %C.41b = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.d9b [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.8c6]
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
// CHECK:STDOUT:   %N.loc4_10.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_10.1 (constants.%N.921)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.5f1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc5_16: %D.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %.loc5_28: %D.elem = field_decl m, element1 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n.m.e40 [concrete = constants.%complete_type.6f9]
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
// CHECK:STDOUT:   %impl.elem0.loc6_40.1: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc6_40.1: <bound method> = bound_method %int_0.loc6_31, %impl.elem0.loc6_40.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ce0]
// CHECK:STDOUT:   %specific_fn.loc6_40.1: <specific function> = specific_function %impl.elem0.loc6_40.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.2: <bound method> = bound_method %int_0.loc6_31, %specific_fn.loc6_40.1 [concrete = constants.%bound_method.e76]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.1: init %i32 = call %bound_method.loc6_40.2(%int_0.loc6_31) [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %.loc6_40.2: init %i32 = converted %int_0.loc6_31, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.1 [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %.loc6_40.3: ref %i32 = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc6_40.4: init %i32 to %.loc6_40.3 = in_place_init %.loc6_40.2 [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %impl.elem0.loc6_40.2: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc6_40.3: <bound method> = bound_method %int_0.loc6_39, %impl.elem0.loc6_40.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ce0]
// CHECK:STDOUT:   %specific_fn.loc6_40.2: <specific function> = specific_function %impl.elem0.loc6_40.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.4: <bound method> = bound_method %int_0.loc6_39, %specific_fn.loc6_40.2 [concrete = constants.%bound_method.e76]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.2: init %i32 = call %bound_method.loc6_40.4(%int_0.loc6_39) [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %.loc6_40.5: init %i32 = converted %int_0.loc6_39, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.2 [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %.loc6_40.6: ref %i32 = class_element_access %return.param, element1
// CHECK:STDOUT:   %.loc6_40.7: init %i32 to %.loc6_40.6 = in_place_init %.loc6_40.5 [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %.loc6_40.8: init %D to %return.param = class_init (%.loc6_40.4, %.loc6_40.7) [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc6_41: init %D = converted %.loc6_40.1, %.loc6_40.8 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc6_41 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc8(%self.param: %C.71f) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc8_70.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc8_70.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc9(%self.param: %C.578) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc9_70.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc9_70.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc10(%self.param: %C.e98) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc10_70.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc10_70.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc11(%self.param: %C.4d1) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc11_70.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc11_70.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc12(%self.param: %C.b75) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc12_70.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc12_70.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc13(%self.param: %C.b5f) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_70.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc13_70.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc14(%self.param: %C.91c) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_70.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc14_70.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc15(%self.param: %C.41b) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc15_70.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc15_70.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.921) {
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%N.921
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.fed) {
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_0.fed
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.d72) {
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_1.d72
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.cb7) {
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_2.cb7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.781) {
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_3.781
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.d29) {
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_4.d29
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.bb1) {
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_5.bb1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.c0f) {
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_6.c0f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.075) {
// CHECK:STDOUT:   %N.loc4_10.1 => constants.%int_7.075
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
// CHECK:STDOUT:   %pattern_type.17a: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %struct_type.n.m: type = struct_type {.n: %i32, .m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.6f9: <witness> = complete_type_witness %struct_type.n.m [concrete]
// CHECK:STDOUT:   %.169: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %pattern_type.d39: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %F0.type: type = fn_type @F0 [concrete]
// CHECK:STDOUT:   %F0: %F0.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %N.921: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b23: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.b23 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ef: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.abb: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f4c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e6c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ca: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ca = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3f1: %ImplicitAs.type.0ef = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f4c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.61c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.3f1) [concrete]
// CHECK:STDOUT:   %.cbf: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.61c, %ImplicitAs.facet.3f1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ce0: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e76: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.fed: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %C.71f: type = class_type @C, @C(%int_0.fed) [concrete]
// CHECK:STDOUT:   %C.val.a76: %C.71f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e47: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %int_1.d72: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.578: type = class_type @C, @C(%int_1.d72) [concrete]
// CHECK:STDOUT:   %int_2.cb7: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.e98: type = class_type @C, @C(%int_2.cb7) [concrete]
// CHECK:STDOUT:   %int_3.781: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.4d1: type = class_type @C, @C(%int_3.781) [concrete]
// CHECK:STDOUT:   %int_4.d29: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %C.b75: type = class_type @C, @C(%int_4.d29) [concrete]
// CHECK:STDOUT:   %int_5.bb1: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %C.b5f: type = class_type @C, @C(%int_5.bb1) [concrete]
// CHECK:STDOUT:   %int_6.c0f: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %C.91c: type = class_type @C, @C(%int_6.c0f) [concrete]
// CHECK:STDOUT:   %int_7.075: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %C.41b: type = class_type @C, @C(%int_7.075) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1d1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6e1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e2a: %ImplicitAs.type.e47 = facet_value %C.71f, (%ImplicitAs.impl_witness.1d1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9d3: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.e2a) [concrete]
// CHECK:STDOUT:   %.280: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9d3, %ImplicitAs.facet.e2a [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.0060e4.1: type = fn_type @C.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.680efc.1: %C.as.ImplicitAs.impl.Convert.type.0060e4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.d49: <bound method> = bound_method %C.val.a76, %C.as.ImplicitAs.impl.Convert.680efc.1 [concrete]
// CHECK:STDOUT:   %.800: ref %C.71f = temporary invalid, %C.val.a76 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.2: type = fn_type @Destroy.Op.loc7_26.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.2: %Destroy.Op.type.3f5f78.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.f55: <bound method> = bound_method %.800, %Destroy.Op.f0ba5f.2 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.1c4: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.4bb: %C.578 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2a0: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ff8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a3e: %ImplicitAs.type.e47 = facet_value %C.578, (%ImplicitAs.impl_witness.2a0) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.327: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.a3e) [concrete]
// CHECK:STDOUT:   %.6da: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.327, %ImplicitAs.facet.a3e [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.0060e4.2: type = fn_type @C.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.680efc.2: %C.as.ImplicitAs.impl.Convert.type.0060e4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.551: <bound method> = bound_method %C.val.4bb, %C.as.ImplicitAs.impl.Convert.680efc.2 [concrete]
// CHECK:STDOUT:   %.217: ref %C.578 = temporary invalid, %C.val.4bb [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.3: type = fn_type @Destroy.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.3: %Destroy.Op.type.3f5f78.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.bba: <bound method> = bound_method %.217, %Destroy.Op.f0ba5f.3 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.099: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.c74: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.282: %C.e98 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e5e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.426 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.80d: %ImplicitAs.type.e47 = facet_value %C.e98, (%ImplicitAs.impl_witness.e5e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6ea: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.80d) [concrete]
// CHECK:STDOUT:   %.500: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6ea, %ImplicitAs.facet.80d [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.0060e4.3: type = fn_type @C.as.ImplicitAs.impl.Convert.3 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.680efc.3: %C.as.ImplicitAs.impl.Convert.type.0060e4.3 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.331: <bound method> = bound_method %C.val.282, %C.as.ImplicitAs.impl.Convert.680efc.3 [concrete]
// CHECK:STDOUT:   %.014: ref %C.e98 = temporary invalid, %C.val.282 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.4: type = fn_type @Destroy.Op.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.4: %Destroy.Op.type.3f5f78.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.601: <bound method> = bound_method %.014, %Destroy.Op.f0ba5f.4 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.eab: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.77a: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.8d5: %C.4d1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2c9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d13 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a94: %ImplicitAs.type.e47 = facet_value %C.4d1, (%ImplicitAs.impl_witness.2c9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.7e6: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.a94) [concrete]
// CHECK:STDOUT:   %.b6c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.7e6, %ImplicitAs.facet.a94 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.0060e4.4: type = fn_type @C.as.ImplicitAs.impl.Convert.4 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.680efc.4: %C.as.ImplicitAs.impl.Convert.type.0060e4.4 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.dee: <bound method> = bound_method %C.val.8d5, %C.as.ImplicitAs.impl.Convert.680efc.4 [concrete]
// CHECK:STDOUT:   %.f7b: ref %C.4d1 = temporary invalid, %C.val.8d5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.5: type = fn_type @Destroy.Op.loc10 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.5: %Destroy.Op.type.3f5f78.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.a92: <bound method> = bound_method %.f7b, %Destroy.Op.f0ba5f.5 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.e80: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.c2d: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.6c3: %C.b75 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1d9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5b9 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f20: %ImplicitAs.type.e47 = facet_value %C.b75, (%ImplicitAs.impl_witness.1d9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a9d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.f20) [concrete]
// CHECK:STDOUT:   %.e4f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a9d, %ImplicitAs.facet.f20 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.0060e4.5: type = fn_type @C.as.ImplicitAs.impl.Convert.5 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.680efc.5: %C.as.ImplicitAs.impl.Convert.type.0060e4.5 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.48e: <bound method> = bound_method %C.val.6c3, %C.as.ImplicitAs.impl.Convert.680efc.5 [concrete]
// CHECK:STDOUT:   %.42a: ref %C.b75 = temporary invalid, %C.val.6c3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.6: type = fn_type @Destroy.Op.loc11 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.6: %Destroy.Op.type.3f5f78.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.f66: <bound method> = bound_method %.42a, %Destroy.Op.f0ba5f.6 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.e8f: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.469: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.484: %C.b5f = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.777: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8a3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b8a: %ImplicitAs.type.e47 = facet_value %C.b5f, (%ImplicitAs.impl_witness.777) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.fb0: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.b8a) [concrete]
// CHECK:STDOUT:   %.c0c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.fb0, %ImplicitAs.facet.b8a [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.0060e4.6: type = fn_type @C.as.ImplicitAs.impl.Convert.6 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.680efc.6: %C.as.ImplicitAs.impl.Convert.type.0060e4.6 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.cf9: <bound method> = bound_method %C.val.484, %C.as.ImplicitAs.impl.Convert.680efc.6 [concrete]
// CHECK:STDOUT:   %.e33: ref %C.b5f = temporary invalid, %C.val.484 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.7: type = fn_type @Destroy.Op.loc12 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.7: %Destroy.Op.type.3f5f78.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.7a5: <bound method> = bound_method %.e33, %Destroy.Op.f0ba5f.7 [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87f: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.700: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.9d9: %C.91c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bf3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a9e [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.75c: %ImplicitAs.type.e47 = facet_value %C.91c, (%ImplicitAs.impl_witness.bf3) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e05: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.75c) [concrete]
// CHECK:STDOUT:   %.a77: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e05, %ImplicitAs.facet.75c [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.0060e4.7: type = fn_type @C.as.ImplicitAs.impl.Convert.7 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.680efc.7: %C.as.ImplicitAs.impl.Convert.type.0060e4.7 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.1c2: <bound method> = bound_method %C.val.9d9, %C.as.ImplicitAs.impl.Convert.680efc.7 [concrete]
// CHECK:STDOUT:   %.d77: ref %C.91c = temporary invalid, %C.val.9d9 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.8: type = fn_type @Destroy.Op.loc13 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.8: %Destroy.Op.type.3f5f78.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.066: <bound method> = bound_method %.d77, %Destroy.Op.f0ba5f.8 [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0bb: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %bound_method.48c: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.51d: %C.41b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5be: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cda [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d20: %ImplicitAs.type.e47 = facet_value %C.41b, (%ImplicitAs.impl_witness.5be) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.123: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.d20) [concrete]
// CHECK:STDOUT:   %.472: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.123, %ImplicitAs.facet.d20 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.0060e4.8: type = fn_type @C.as.ImplicitAs.impl.Convert.8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.680efc.8: %C.as.ImplicitAs.impl.Convert.type.0060e4.8 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.a31: <bound method> = bound_method %C.val.51d, %C.as.ImplicitAs.impl.Convert.680efc.8 [concrete]
// CHECK:STDOUT:   %.84d: ref %C.41b = temporary invalid, %C.val.51d [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.9: type = fn_type @Destroy.Op.loc14 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.9: %Destroy.Op.type.3f5f78.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound.d03: <bound method> = bound_method %.84d, %Destroy.Op.f0ba5f.9 [concrete]
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
// CHECK:STDOUT:   %P.import_ref.7a9: <witness> = import_ref P//library, loc5_35, loaded [concrete = constants.%complete_type.6f9]
// CHECK:STDOUT:   %P.import_ref.222 = import_ref P//library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.521 = import_ref P//library, loc5_16, unloaded
// CHECK:STDOUT:   %P.import_ref.f09 = import_ref P//library, loc5_28, unloaded
// CHECK:STDOUT:   %P.C: %C.type = import_ref P//library, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %P.import_ref.8f2: <witness> = import_ref P//library, loc4_19, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %P.import_ref.db1 = import_ref P//library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.5b8: %i32 = import_ref P//library, loc4_10, loaded [symbolic = @C.%N (constants.%N.921)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.b23 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.fa9: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.abb)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e6c = impl_witness_table (%Core.import_ref.fa9), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %P.import_ref.ee1: <witness> = import_ref P//library, loc8_33, loaded [concrete = constants.%ImplicitAs.impl_witness.1d1]
// CHECK:STDOUT:   %P.import_ref.b6e: type = import_ref P//library, loc8_9, loaded [concrete = constants.%C.71f]
// CHECK:STDOUT:   %P.import_ref.886dc9.1: type = import_ref P//library, loc8_31, loaded [concrete = constants.%ImplicitAs.type.e47]
// CHECK:STDOUT:   %P.import_ref.261: <witness> = import_ref P//library, loc9_33, loaded [concrete = constants.%ImplicitAs.impl_witness.2a0]
// CHECK:STDOUT:   %P.import_ref.942: type = import_ref P//library, loc9_9, loaded [concrete = constants.%C.578]
// CHECK:STDOUT:   %P.import_ref.886dc9.2: type = import_ref P//library, loc9_31, loaded [concrete = constants.%ImplicitAs.type.e47]
// CHECK:STDOUT:   %P.import_ref.84b: <witness> = import_ref P//library, loc10_33, loaded [concrete = constants.%ImplicitAs.impl_witness.e5e]
// CHECK:STDOUT:   %P.import_ref.a89: type = import_ref P//library, loc10_9, loaded [concrete = constants.%C.e98]
// CHECK:STDOUT:   %P.import_ref.886dc9.3: type = import_ref P//library, loc10_31, loaded [concrete = constants.%ImplicitAs.type.e47]
// CHECK:STDOUT:   %P.import_ref.1cc: <witness> = import_ref P//library, loc11_33, loaded [concrete = constants.%ImplicitAs.impl_witness.2c9]
// CHECK:STDOUT:   %P.import_ref.0f9: type = import_ref P//library, loc11_9, loaded [concrete = constants.%C.4d1]
// CHECK:STDOUT:   %P.import_ref.886dc9.4: type = import_ref P//library, loc11_31, loaded [concrete = constants.%ImplicitAs.type.e47]
// CHECK:STDOUT:   %P.import_ref.a61: <witness> = import_ref P//library, loc12_33, loaded [concrete = constants.%ImplicitAs.impl_witness.1d9]
// CHECK:STDOUT:   %P.import_ref.c75: type = import_ref P//library, loc12_9, loaded [concrete = constants.%C.b75]
// CHECK:STDOUT:   %P.import_ref.886dc9.5: type = import_ref P//library, loc12_31, loaded [concrete = constants.%ImplicitAs.type.e47]
// CHECK:STDOUT:   %P.import_ref.ed0: <witness> = import_ref P//library, loc13_33, loaded [concrete = constants.%ImplicitAs.impl_witness.777]
// CHECK:STDOUT:   %P.import_ref.f83: type = import_ref P//library, loc13_9, loaded [concrete = constants.%C.b5f]
// CHECK:STDOUT:   %P.import_ref.886dc9.6: type = import_ref P//library, loc13_31, loaded [concrete = constants.%ImplicitAs.type.e47]
// CHECK:STDOUT:   %P.import_ref.f2a: <witness> = import_ref P//library, loc14_33, loaded [concrete = constants.%ImplicitAs.impl_witness.bf3]
// CHECK:STDOUT:   %P.import_ref.4e7: type = import_ref P//library, loc14_9, loaded [concrete = constants.%C.91c]
// CHECK:STDOUT:   %P.import_ref.886dc9.7: type = import_ref P//library, loc14_31, loaded [concrete = constants.%ImplicitAs.type.e47]
// CHECK:STDOUT:   %P.import_ref.4aa: <witness> = import_ref P//library, loc15_33, loaded [concrete = constants.%ImplicitAs.impl_witness.5be]
// CHECK:STDOUT:   %P.import_ref.f49: type = import_ref P//library, loc15_9, loaded [concrete = constants.%C.41b]
// CHECK:STDOUT:   %P.import_ref.886dc9.8: type = import_ref P//library, loc15_31, loaded [concrete = constants.%ImplicitAs.type.e47]
// CHECK:STDOUT:   %P.import_ref.64e37a.1: %C.as.ImplicitAs.impl.Convert.type.0060e4.1 = import_ref P//library, loc8_72, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.680efc.1]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6e1 = impl_witness_table (%P.import_ref.64e37a.1), @C.as.ImplicitAs.impl.22c [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %P.import_ref.64e37a.2: %C.as.ImplicitAs.impl.Convert.type.0060e4.2 = import_ref P//library, loc9_72, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.680efc.2]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ff8 = impl_witness_table (%P.import_ref.64e37a.2), @C.as.ImplicitAs.impl.0be [concrete]
// CHECK:STDOUT:   %P.import_ref.64e37a.3: %C.as.ImplicitAs.impl.Convert.type.0060e4.3 = import_ref P//library, loc10_72, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.680efc.3]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.426 = impl_witness_table (%P.import_ref.64e37a.3), @C.as.ImplicitAs.impl.c06 [concrete]
// CHECK:STDOUT:   %P.import_ref.64e37a.4: %C.as.ImplicitAs.impl.Convert.type.0060e4.4 = import_ref P//library, loc11_72, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.680efc.4]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d13 = impl_witness_table (%P.import_ref.64e37a.4), @C.as.ImplicitAs.impl.799 [concrete]
// CHECK:STDOUT:   %P.import_ref.64e37a.5: %C.as.ImplicitAs.impl.Convert.type.0060e4.5 = import_ref P//library, loc12_72, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.680efc.5]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5b9 = impl_witness_table (%P.import_ref.64e37a.5), @C.as.ImplicitAs.impl.435 [concrete]
// CHECK:STDOUT:   %P.import_ref.64e37a.6: %C.as.ImplicitAs.impl.Convert.type.0060e4.6 = import_ref P//library, loc13_72, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.680efc.6]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8a3 = impl_witness_table (%P.import_ref.64e37a.6), @C.as.ImplicitAs.impl.6b3 [concrete]
// CHECK:STDOUT:   %P.import_ref.64e37a.7: %C.as.ImplicitAs.impl.Convert.type.0060e4.7 = import_ref P//library, loc14_72, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.680efc.7]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a9e = impl_witness_table (%P.import_ref.64e37a.7), @C.as.ImplicitAs.impl.2dd [concrete]
// CHECK:STDOUT:   %P.import_ref.64e37a.8: %C.as.ImplicitAs.impl.Convert.type.0060e4.8 = import_ref P//library, loc15_72, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.680efc.8]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cda = impl_witness_table (%P.import_ref.64e37a.8), @C.as.ImplicitAs.impl.724 [concrete]
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
// CHECK:STDOUT:     %n.param_patt: %pattern_type.17a = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.17a = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d39 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d39 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %P.ref.loc6: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%P.D [concrete = constants.%D]
// CHECK:STDOUT:     %.loc6_26.10: Core.Form = init_form %D.ref [concrete = constants.%.169]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.22c: imports.%P.import_ref.b6e as imports.%P.import_ref.886dc9.1 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.ee1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.0be: imports.%P.import_ref.942 as imports.%P.import_ref.886dc9.2 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.261
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.c06: imports.%P.import_ref.a89 as imports.%P.import_ref.886dc9.3 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.84b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.799: imports.%P.import_ref.0f9 as imports.%P.import_ref.886dc9.4 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.1cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.435: imports.%P.import_ref.c75 as imports.%P.import_ref.886dc9.5 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.a61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.6b3: imports.%P.import_ref.f83 as imports.%P.import_ref.886dc9.6 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.ed0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.2dd: imports.%P.import_ref.4e7 as imports.%P.import_ref.886dc9.7 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.f2a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.724: imports.%P.import_ref.f49 as imports.%P.import_ref.886dc9.8 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.4aa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "library.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%P.import_ref.7a9
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%P.import_ref.222
// CHECK:STDOUT:   .n = imports.%P.import_ref.521
// CHECK:STDOUT:   .m = imports.%P.import_ref.f09
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%P.import_ref.5b8: %i32) [from "library.carbon"] {
// CHECK:STDOUT:   %N: %i32 = symbolic_binding N, 0 [symbolic = %N (constants.%N.921)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%P.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%P.import_ref.db1
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
// CHECK:STDOUT:   %impl.elem0.loc7_34: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc7_34.1: <bound method> = bound_method %int_0, %impl.elem0.loc7_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ce0]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_34.2: <bound method> = bound_method %int_0, %specific_fn.loc7 [concrete = constants.%bound_method.e76]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_34.2(%int_0) [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %.loc7_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %.loc7_34.2: %i32 = converted %int_0, %.loc7_34.1 [concrete = constants.%int_0.fed]
// CHECK:STDOUT:   %C.loc7: type = class_type @C, @C(constants.%int_0.fed) [concrete = constants.%C.71f]
// CHECK:STDOUT:   %.loc7_24.2: ref %C.71f = temporary_storage
// CHECK:STDOUT:   %.loc7_24.3: init %C.71f to %.loc7_24.2 = class_init () [concrete = constants.%C.val.a76]
// CHECK:STDOUT:   %.loc7_26.1: init %C.71f = converted %.loc7_24.1, %.loc7_24.3 [concrete = constants.%C.val.a76]
// CHECK:STDOUT:   %impl.elem0.loc7_35: %.280 = impl_witness_access constants.%ImplicitAs.impl_witness.1d1, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.680efc.1]
// CHECK:STDOUT:   %bound_method.loc7_35: <bound method> = bound_method %.loc7_26.1, %impl.elem0.loc7_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.d49]
// CHECK:STDOUT:   %.loc6_26.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc7_26.2: ref %C.71f = temporary %.loc7_24.2, %.loc7_26.1 [concrete = constants.%.800]
// CHECK:STDOUT:   %.loc7_26.3: %C.71f = acquire_value %.loc7_26.2 [concrete = constants.%C.val.a76]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc7: init %D to %.loc6_26.1 = call %bound_method.loc7_35(%.loc7_26.3)
// CHECK:STDOUT:   %.loc7_35: init %D = converted %.loc7_26.1, %C.as.ImplicitAs.impl.Convert.call.loc7
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.1: init %empty_tuple.type = call constants.%Destroy.Op.bound.f55(constants.%.800)
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
// CHECK:STDOUT:   %impl.elem0.loc8_34: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b7]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_1, %specific_fn.loc8 [concrete = constants.%bound_method.1c4]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_34.2(%int_1) [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_1, %.loc8_34.1 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %C.loc8: type = class_type @C, @C(constants.%int_1.d72) [concrete = constants.%C.578]
// CHECK:STDOUT:   %.loc8_24.2: ref %C.578 = temporary_storage
// CHECK:STDOUT:   %.loc8_24.3: init %C.578 to %.loc8_24.2 = class_init () [concrete = constants.%C.val.4bb]
// CHECK:STDOUT:   %.loc8_26.1: init %C.578 = converted %.loc8_24.1, %.loc8_24.3 [concrete = constants.%C.val.4bb]
// CHECK:STDOUT:   %impl.elem0.loc8_35: %.6da = impl_witness_access constants.%ImplicitAs.impl_witness.2a0, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.680efc.2]
// CHECK:STDOUT:   %bound_method.loc8_35: <bound method> = bound_method %.loc8_26.1, %impl.elem0.loc8_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.551]
// CHECK:STDOUT:   %.loc6_26.2: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc8_26.2: ref %C.578 = temporary %.loc8_24.2, %.loc8_26.1 [concrete = constants.%.217]
// CHECK:STDOUT:   %.loc8_26.3: %C.578 = acquire_value %.loc8_26.2 [concrete = constants.%C.val.4bb]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc8: init %D to %.loc6_26.2 = call %bound_method.loc8_35(%.loc8_26.3)
// CHECK:STDOUT:   %.loc8_35: init %D = converted %.loc8_26.1, %C.as.ImplicitAs.impl.Convert.call.loc8
// CHECK:STDOUT:   %Destroy.Op.call.loc8_26.1: init %empty_tuple.type = call constants.%Destroy.Op.bound.bba(constants.%.217)
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.2: init %empty_tuple.type = call constants.%Destroy.Op.bound.f55(constants.%.800)
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
// CHECK:STDOUT:   %impl.elem0.loc9_34: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %int_2, %impl.elem0.loc9_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.099]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %int_2, %specific_fn.loc9 [concrete = constants.%bound_method.c74]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_34.2(%int_2) [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc9_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %.loc9_34.2: %i32 = converted %int_2, %.loc9_34.1 [concrete = constants.%int_2.cb7]
// CHECK:STDOUT:   %C.loc9: type = class_type @C, @C(constants.%int_2.cb7) [concrete = constants.%C.e98]
// CHECK:STDOUT:   %.loc9_24.2: ref %C.e98 = temporary_storage
// CHECK:STDOUT:   %.loc9_24.3: init %C.e98 to %.loc9_24.2 = class_init () [concrete = constants.%C.val.282]
// CHECK:STDOUT:   %.loc9_26.1: init %C.e98 = converted %.loc9_24.1, %.loc9_24.3 [concrete = constants.%C.val.282]
// CHECK:STDOUT:   %impl.elem0.loc9_35: %.500 = impl_witness_access constants.%ImplicitAs.impl_witness.e5e, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.680efc.3]
// CHECK:STDOUT:   %bound_method.loc9_35: <bound method> = bound_method %.loc9_26.1, %impl.elem0.loc9_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.331]
// CHECK:STDOUT:   %.loc6_26.3: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc9_26.2: ref %C.e98 = temporary %.loc9_24.2, %.loc9_26.1 [concrete = constants.%.014]
// CHECK:STDOUT:   %.loc9_26.3: %C.e98 = acquire_value %.loc9_26.2 [concrete = constants.%C.val.282]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc9: init %D to %.loc6_26.3 = call %bound_method.loc9_35(%.loc9_26.3)
// CHECK:STDOUT:   %.loc9_35: init %D = converted %.loc9_26.1, %C.as.ImplicitAs.impl.Convert.call.loc9
// CHECK:STDOUT:   %Destroy.Op.call.loc9_26.1: init %empty_tuple.type = call constants.%Destroy.Op.bound.601(constants.%.014)
// CHECK:STDOUT:   %Destroy.Op.call.loc8_26.2: init %empty_tuple.type = call constants.%Destroy.Op.bound.bba(constants.%.217)
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.3: init %empty_tuple.type = call constants.%Destroy.Op.bound.f55(constants.%.800)
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
// CHECK:STDOUT:   %impl.elem0.loc10_34: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc10_34.1: <bound method> = bound_method %int_3, %impl.elem0.loc10_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.eab]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_34.2: <bound method> = bound_method %int_3, %specific_fn.loc10 [concrete = constants.%bound_method.77a]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %i32 = call %bound_method.loc10_34.2(%int_3) [concrete = constants.%int_3.781]
// CHECK:STDOUT:   %.loc10_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_3.781]
// CHECK:STDOUT:   %.loc10_34.2: %i32 = converted %int_3, %.loc10_34.1 [concrete = constants.%int_3.781]
// CHECK:STDOUT:   %C.loc10: type = class_type @C, @C(constants.%int_3.781) [concrete = constants.%C.4d1]
// CHECK:STDOUT:   %.loc10_24.2: ref %C.4d1 = temporary_storage
// CHECK:STDOUT:   %.loc10_24.3: init %C.4d1 to %.loc10_24.2 = class_init () [concrete = constants.%C.val.8d5]
// CHECK:STDOUT:   %.loc10_26.1: init %C.4d1 = converted %.loc10_24.1, %.loc10_24.3 [concrete = constants.%C.val.8d5]
// CHECK:STDOUT:   %impl.elem0.loc10_35: %.b6c = impl_witness_access constants.%ImplicitAs.impl_witness.2c9, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.680efc.4]
// CHECK:STDOUT:   %bound_method.loc10_35: <bound method> = bound_method %.loc10_26.1, %impl.elem0.loc10_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.dee]
// CHECK:STDOUT:   %.loc6_26.4: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc10_26.2: ref %C.4d1 = temporary %.loc10_24.2, %.loc10_26.1 [concrete = constants.%.f7b]
// CHECK:STDOUT:   %.loc10_26.3: %C.4d1 = acquire_value %.loc10_26.2 [concrete = constants.%C.val.8d5]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc10: init %D to %.loc6_26.4 = call %bound_method.loc10_35(%.loc10_26.3)
// CHECK:STDOUT:   %.loc10_35: init %D = converted %.loc10_26.1, %C.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %Destroy.Op.call.loc10_26.1: init %empty_tuple.type = call constants.%Destroy.Op.bound.a92(constants.%.f7b)
// CHECK:STDOUT:   %Destroy.Op.call.loc9_26.2: init %empty_tuple.type = call constants.%Destroy.Op.bound.601(constants.%.014)
// CHECK:STDOUT:   %Destroy.Op.call.loc8_26.3: init %empty_tuple.type = call constants.%Destroy.Op.bound.bba(constants.%.217)
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.4: init %empty_tuple.type = call constants.%Destroy.Op.bound.f55(constants.%.800)
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
// CHECK:STDOUT:   %impl.elem0.loc11_34: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc11_34.1: <bound method> = bound_method %int_4, %impl.elem0.loc11_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.e80]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_34.2: <bound method> = bound_method %int_4, %specific_fn.loc11 [concrete = constants.%bound_method.c2d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %i32 = call %bound_method.loc11_34.2(%int_4) [concrete = constants.%int_4.d29]
// CHECK:STDOUT:   %.loc11_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_4.d29]
// CHECK:STDOUT:   %.loc11_34.2: %i32 = converted %int_4, %.loc11_34.1 [concrete = constants.%int_4.d29]
// CHECK:STDOUT:   %C.loc11: type = class_type @C, @C(constants.%int_4.d29) [concrete = constants.%C.b75]
// CHECK:STDOUT:   %.loc11_24.2: ref %C.b75 = temporary_storage
// CHECK:STDOUT:   %.loc11_24.3: init %C.b75 to %.loc11_24.2 = class_init () [concrete = constants.%C.val.6c3]
// CHECK:STDOUT:   %.loc11_26.1: init %C.b75 = converted %.loc11_24.1, %.loc11_24.3 [concrete = constants.%C.val.6c3]
// CHECK:STDOUT:   %impl.elem0.loc11_35: %.e4f = impl_witness_access constants.%ImplicitAs.impl_witness.1d9, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.680efc.5]
// CHECK:STDOUT:   %bound_method.loc11_35: <bound method> = bound_method %.loc11_26.1, %impl.elem0.loc11_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.48e]
// CHECK:STDOUT:   %.loc6_26.5: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc11_26.2: ref %C.b75 = temporary %.loc11_24.2, %.loc11_26.1 [concrete = constants.%.42a]
// CHECK:STDOUT:   %.loc11_26.3: %C.b75 = acquire_value %.loc11_26.2 [concrete = constants.%C.val.6c3]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc11: init %D to %.loc6_26.5 = call %bound_method.loc11_35(%.loc11_26.3)
// CHECK:STDOUT:   %.loc11_35: init %D = converted %.loc11_26.1, %C.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %Destroy.Op.call.loc11_26.1: init %empty_tuple.type = call constants.%Destroy.Op.bound.f66(constants.%.42a)
// CHECK:STDOUT:   %Destroy.Op.call.loc10_26.2: init %empty_tuple.type = call constants.%Destroy.Op.bound.a92(constants.%.f7b)
// CHECK:STDOUT:   %Destroy.Op.call.loc9_26.3: init %empty_tuple.type = call constants.%Destroy.Op.bound.601(constants.%.014)
// CHECK:STDOUT:   %Destroy.Op.call.loc8_26.4: init %empty_tuple.type = call constants.%Destroy.Op.bound.bba(constants.%.217)
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.5: init %empty_tuple.type = call constants.%Destroy.Op.bound.f55(constants.%.800)
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
// CHECK:STDOUT:   %impl.elem0.loc12_34: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc12_34.1: <bound method> = bound_method %int_5, %impl.elem0.loc12_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.e8f]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_34.2: <bound method> = bound_method %int_5, %specific_fn.loc12 [concrete = constants.%bound_method.469]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %i32 = call %bound_method.loc12_34.2(%int_5) [concrete = constants.%int_5.bb1]
// CHECK:STDOUT:   %.loc12_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_5.bb1]
// CHECK:STDOUT:   %.loc12_34.2: %i32 = converted %int_5, %.loc12_34.1 [concrete = constants.%int_5.bb1]
// CHECK:STDOUT:   %C.loc12: type = class_type @C, @C(constants.%int_5.bb1) [concrete = constants.%C.b5f]
// CHECK:STDOUT:   %.loc12_24.2: ref %C.b5f = temporary_storage
// CHECK:STDOUT:   %.loc12_24.3: init %C.b5f to %.loc12_24.2 = class_init () [concrete = constants.%C.val.484]
// CHECK:STDOUT:   %.loc12_26.1: init %C.b5f = converted %.loc12_24.1, %.loc12_24.3 [concrete = constants.%C.val.484]
// CHECK:STDOUT:   %impl.elem0.loc12_35: %.c0c = impl_witness_access constants.%ImplicitAs.impl_witness.777, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.680efc.6]
// CHECK:STDOUT:   %bound_method.loc12_35: <bound method> = bound_method %.loc12_26.1, %impl.elem0.loc12_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.cf9]
// CHECK:STDOUT:   %.loc6_26.6: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc12_26.2: ref %C.b5f = temporary %.loc12_24.2, %.loc12_26.1 [concrete = constants.%.e33]
// CHECK:STDOUT:   %.loc12_26.3: %C.b5f = acquire_value %.loc12_26.2 [concrete = constants.%C.val.484]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc12: init %D to %.loc6_26.6 = call %bound_method.loc12_35(%.loc12_26.3)
// CHECK:STDOUT:   %.loc12_35: init %D = converted %.loc12_26.1, %C.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %Destroy.Op.call.loc12_26.1: init %empty_tuple.type = call constants.%Destroy.Op.bound.7a5(constants.%.e33)
// CHECK:STDOUT:   %Destroy.Op.call.loc11_26.2: init %empty_tuple.type = call constants.%Destroy.Op.bound.f66(constants.%.42a)
// CHECK:STDOUT:   %Destroy.Op.call.loc10_26.3: init %empty_tuple.type = call constants.%Destroy.Op.bound.a92(constants.%.f7b)
// CHECK:STDOUT:   %Destroy.Op.call.loc9_26.4: init %empty_tuple.type = call constants.%Destroy.Op.bound.601(constants.%.014)
// CHECK:STDOUT:   %Destroy.Op.call.loc8_26.5: init %empty_tuple.type = call constants.%Destroy.Op.bound.bba(constants.%.217)
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.6: init %empty_tuple.type = call constants.%Destroy.Op.bound.f55(constants.%.800)
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
// CHECK:STDOUT:   %impl.elem0.loc13_34: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc13_34.1: <bound method> = bound_method %int_6, %impl.elem0.loc13_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87f]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_34.2: <bound method> = bound_method %int_6, %specific_fn.loc13 [concrete = constants.%bound_method.700]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %i32 = call %bound_method.loc13_34.2(%int_6) [concrete = constants.%int_6.c0f]
// CHECK:STDOUT:   %.loc13_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%int_6.c0f]
// CHECK:STDOUT:   %.loc13_34.2: %i32 = converted %int_6, %.loc13_34.1 [concrete = constants.%int_6.c0f]
// CHECK:STDOUT:   %C.loc13: type = class_type @C, @C(constants.%int_6.c0f) [concrete = constants.%C.91c]
// CHECK:STDOUT:   %.loc13_24.2: ref %C.91c = temporary_storage
// CHECK:STDOUT:   %.loc13_24.3: init %C.91c to %.loc13_24.2 = class_init () [concrete = constants.%C.val.9d9]
// CHECK:STDOUT:   %.loc13_26.1: init %C.91c = converted %.loc13_24.1, %.loc13_24.3 [concrete = constants.%C.val.9d9]
// CHECK:STDOUT:   %impl.elem0.loc13_35: %.a77 = impl_witness_access constants.%ImplicitAs.impl_witness.bf3, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.680efc.7]
// CHECK:STDOUT:   %bound_method.loc13_35: <bound method> = bound_method %.loc13_26.1, %impl.elem0.loc13_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.1c2]
// CHECK:STDOUT:   %.loc6_26.7: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc13_26.2: ref %C.91c = temporary %.loc13_24.2, %.loc13_26.1 [concrete = constants.%.d77]
// CHECK:STDOUT:   %.loc13_26.3: %C.91c = acquire_value %.loc13_26.2 [concrete = constants.%C.val.9d9]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc13: init %D to %.loc6_26.7 = call %bound_method.loc13_35(%.loc13_26.3)
// CHECK:STDOUT:   %.loc13_35: init %D = converted %.loc13_26.1, %C.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %Destroy.Op.call.loc13_26.1: init %empty_tuple.type = call constants.%Destroy.Op.bound.066(constants.%.d77)
// CHECK:STDOUT:   %Destroy.Op.call.loc12_26.2: init %empty_tuple.type = call constants.%Destroy.Op.bound.7a5(constants.%.e33)
// CHECK:STDOUT:   %Destroy.Op.call.loc11_26.3: init %empty_tuple.type = call constants.%Destroy.Op.bound.f66(constants.%.42a)
// CHECK:STDOUT:   %Destroy.Op.call.loc10_26.4: init %empty_tuple.type = call constants.%Destroy.Op.bound.a92(constants.%.f7b)
// CHECK:STDOUT:   %Destroy.Op.call.loc9_26.5: init %empty_tuple.type = call constants.%Destroy.Op.bound.601(constants.%.014)
// CHECK:STDOUT:   %Destroy.Op.call.loc8_26.6: init %empty_tuple.type = call constants.%Destroy.Op.bound.bba(constants.%.217)
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.7: init %empty_tuple.type = call constants.%Destroy.Op.bound.f55(constants.%.800)
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
// CHECK:STDOUT:   %impl.elem0.loc14_34: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc14_34.1: <bound method> = bound_method %int_7, %impl.elem0.loc14_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0bb]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.2: <bound method> = bound_method %int_7, %specific_fn.loc14 [concrete = constants.%bound_method.48c]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14: init %i32 = call %bound_method.loc14_34.2(%int_7) [concrete = constants.%int_7.075]
// CHECK:STDOUT:   %.loc14_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14 [concrete = constants.%int_7.075]
// CHECK:STDOUT:   %.loc14_34.2: %i32 = converted %int_7, %.loc14_34.1 [concrete = constants.%int_7.075]
// CHECK:STDOUT:   %C.loc14: type = class_type @C, @C(constants.%int_7.075) [concrete = constants.%C.41b]
// CHECK:STDOUT:   %.loc14_24.2: ref %C.41b = temporary_storage
// CHECK:STDOUT:   %.loc14_24.3: init %C.41b to %.loc14_24.2 = class_init () [concrete = constants.%C.val.51d]
// CHECK:STDOUT:   %.loc14_26.1: init %C.41b = converted %.loc14_24.1, %.loc14_24.3 [concrete = constants.%C.val.51d]
// CHECK:STDOUT:   %impl.elem0.loc14_35: %.472 = impl_witness_access constants.%ImplicitAs.impl_witness.5be, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.680efc.8]
// CHECK:STDOUT:   %bound_method.loc14_35: <bound method> = bound_method %.loc14_26.1, %impl.elem0.loc14_35 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.a31]
// CHECK:STDOUT:   %.loc6_26.8: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc14_26.2: ref %C.41b = temporary %.loc14_24.2, %.loc14_26.1 [concrete = constants.%.84d]
// CHECK:STDOUT:   %.loc14_26.3: %C.41b = acquire_value %.loc14_26.2 [concrete = constants.%C.val.51d]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc14: init %D to %.loc6_26.8 = call %bound_method.loc14_35(%.loc14_26.3)
// CHECK:STDOUT:   %.loc14_35: init %D = converted %.loc14_26.1, %C.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %Destroy.Op.call.loc14_26.1: init %empty_tuple.type = call constants.%Destroy.Op.bound.d03(constants.%.84d)
// CHECK:STDOUT:   %Destroy.Op.call.loc13_26.2: init %empty_tuple.type = call constants.%Destroy.Op.bound.066(constants.%.d77)
// CHECK:STDOUT:   %Destroy.Op.call.loc12_26.3: init %empty_tuple.type = call constants.%Destroy.Op.bound.7a5(constants.%.e33)
// CHECK:STDOUT:   %Destroy.Op.call.loc11_26.4: init %empty_tuple.type = call constants.%Destroy.Op.bound.f66(constants.%.42a)
// CHECK:STDOUT:   %Destroy.Op.call.loc10_26.5: init %empty_tuple.type = call constants.%Destroy.Op.bound.a92(constants.%.f7b)
// CHECK:STDOUT:   %Destroy.Op.call.loc9_26.6: init %empty_tuple.type = call constants.%Destroy.Op.bound.601(constants.%.014)
// CHECK:STDOUT:   %Destroy.Op.call.loc8_26.7: init %empty_tuple.type = call constants.%Destroy.Op.bound.bba(constants.%.217)
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.8: init %empty_tuple.type = call constants.%Destroy.Op.bound.f55(constants.%.800)
// CHECK:STDOUT:   return %.loc14_35 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc14:
// CHECK:STDOUT:   %P.ref.loc15: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%P.Make [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc6_26.9: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc6_26.9 = call %Make.ref()
// CHECK:STDOUT:   %Destroy.Op.call.loc14_26.2: init %empty_tuple.type = call constants.%Destroy.Op.bound.d03(constants.%.84d)
// CHECK:STDOUT:   %Destroy.Op.call.loc13_26.3: init %empty_tuple.type = call constants.%Destroy.Op.bound.066(constants.%.d77)
// CHECK:STDOUT:   %Destroy.Op.call.loc12_26.4: init %empty_tuple.type = call constants.%Destroy.Op.bound.7a5(constants.%.e33)
// CHECK:STDOUT:   %Destroy.Op.call.loc11_26.5: init %empty_tuple.type = call constants.%Destroy.Op.bound.f66(constants.%.42a)
// CHECK:STDOUT:   %Destroy.Op.call.loc10_26.6: init %empty_tuple.type = call constants.%Destroy.Op.bound.a92(constants.%.f7b)
// CHECK:STDOUT:   %Destroy.Op.call.loc9_26.7: init %empty_tuple.type = call constants.%Destroy.Op.bound.601(constants.%.014)
// CHECK:STDOUT:   %Destroy.Op.call.loc8_26.8: init %empty_tuple.type = call constants.%Destroy.Op.bound.bba(constants.%.217)
// CHECK:STDOUT:   %Destroy.Op.call.loc7_26.9: init %empty_tuple.type = call constants.%Destroy.Op.bound.f55(constants.%.800)
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.1 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_26.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_26.2(%self.param: ref %C.71f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.2 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8(%self.param: ref %C.578) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.3 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9(%self.param: ref %C.e98) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.4 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10(%self.param: ref %C.4d1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.5 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11(%self.param: ref %C.b75) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.6 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12(%self.param: ref %C.b5f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.7 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13(%self.param: ref %C.91c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.8 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14(%self.param: ref %C.41b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.921) {
// CHECK:STDOUT:   %N => constants.%N.921
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.fed) {
// CHECK:STDOUT:   %N => constants.%int_0.fed
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.d72) {
// CHECK:STDOUT:   %N => constants.%int_1.d72
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.cb7) {
// CHECK:STDOUT:   %N => constants.%int_2.cb7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.781) {
// CHECK:STDOUT:   %N => constants.%int_3.781
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.d29) {
// CHECK:STDOUT:   %N => constants.%int_4.d29
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.bb1) {
// CHECK:STDOUT:   %N => constants.%int_5.bb1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.c0f) {
// CHECK:STDOUT:   %N => constants.%int_6.c0f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.075) {
// CHECK:STDOUT:   %N => constants.%int_7.075
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
