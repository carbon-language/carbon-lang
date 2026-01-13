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

impl C(0) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(1) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(2) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(3) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(4) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(5) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(6) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }
impl C(7) as Core.ImplicitAs(D) { fn Convert[self: Self]() -> D { return Make(); } }

// --- user.carbon

import P library "library";

// Check that we don't crash if we import functions and reallocate the function
// value store while handling a `return` statement.
fn F0(n: i32) -> P.D {
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
// CHECK:STDOUT:   %N.5de: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.a67: type = class_type @C, @C(%N.5de) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.de8: type = struct_type {.n: %i32, .m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ea0: <witness> = complete_type_witness %struct_type.n.m.de8 [concrete]
// CHECK:STDOUT:   %.6ef: form = init_form %D, call_param0 [concrete]
// CHECK:STDOUT:   %pattern_type.c49: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.819: type = struct_type {.n: Core.IntLiteral, .m: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.n.m.819 = struct_value (%int_0.5c6, %int_0.5c6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b94: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %.863: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.b94 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.d2e: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_0.6a9, %int_0.6a9) [concrete]
// CHECK:STDOUT:   %C.4f5: type = class_type @C, @C(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1a9: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.78c: <witness> = impl_witness @C.as.ImplicitAs.impl.c43.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.2de: type = pattern_type %C.4f5 [concrete]
// CHECK:STDOUT:   %.382: form = init_form %D, call_param1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.f45: type = fn_type @C.as.ImplicitAs.impl.Convert.loc8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.72e: %C.as.ImplicitAs.impl.Convert.type.f45 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.95b: type = class_type @C, @C(%int_1.5d2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.cf2: <witness> = impl_witness @C.as.ImplicitAs.impl.7c3.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.4f8: type = pattern_type %C.95b [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.30c: type = fn_type @C.as.ImplicitAs.impl.Convert.loc9 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.330: %C.as.ImplicitAs.impl.Convert.type.30c = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.a93: type = class_type @C, @C(%int_2.ef8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.7b9: <witness> = impl_witness @C.as.ImplicitAs.impl.fa3.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.97c: type = pattern_type %C.a93 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.9c9: type = fn_type @C.as.ImplicitAs.impl.Convert.loc10 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.4d5: %C.as.ImplicitAs.impl.Convert.type.9c9 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.1d7: type = class_type @C, @C(%int_3.822) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.377: <witness> = impl_witness @C.as.ImplicitAs.impl.158.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.00b: type = pattern_type %C.1d7 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.a26: type = fn_type @C.as.ImplicitAs.impl.Convert.loc11 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.85c: %C.as.ImplicitAs.impl.Convert.type.a26 = struct_value () [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.6d7: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %C.e40: type = class_type @C, @C(%int_4.940) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a25: <witness> = impl_witness @C.as.ImplicitAs.impl.f64.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9d7: type = pattern_type %C.e40 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.6b2: type = fn_type @C.as.ImplicitAs.impl.Convert.loc12 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.97b: %C.as.ImplicitAs.impl.Convert.type.6b2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.e9d: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %C.3f1: type = class_type @C, @C(%int_5.0f6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fff: <witness> = impl_witness @C.as.ImplicitAs.impl.d86.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.ed0: type = pattern_type %C.3f1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.6ff: type = fn_type @C.as.ImplicitAs.impl.Convert.loc13 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.d99: %C.as.ImplicitAs.impl.Convert.type.6ff = struct_value () [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.502: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.bc2: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.e56: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %C.0fa: type = class_type @C, @C(%int_6.e56) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.178: <witness> = impl_witness @C.as.ImplicitAs.impl.f13.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.6f3: type = pattern_type %C.0fa [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.969: type = fn_type @C.as.ImplicitAs.impl.Convert.loc14 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.b4c: %C.as.ImplicitAs.impl.Convert.type.969 = struct_value () [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1e0: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.bf2: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.0b1: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %C.2b7: type = class_type @C, @C(%int_7.0b1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ad8: <witness> = impl_witness @C.as.ImplicitAs.impl.83b.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.aa6: type = pattern_type %C.2b7 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.bb0: type = fn_type @C.as.ImplicitAs.impl.Convert.loc15 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.aa7: %C.as.ImplicitAs.impl.Convert.type.bb0 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.7ce = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_9.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_9.1 (constants.%N.5de)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc6_14: form = init_form %D.ref, call_param0 [concrete = constants.%.6ef]
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.c43 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc8_9.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_9.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_9.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc8_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc8_9.2: %i32 = converted %int_0, %.loc8_9.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_0.6a9) [concrete = constants.%C.4f5]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.1a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.7c3 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %impl.elem0: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc9_9.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_9.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_9.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc9_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc9_9.2: %i32 = converted %int_1, %.loc9_9.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_1.5d2) [concrete = constants.%C.95b]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.1a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.fa3 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc10_9.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_9.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.646]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc10_9.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc10_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc10_9.2: %i32 = converted %int_2, %.loc10_9.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_2.ef8) [concrete = constants.%C.a93]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.1a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.158 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc11_9.1: <bound method> = bound_method %int_3, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_9.2: <bound method> = bound_method %int_3, %specific_fn [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_9.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc11_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc11_9.2: %i32 = converted %int_3, %.loc11_9.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_3.822) [concrete = constants.%C.1d7]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.1a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.f64 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %impl.elem0: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc12_9.1: <bound method> = bound_method %int_4, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc12_9.2: <bound method> = bound_method %int_4, %specific_fn [concrete = constants.%bound_method.6d7]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_9.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %.loc12_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %.loc12_9.2: %i32 = converted %int_4, %.loc12_9.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_4.940) [concrete = constants.%C.e40]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.1a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.d86 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:     %impl.elem0: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc13_9.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc13_9.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method.e9d]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_9.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %.loc13_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %.loc13_9.2: %i32 = converted %int_5, %.loc13_9.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_5.0f6) [concrete = constants.%C.3f1]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.1a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.f13 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:     %impl.elem0: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %int_6, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.502]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %int_6, %specific_fn [concrete = constants.%bound_method.bc2]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc14_9.2(%int_6) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:     %.loc14_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_6.e56]
// CHECK:STDOUT:     %.loc14_9.2: %i32 = converted %int_6, %.loc14_9.1 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_6.e56) [concrete = constants.%C.0fa]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.1a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.83b [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:     %impl.elem0: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc15_9.1: <bound method> = bound_method %int_7, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1e0]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc15_9.2: <bound method> = bound_method %int_7, %specific_fn [concrete = constants.%bound_method.bf2]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_9.2(%int_7) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:     %.loc15_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:     %.loc15_9.2: %i32 = converted %int_7, %.loc15_9.1 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_7.0b1) [concrete = constants.%C.2b7]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.1a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.c43: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.f45 = fn_decl @C.as.ImplicitAs.impl.Convert.loc8 [concrete = constants.%C.as.ImplicitAs.impl.Convert.72e] {
// CHECK:STDOUT:     %self.patt: %pattern_type.2de = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2de = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc8_63.2: form = init_form %D.ref, call_param1 [concrete = constants.%.382]
// CHECK:STDOUT:     %self.param: %C.4f5 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.c43.%C [concrete = constants.%C.4f5]
// CHECK:STDOUT:     %self: %C.4f5 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.c43 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.78c]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.7c3: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.30c = fn_decl @C.as.ImplicitAs.impl.Convert.loc9 [concrete = constants.%C.as.ImplicitAs.impl.Convert.330] {
// CHECK:STDOUT:     %self.patt: %pattern_type.4f8 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.4f8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc9_63.2: form = init_form %D.ref, call_param1 [concrete = constants.%.382]
// CHECK:STDOUT:     %self.param: %C.95b = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.7c3.%C [concrete = constants.%C.95b]
// CHECK:STDOUT:     %self: %C.95b = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.7c3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.cf2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.fa3: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.9c9 = fn_decl @C.as.ImplicitAs.impl.Convert.loc10 [concrete = constants.%C.as.ImplicitAs.impl.Convert.4d5] {
// CHECK:STDOUT:     %self.patt: %pattern_type.97c = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.97c = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc10_63.2: form = init_form %D.ref, call_param1 [concrete = constants.%.382]
// CHECK:STDOUT:     %self.param: %C.a93 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.fa3.%C [concrete = constants.%C.a93]
// CHECK:STDOUT:     %self: %C.a93 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.fa3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.7b9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.158: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.a26 = fn_decl @C.as.ImplicitAs.impl.Convert.loc11 [concrete = constants.%C.as.ImplicitAs.impl.Convert.85c] {
// CHECK:STDOUT:     %self.patt: %pattern_type.00b = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.00b = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc11_63.2: form = init_form %D.ref, call_param1 [concrete = constants.%.382]
// CHECK:STDOUT:     %self.param: %C.1d7 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.158.%C [concrete = constants.%C.1d7]
// CHECK:STDOUT:     %self: %C.1d7 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.158 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.377]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.f64: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.6b2 = fn_decl @C.as.ImplicitAs.impl.Convert.loc12 [concrete = constants.%C.as.ImplicitAs.impl.Convert.97b] {
// CHECK:STDOUT:     %self.patt: %pattern_type.9d7 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d7 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc12_63.2: form = init_form %D.ref, call_param1 [concrete = constants.%.382]
// CHECK:STDOUT:     %self.param: %C.e40 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.f64.%C [concrete = constants.%C.e40]
// CHECK:STDOUT:     %self: %C.e40 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.f64 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.a25]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.d86: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.6ff = fn_decl @C.as.ImplicitAs.impl.Convert.loc13 [concrete = constants.%C.as.ImplicitAs.impl.Convert.d99] {
// CHECK:STDOUT:     %self.patt: %pattern_type.ed0 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ed0 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc13_63.2: form = init_form %D.ref, call_param1 [concrete = constants.%.382]
// CHECK:STDOUT:     %self.param: %C.3f1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.d86.%C [concrete = constants.%C.3f1]
// CHECK:STDOUT:     %self: %C.3f1 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.d86 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.fff]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.f13: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.969 = fn_decl @C.as.ImplicitAs.impl.Convert.loc14 [concrete = constants.%C.as.ImplicitAs.impl.Convert.b4c] {
// CHECK:STDOUT:     %self.patt: %pattern_type.6f3 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6f3 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc14_63.2: form = init_form %D.ref, call_param1 [concrete = constants.%.382]
// CHECK:STDOUT:     %self.param: %C.0fa = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.f13.%C [concrete = constants.%C.0fa]
// CHECK:STDOUT:     %self: %C.0fa = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.f13 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.178]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.83b: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.bb0 = fn_decl @C.as.ImplicitAs.impl.Convert.loc15 [concrete = constants.%C.as.ImplicitAs.impl.Convert.aa7] {
// CHECK:STDOUT:     %self.patt: %pattern_type.aa6 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.aa6 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc15_63.2: form = init_form %D.ref, call_param1 [concrete = constants.%.382]
// CHECK:STDOUT:     %self.param: %C.2b7 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.83b.%C [concrete = constants.%C.2b7]
// CHECK:STDOUT:     %self: %C.2b7 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.83b [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.ad8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%N.loc4_9.2: %i32) {
// CHECK:STDOUT:   %N.loc4_9.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_9.1 (constants.%N.5de)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.a67
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %int_32.loc5_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5_16: %D.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc5_30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_30: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5_28: %D.elem = field_decl m, element1 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n.m.de8 [concrete = constants.%complete_type.ea0]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .n = %.loc5_16
// CHECK:STDOUT:   .m = %.loc5_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc6_31: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_0.loc6_39: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc6_40.1: %struct_type.n.m.819 = struct_literal (%int_0.loc6_31, %int_0.loc6_39) [concrete = constants.%struct]
// CHECK:STDOUT:   %impl.elem0.loc6_40.1: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc6_40.1: <bound method> = bound_method %int_0.loc6_31, %impl.elem0.loc6_40.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc6_40.1: <specific function> = specific_function %impl.elem0.loc6_40.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.2: <bound method> = bound_method %int_0.loc6_31, %specific_fn.loc6_40.1 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.1: init %i32 = call %bound_method.loc6_40.2(%int_0.loc6_31) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.2: init %i32 = converted %int_0.loc6_31, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.3: ref %i32 = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc6_40.4: init %i32 = initialize_from %.loc6_40.2 to %.loc6_40.3 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %impl.elem0.loc6_40.2: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc6_40.3: <bound method> = bound_method %int_0.loc6_39, %impl.elem0.loc6_40.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc6_40.2: <specific function> = specific_function %impl.elem0.loc6_40.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.4: <bound method> = bound_method %int_0.loc6_39, %specific_fn.loc6_40.2 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.2: init %i32 = call %bound_method.loc6_40.4(%int_0.loc6_39) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.5: init %i32 = converted %int_0.loc6_39, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.2 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.6: ref %i32 = class_element_access %return.param, element1
// CHECK:STDOUT:   %.loc6_40.7: init %i32 = initialize_from %.loc6_40.5 to %.loc6_40.6 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.8: init %D = class_init (%.loc6_40.4, %.loc6_40.7), %return.param [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc6_41: init %D = converted %.loc6_40.1, %.loc6_40.8 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc6_41 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc8(%self.param: %C.4f5) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc8_63.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc8_63.1
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc9(%self.param: %C.95b) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc9_63.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc9_63.1
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc10(%self.param: %C.a93) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc10_63.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc10_63.1
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc11(%self.param: %C.1d7) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc11_63.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc11_63.1
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc12(%self.param: %C.e40) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc12_63.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc12_63.1
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc13(%self.param: %C.3f1) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13_63.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc13_63.1
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc14(%self.param: %C.0fa) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14_63.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc14_63.1
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc15(%self.param: %C.2b7) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc15_63.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc15_63.1
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.5de) {
// CHECK:STDOUT:   %N.loc4_9.1 => constants.%N.5de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.6a9) {
// CHECK:STDOUT:   %N.loc4_9.1 => constants.%int_0.6a9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.5d2) {
// CHECK:STDOUT:   %N.loc4_9.1 => constants.%int_1.5d2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.ef8) {
// CHECK:STDOUT:   %N.loc4_9.1 => constants.%int_2.ef8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.822) {
// CHECK:STDOUT:   %N.loc4_9.1 => constants.%int_3.822
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.940) {
// CHECK:STDOUT:   %N.loc4_9.1 => constants.%int_4.940
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.0f6) {
// CHECK:STDOUT:   %N.loc4_9.1 => constants.%int_5.0f6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.e56) {
// CHECK:STDOUT:   %N.loc4_9.1 => constants.%int_6.e56
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.0b1) {
// CHECK:STDOUT:   %N.loc4_9.1 => constants.%int_7.0b1
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
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %struct_type.n.m: type = struct_type {.n: %i32, .m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ea0: <witness> = complete_type_witness %struct_type.n.m [concrete]
// CHECK:STDOUT:   %.6c0: form = init_form %D, call_param1 [concrete]
// CHECK:STDOUT:   %pattern_type.f8a: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %F0.type: type = fn_type @F0 [concrete]
// CHECK:STDOUT:   %F0: %F0.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %N.5de: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b94: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %.863: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.b94 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.d2e: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %C.b00: type = class_type @C, @C(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %C.val.452: %C.b00 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.899: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.334: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.674: type = class_type @C, @C(%int_1.5d2) [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.681: type = class_type @C, @C(%int_2.ef8) [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.7ac: type = class_type @C, @C(%int_3.822) [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %C.89d: type = class_type @C, @C(%int_4.940) [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %C.f0a: type = class_type @C, @C(%int_5.0f6) [concrete]
// CHECK:STDOUT:   %int_6.e56: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %C.c60: type = class_type @C, @C(%int_6.e56) [concrete]
// CHECK:STDOUT:   %int_7.0b1: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %C.304: type = class_type @C, @C(%int_7.0b1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.01b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1e8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.774: %ImplicitAs.type.899 = facet_value %C.b00, (%ImplicitAs.impl_witness.01b) [concrete]
// CHECK:STDOUT:   %.094: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.774 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.994: type = fn_type @C.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.78d: %C.as.ImplicitAs.impl.Convert.type.994 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc7 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.678: %C.674 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.638: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c5c [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.0f6: %ImplicitAs.type.899 = facet_value %C.674, (%ImplicitAs.impl_witness.638) [concrete]
// CHECK:STDOUT:   %.deb: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.0f6 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.900: type = fn_type @C.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.b44: %C.as.ImplicitAs.impl.Convert.type.900 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.2: type = fn_type @DestroyOp.loc8 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.2: %DestroyOp.type.3e79c2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.fb5: %C.681 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a8d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5db [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.53b: %ImplicitAs.type.899 = facet_value %C.681, (%ImplicitAs.impl_witness.a8d) [concrete]
// CHECK:STDOUT:   %.8f9: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.53b [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae1: type = fn_type @C.as.ImplicitAs.impl.Convert.3 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.dd6: %C.as.ImplicitAs.impl.Convert.type.ae1 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.3: type = fn_type @DestroyOp.loc9 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.3: %DestroyOp.type.3e79c2.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.fe7: %C.7ac = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.56b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.126 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fe0: %ImplicitAs.type.899 = facet_value %C.7ac, (%ImplicitAs.impl_witness.56b) [concrete]
// CHECK:STDOUT:   %.286: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.fe0 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.cbe: type = fn_type @C.as.ImplicitAs.impl.Convert.4 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.d1c: %C.as.ImplicitAs.impl.Convert.type.cbe = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.4: type = fn_type @DestroyOp.loc10 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.4: %DestroyOp.type.3e79c2.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.6d7: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.1dd: %C.89d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e05: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.92c [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fd8: %ImplicitAs.type.899 = facet_value %C.89d, (%ImplicitAs.impl_witness.e05) [concrete]
// CHECK:STDOUT:   %.3c4: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.fd8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.876: type = fn_type @C.as.ImplicitAs.impl.Convert.5 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.db5: %C.as.ImplicitAs.impl.Convert.type.876 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.5: type = fn_type @DestroyOp.loc11 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.5: %DestroyOp.type.3e79c2.5 = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.e9d: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.a4a: %C.f0a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.262: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d8f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.357: %ImplicitAs.type.899 = facet_value %C.f0a, (%ImplicitAs.impl_witness.262) [concrete]
// CHECK:STDOUT:   %.4d7: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.357 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.043: type = fn_type @C.as.ImplicitAs.impl.Convert.6 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.a63: %C.as.ImplicitAs.impl.Convert.type.043 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.6: type = fn_type @DestroyOp.loc12 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.6: %DestroyOp.type.3e79c2.6 = struct_value () [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.502: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.bc2: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.a4b: %C.c60 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.7d1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.70a [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.510: %ImplicitAs.type.899 = facet_value %C.c60, (%ImplicitAs.impl_witness.7d1) [concrete]
// CHECK:STDOUT:   %.3aa: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.510 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.7be: type = fn_type @C.as.ImplicitAs.impl.Convert.7 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.e2f: %C.as.ImplicitAs.impl.Convert.type.7be = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.7: type = fn_type @DestroyOp.loc13 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.7: %DestroyOp.type.3e79c2.7 = struct_value () [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1e0: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.bf2: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.f9f: %C.304 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.356: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.575 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.965: %ImplicitAs.type.899 = facet_value %C.304, (%ImplicitAs.impl_witness.356) [concrete]
// CHECK:STDOUT:   %.dad: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.965 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.c30: type = fn_type @C.as.ImplicitAs.impl.Convert.8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.5ea: %C.as.ImplicitAs.impl.Convert.type.c30 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.8: type = fn_type @DestroyOp.loc14 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.8: %DestroyOp.type.3e79c2.8 = struct_value () [concrete]
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
// CHECK:STDOUT:   %P.import_ref.a7d: <witness> = import_ref P//library, loc5_35, loaded [concrete = constants.%complete_type.ea0]
// CHECK:STDOUT:   %P.import_ref.a4d = import_ref P//library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.f32 = import_ref P//library, loc5_16, unloaded
// CHECK:STDOUT:   %P.import_ref.eba = import_ref P//library, loc5_28, unloaded
// CHECK:STDOUT:   %P.C: %C.type = import_ref P//library, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %P.import_ref.8f2: <witness> = import_ref P//library, loc4_19, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %P.import_ref.01d = import_ref P//library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.67b: %i32 = import_ref P//library, loc4_9, loaded [symbolic = @C.%N (constants.%N.5de)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %P.import_ref.baa: <witness> = import_ref P//library, loc8_33, loaded [concrete = constants.%ImplicitAs.impl_witness.01b]
// CHECK:STDOUT:   %P.import_ref.a2e: type = import_ref P//library, loc8_9, loaded [concrete = constants.%C.b00]
// CHECK:STDOUT:   %P.import_ref.68d5e2.1: type = import_ref P//library, loc8_31, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.367: <witness> = import_ref P//library, loc9_33, loaded [concrete = constants.%ImplicitAs.impl_witness.638]
// CHECK:STDOUT:   %P.import_ref.203: type = import_ref P//library, loc9_9, loaded [concrete = constants.%C.674]
// CHECK:STDOUT:   %P.import_ref.68d5e2.2: type = import_ref P//library, loc9_31, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.9fc: <witness> = import_ref P//library, loc10_33, loaded [concrete = constants.%ImplicitAs.impl_witness.a8d]
// CHECK:STDOUT:   %P.import_ref.956: type = import_ref P//library, loc10_9, loaded [concrete = constants.%C.681]
// CHECK:STDOUT:   %P.import_ref.68d5e2.3: type = import_ref P//library, loc10_31, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.b266: <witness> = import_ref P//library, loc11_33, loaded [concrete = constants.%ImplicitAs.impl_witness.56b]
// CHECK:STDOUT:   %P.import_ref.51c: type = import_ref P//library, loc11_9, loaded [concrete = constants.%C.7ac]
// CHECK:STDOUT:   %P.import_ref.68d5e2.4: type = import_ref P//library, loc11_31, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.442: <witness> = import_ref P//library, loc12_33, loaded [concrete = constants.%ImplicitAs.impl_witness.e05]
// CHECK:STDOUT:   %P.import_ref.883: type = import_ref P//library, loc12_9, loaded [concrete = constants.%C.89d]
// CHECK:STDOUT:   %P.import_ref.68d5e2.5: type = import_ref P//library, loc12_31, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.766: <witness> = import_ref P//library, loc13_33, loaded [concrete = constants.%ImplicitAs.impl_witness.262]
// CHECK:STDOUT:   %P.import_ref.0c8: type = import_ref P//library, loc13_9, loaded [concrete = constants.%C.f0a]
// CHECK:STDOUT:   %P.import_ref.68d5e2.6: type = import_ref P//library, loc13_31, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.63e: <witness> = import_ref P//library, loc14_33, loaded [concrete = constants.%ImplicitAs.impl_witness.7d1]
// CHECK:STDOUT:   %P.import_ref.92b: type = import_ref P//library, loc14_9, loaded [concrete = constants.%C.c60]
// CHECK:STDOUT:   %P.import_ref.68d5e2.7: type = import_ref P//library, loc14_31, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.24c: <witness> = import_ref P//library, loc15_33, loaded [concrete = constants.%ImplicitAs.impl_witness.356]
// CHECK:STDOUT:   %P.import_ref.854: type = import_ref P//library, loc15_9, loaded [concrete = constants.%C.304]
// CHECK:STDOUT:   %P.import_ref.68d5e2.8: type = import_ref P//library, loc15_31, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.623: %C.as.ImplicitAs.impl.Convert.type.994 = import_ref P//library, loc8_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.78d]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1e8 = impl_witness_table (%P.import_ref.623), @C.as.ImplicitAs.impl.87c [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %P.import_ref.a4a: %C.as.ImplicitAs.impl.Convert.type.900 = import_ref P//library, loc9_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.b44]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c5c = impl_witness_table (%P.import_ref.a4a), @C.as.ImplicitAs.impl.a01 [concrete]
// CHECK:STDOUT:   %P.import_ref.e4a: %C.as.ImplicitAs.impl.Convert.type.ae1 = import_ref P//library, loc10_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.dd6]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5db = impl_witness_table (%P.import_ref.e4a), @C.as.ImplicitAs.impl.390 [concrete]
// CHECK:STDOUT:   %P.import_ref.b26b: %C.as.ImplicitAs.impl.Convert.type.cbe = import_ref P//library, loc11_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.d1c]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.126 = impl_witness_table (%P.import_ref.b26b), @C.as.ImplicitAs.impl.071 [concrete]
// CHECK:STDOUT:   %P.import_ref.6da: %C.as.ImplicitAs.impl.Convert.type.876 = import_ref P//library, loc12_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.db5]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.92c = impl_witness_table (%P.import_ref.6da), @C.as.ImplicitAs.impl.891 [concrete]
// CHECK:STDOUT:   %P.import_ref.200: %C.as.ImplicitAs.impl.Convert.type.043 = import_ref P//library, loc13_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.a63]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d8f = impl_witness_table (%P.import_ref.200), @C.as.ImplicitAs.impl.882 [concrete]
// CHECK:STDOUT:   %P.import_ref.010: %C.as.ImplicitAs.impl.Convert.type.7be = import_ref P//library, loc14_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.e2f]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.70a = impl_witness_table (%P.import_ref.010), @C.as.ImplicitAs.impl.ad9 [concrete]
// CHECK:STDOUT:   %P.import_ref.624: %C.as.ImplicitAs.impl.Convert.type.c30 = import_ref P//library, loc15_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.5ea]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.575 = impl_witness_table (%P.import_ref.624), @C.as.ImplicitAs.impl.5d4 [concrete]
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
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = value_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.f8a = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.f8a = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %P.ref.loc6: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%P.D [concrete = constants.%D]
// CHECK:STDOUT:     %.loc6_19.10: form = init_form %D.ref, call_param1 [concrete = constants.%.6c0]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc6_10: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.87c: imports.%P.import_ref.a2e as imports.%P.import_ref.68d5e2.1 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.baa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.a01: imports.%P.import_ref.203 as imports.%P.import_ref.68d5e2.2 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.367
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.390: imports.%P.import_ref.956 as imports.%P.import_ref.68d5e2.3 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.9fc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.071: imports.%P.import_ref.51c as imports.%P.import_ref.68d5e2.4 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.b266
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.891: imports.%P.import_ref.883 as imports.%P.import_ref.68d5e2.5 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.442
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.882: imports.%P.import_ref.0c8 as imports.%P.import_ref.68d5e2.6 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.766
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.ad9: imports.%P.import_ref.92b as imports.%P.import_ref.68d5e2.7 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.63e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.5d4: imports.%P.import_ref.854 as imports.%P.import_ref.68d5e2.8 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.24c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "library.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%P.import_ref.a7d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%P.import_ref.a4d
// CHECK:STDOUT:   .n = imports.%P.import_ref.f32
// CHECK:STDOUT:   .m = imports.%P.import_ref.eba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%P.import_ref.67b: %i32) [from "library.carbon"] {
// CHECK:STDOUT:   %N: %i32 = symbolic_binding N, 0 [symbolic = %N (constants.%N.5de)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%P.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%P.import_ref.01d
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F0(%n.param: %i32) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %false.loc7: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc7 br !if.then.loc7 else br !if.else.loc7
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc7:
// CHECK:STDOUT:   %.loc7_24.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc7: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc7: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc7_34: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc7_34.1: <bound method> = bound_method %int_0, %impl.elem0.loc7_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_34.2: <bound method> = bound_method %int_0, %specific_fn.loc7 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_34.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc7_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc7_34.2: %i32 = converted %int_0, %.loc7_34.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %C.loc7: type = class_type @C, @C(constants.%int_0.6a9) [concrete = constants.%C.b00]
// CHECK:STDOUT:   %.loc7_24.2: ref %C.b00 = temporary_storage
// CHECK:STDOUT:   %.loc7_24.3: init %C.b00 = class_init (), %.loc7_24.2 [concrete = constants.%C.val.452]
// CHECK:STDOUT:   %.loc7_24.4: ref %C.b00 = temporary %.loc7_24.2, %.loc7_24.3
// CHECK:STDOUT:   %.loc7_26.1: ref %C.b00 = converted %.loc7_24.1, %.loc7_24.4
// CHECK:STDOUT:   %impl.elem0.loc7_35: %.094 = impl_witness_access constants.%ImplicitAs.impl_witness.01b, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.78d]
// CHECK:STDOUT:   %bound_method.loc7_35: <bound method> = bound_method %.loc7_26.1, %impl.elem0.loc7_35
// CHECK:STDOUT:   %.loc6_19.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc7_26.2: %C.b00 = acquire_value %.loc7_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc7: init %D = call %bound_method.loc7_35(%.loc7_26.2) to %.loc6_19.1
// CHECK:STDOUT:   %.loc7_35: init %D = converted %.loc7_26.1, %C.as.ImplicitAs.impl.Convert.call.loc7
// CHECK:STDOUT:   %DestroyOp.bound.loc7_24.1: <bound method> = bound_method %.loc7_24.4, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc7_24.1: init %empty_tuple.type = call %DestroyOp.bound.loc7_24.1(%.loc7_24.4)
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
// CHECK:STDOUT:   %impl.elem0.loc8_34: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_1, %specific_fn.loc8 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_34.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_1, %.loc8_34.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %C.loc8: type = class_type @C, @C(constants.%int_1.5d2) [concrete = constants.%C.674]
// CHECK:STDOUT:   %.loc8_24.2: ref %C.674 = temporary_storage
// CHECK:STDOUT:   %.loc8_24.3: init %C.674 = class_init (), %.loc8_24.2 [concrete = constants.%C.val.678]
// CHECK:STDOUT:   %.loc8_24.4: ref %C.674 = temporary %.loc8_24.2, %.loc8_24.3
// CHECK:STDOUT:   %.loc8_26.1: ref %C.674 = converted %.loc8_24.1, %.loc8_24.4
// CHECK:STDOUT:   %impl.elem0.loc8_35: %.deb = impl_witness_access constants.%ImplicitAs.impl_witness.638, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.b44]
// CHECK:STDOUT:   %bound_method.loc8_35: <bound method> = bound_method %.loc8_26.1, %impl.elem0.loc8_35
// CHECK:STDOUT:   %.loc6_19.2: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc8_26.2: %C.674 = acquire_value %.loc8_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc8: init %D = call %bound_method.loc8_35(%.loc8_26.2) to %.loc6_19.2
// CHECK:STDOUT:   %.loc8_35: init %D = converted %.loc8_26.1, %C.as.ImplicitAs.impl.Convert.call.loc8
// CHECK:STDOUT:   %DestroyOp.bound.loc8_24.1: <bound method> = bound_method %.loc8_24.4, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc8_24.1: init %empty_tuple.type = call %DestroyOp.bound.loc8_24.1(%.loc8_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc7_24.2: <bound method> = bound_method %.loc7_24.4, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc7_24.2: init %empty_tuple.type = call %DestroyOp.bound.loc7_24.2(%.loc7_24.4)
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
// CHECK:STDOUT:   %impl.elem0.loc9_34: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %int_2, %impl.elem0.loc9_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %int_2, %specific_fn.loc9 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_34.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_34.2: %i32 = converted %int_2, %.loc9_34.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %C.loc9: type = class_type @C, @C(constants.%int_2.ef8) [concrete = constants.%C.681]
// CHECK:STDOUT:   %.loc9_24.2: ref %C.681 = temporary_storage
// CHECK:STDOUT:   %.loc9_24.3: init %C.681 = class_init (), %.loc9_24.2 [concrete = constants.%C.val.fb5]
// CHECK:STDOUT:   %.loc9_24.4: ref %C.681 = temporary %.loc9_24.2, %.loc9_24.3
// CHECK:STDOUT:   %.loc9_26.1: ref %C.681 = converted %.loc9_24.1, %.loc9_24.4
// CHECK:STDOUT:   %impl.elem0.loc9_35: %.8f9 = impl_witness_access constants.%ImplicitAs.impl_witness.a8d, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.dd6]
// CHECK:STDOUT:   %bound_method.loc9_35: <bound method> = bound_method %.loc9_26.1, %impl.elem0.loc9_35
// CHECK:STDOUT:   %.loc6_19.3: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc9_26.2: %C.681 = acquire_value %.loc9_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc9: init %D = call %bound_method.loc9_35(%.loc9_26.2) to %.loc6_19.3
// CHECK:STDOUT:   %.loc9_35: init %D = converted %.loc9_26.1, %C.as.ImplicitAs.impl.Convert.call.loc9
// CHECK:STDOUT:   %DestroyOp.bound.loc9_24.1: <bound method> = bound_method %.loc9_24.4, constants.%DestroyOp.b0ebf8.3
// CHECK:STDOUT:   %DestroyOp.call.loc9_24.1: init %empty_tuple.type = call %DestroyOp.bound.loc9_24.1(%.loc9_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc8_24.2: <bound method> = bound_method %.loc8_24.4, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc8_24.2: init %empty_tuple.type = call %DestroyOp.bound.loc8_24.2(%.loc8_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc7_24.3: <bound method> = bound_method %.loc7_24.4, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc7_24.3: init %empty_tuple.type = call %DestroyOp.bound.loc7_24.3(%.loc7_24.4)
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
// CHECK:STDOUT:   %impl.elem0.loc10_34: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc10_34.1: <bound method> = bound_method %int_3, %impl.elem0.loc10_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_34.2: <bound method> = bound_method %int_3, %specific_fn.loc10 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %i32 = call %bound_method.loc10_34.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc10_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc10_34.2: %i32 = converted %int_3, %.loc10_34.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %C.loc10: type = class_type @C, @C(constants.%int_3.822) [concrete = constants.%C.7ac]
// CHECK:STDOUT:   %.loc10_24.2: ref %C.7ac = temporary_storage
// CHECK:STDOUT:   %.loc10_24.3: init %C.7ac = class_init (), %.loc10_24.2 [concrete = constants.%C.val.fe7]
// CHECK:STDOUT:   %.loc10_24.4: ref %C.7ac = temporary %.loc10_24.2, %.loc10_24.3
// CHECK:STDOUT:   %.loc10_26.1: ref %C.7ac = converted %.loc10_24.1, %.loc10_24.4
// CHECK:STDOUT:   %impl.elem0.loc10_35: %.286 = impl_witness_access constants.%ImplicitAs.impl_witness.56b, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.d1c]
// CHECK:STDOUT:   %bound_method.loc10_35: <bound method> = bound_method %.loc10_26.1, %impl.elem0.loc10_35
// CHECK:STDOUT:   %.loc6_19.4: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc10_26.2: %C.7ac = acquire_value %.loc10_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc10: init %D = call %bound_method.loc10_35(%.loc10_26.2) to %.loc6_19.4
// CHECK:STDOUT:   %.loc10_35: init %D = converted %.loc10_26.1, %C.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %DestroyOp.bound.loc10_24.1: <bound method> = bound_method %.loc10_24.4, constants.%DestroyOp.b0ebf8.4
// CHECK:STDOUT:   %DestroyOp.call.loc10_24.1: init %empty_tuple.type = call %DestroyOp.bound.loc10_24.1(%.loc10_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc9_24.2: <bound method> = bound_method %.loc9_24.4, constants.%DestroyOp.b0ebf8.3
// CHECK:STDOUT:   %DestroyOp.call.loc9_24.2: init %empty_tuple.type = call %DestroyOp.bound.loc9_24.2(%.loc9_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc8_24.3: <bound method> = bound_method %.loc8_24.4, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc8_24.3: init %empty_tuple.type = call %DestroyOp.bound.loc8_24.3(%.loc8_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc7_24.4: <bound method> = bound_method %.loc7_24.4, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc7_24.4: init %empty_tuple.type = call %DestroyOp.bound.loc7_24.4(%.loc7_24.4)
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
// CHECK:STDOUT:   %impl.elem0.loc11_34: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc11_34.1: <bound method> = bound_method %int_4, %impl.elem0.loc11_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_34.2: <bound method> = bound_method %int_4, %specific_fn.loc11 [concrete = constants.%bound_method.6d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %i32 = call %bound_method.loc11_34.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc11_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc11_34.2: %i32 = converted %int_4, %.loc11_34.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %C.loc11: type = class_type @C, @C(constants.%int_4.940) [concrete = constants.%C.89d]
// CHECK:STDOUT:   %.loc11_24.2: ref %C.89d = temporary_storage
// CHECK:STDOUT:   %.loc11_24.3: init %C.89d = class_init (), %.loc11_24.2 [concrete = constants.%C.val.1dd]
// CHECK:STDOUT:   %.loc11_24.4: ref %C.89d = temporary %.loc11_24.2, %.loc11_24.3
// CHECK:STDOUT:   %.loc11_26.1: ref %C.89d = converted %.loc11_24.1, %.loc11_24.4
// CHECK:STDOUT:   %impl.elem0.loc11_35: %.3c4 = impl_witness_access constants.%ImplicitAs.impl_witness.e05, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.db5]
// CHECK:STDOUT:   %bound_method.loc11_35: <bound method> = bound_method %.loc11_26.1, %impl.elem0.loc11_35
// CHECK:STDOUT:   %.loc6_19.5: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc11_26.2: %C.89d = acquire_value %.loc11_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc11: init %D = call %bound_method.loc11_35(%.loc11_26.2) to %.loc6_19.5
// CHECK:STDOUT:   %.loc11_35: init %D = converted %.loc11_26.1, %C.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %DestroyOp.bound.loc11_24.1: <bound method> = bound_method %.loc11_24.4, constants.%DestroyOp.b0ebf8.5
// CHECK:STDOUT:   %DestroyOp.call.loc11_24.1: init %empty_tuple.type = call %DestroyOp.bound.loc11_24.1(%.loc11_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_24.2: <bound method> = bound_method %.loc10_24.4, constants.%DestroyOp.b0ebf8.4
// CHECK:STDOUT:   %DestroyOp.call.loc10_24.2: init %empty_tuple.type = call %DestroyOp.bound.loc10_24.2(%.loc10_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc9_24.3: <bound method> = bound_method %.loc9_24.4, constants.%DestroyOp.b0ebf8.3
// CHECK:STDOUT:   %DestroyOp.call.loc9_24.3: init %empty_tuple.type = call %DestroyOp.bound.loc9_24.3(%.loc9_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc8_24.4: <bound method> = bound_method %.loc8_24.4, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc8_24.4: init %empty_tuple.type = call %DestroyOp.bound.loc8_24.4(%.loc8_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc7_24.5: <bound method> = bound_method %.loc7_24.4, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc7_24.5: init %empty_tuple.type = call %DestroyOp.bound.loc7_24.5(%.loc7_24.4)
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
// CHECK:STDOUT:   %impl.elem0.loc12_34: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc12_34.1: <bound method> = bound_method %int_5, %impl.elem0.loc12_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_34.2: <bound method> = bound_method %int_5, %specific_fn.loc12 [concrete = constants.%bound_method.e9d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %i32 = call %bound_method.loc12_34.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc12_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc12_34.2: %i32 = converted %int_5, %.loc12_34.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %C.loc12: type = class_type @C, @C(constants.%int_5.0f6) [concrete = constants.%C.f0a]
// CHECK:STDOUT:   %.loc12_24.2: ref %C.f0a = temporary_storage
// CHECK:STDOUT:   %.loc12_24.3: init %C.f0a = class_init (), %.loc12_24.2 [concrete = constants.%C.val.a4a]
// CHECK:STDOUT:   %.loc12_24.4: ref %C.f0a = temporary %.loc12_24.2, %.loc12_24.3
// CHECK:STDOUT:   %.loc12_26.1: ref %C.f0a = converted %.loc12_24.1, %.loc12_24.4
// CHECK:STDOUT:   %impl.elem0.loc12_35: %.4d7 = impl_witness_access constants.%ImplicitAs.impl_witness.262, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.a63]
// CHECK:STDOUT:   %bound_method.loc12_35: <bound method> = bound_method %.loc12_26.1, %impl.elem0.loc12_35
// CHECK:STDOUT:   %.loc6_19.6: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc12_26.2: %C.f0a = acquire_value %.loc12_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc12: init %D = call %bound_method.loc12_35(%.loc12_26.2) to %.loc6_19.6
// CHECK:STDOUT:   %.loc12_35: init %D = converted %.loc12_26.1, %C.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %DestroyOp.bound.loc12_24.1: <bound method> = bound_method %.loc12_24.4, constants.%DestroyOp.b0ebf8.6
// CHECK:STDOUT:   %DestroyOp.call.loc12_24.1: init %empty_tuple.type = call %DestroyOp.bound.loc12_24.1(%.loc12_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc11_24.2: <bound method> = bound_method %.loc11_24.4, constants.%DestroyOp.b0ebf8.5
// CHECK:STDOUT:   %DestroyOp.call.loc11_24.2: init %empty_tuple.type = call %DestroyOp.bound.loc11_24.2(%.loc11_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_24.3: <bound method> = bound_method %.loc10_24.4, constants.%DestroyOp.b0ebf8.4
// CHECK:STDOUT:   %DestroyOp.call.loc10_24.3: init %empty_tuple.type = call %DestroyOp.bound.loc10_24.3(%.loc10_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc9_24.4: <bound method> = bound_method %.loc9_24.4, constants.%DestroyOp.b0ebf8.3
// CHECK:STDOUT:   %DestroyOp.call.loc9_24.4: init %empty_tuple.type = call %DestroyOp.bound.loc9_24.4(%.loc9_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc8_24.5: <bound method> = bound_method %.loc8_24.4, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc8_24.5: init %empty_tuple.type = call %DestroyOp.bound.loc8_24.5(%.loc8_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc7_24.6: <bound method> = bound_method %.loc7_24.4, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc7_24.6: init %empty_tuple.type = call %DestroyOp.bound.loc7_24.6(%.loc7_24.4)
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
// CHECK:STDOUT:   %impl.elem0.loc13_34: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc13_34.1: <bound method> = bound_method %int_6, %impl.elem0.loc13_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.502]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_34.2: <bound method> = bound_method %int_6, %specific_fn.loc13 [concrete = constants.%bound_method.bc2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %i32 = call %bound_method.loc13_34.2(%int_6) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc13_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc13_34.2: %i32 = converted %int_6, %.loc13_34.1 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %C.loc13: type = class_type @C, @C(constants.%int_6.e56) [concrete = constants.%C.c60]
// CHECK:STDOUT:   %.loc13_24.2: ref %C.c60 = temporary_storage
// CHECK:STDOUT:   %.loc13_24.3: init %C.c60 = class_init (), %.loc13_24.2 [concrete = constants.%C.val.a4b]
// CHECK:STDOUT:   %.loc13_24.4: ref %C.c60 = temporary %.loc13_24.2, %.loc13_24.3
// CHECK:STDOUT:   %.loc13_26.1: ref %C.c60 = converted %.loc13_24.1, %.loc13_24.4
// CHECK:STDOUT:   %impl.elem0.loc13_35: %.3aa = impl_witness_access constants.%ImplicitAs.impl_witness.7d1, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.e2f]
// CHECK:STDOUT:   %bound_method.loc13_35: <bound method> = bound_method %.loc13_26.1, %impl.elem0.loc13_35
// CHECK:STDOUT:   %.loc6_19.7: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc13_26.2: %C.c60 = acquire_value %.loc13_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc13: init %D = call %bound_method.loc13_35(%.loc13_26.2) to %.loc6_19.7
// CHECK:STDOUT:   %.loc13_35: init %D = converted %.loc13_26.1, %C.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %DestroyOp.bound.loc13_24.1: <bound method> = bound_method %.loc13_24.4, constants.%DestroyOp.b0ebf8.7
// CHECK:STDOUT:   %DestroyOp.call.loc13_24.1: init %empty_tuple.type = call %DestroyOp.bound.loc13_24.1(%.loc13_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc12_24.2: <bound method> = bound_method %.loc12_24.4, constants.%DestroyOp.b0ebf8.6
// CHECK:STDOUT:   %DestroyOp.call.loc12_24.2: init %empty_tuple.type = call %DestroyOp.bound.loc12_24.2(%.loc12_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc11_24.3: <bound method> = bound_method %.loc11_24.4, constants.%DestroyOp.b0ebf8.5
// CHECK:STDOUT:   %DestroyOp.call.loc11_24.3: init %empty_tuple.type = call %DestroyOp.bound.loc11_24.3(%.loc11_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_24.4: <bound method> = bound_method %.loc10_24.4, constants.%DestroyOp.b0ebf8.4
// CHECK:STDOUT:   %DestroyOp.call.loc10_24.4: init %empty_tuple.type = call %DestroyOp.bound.loc10_24.4(%.loc10_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc9_24.5: <bound method> = bound_method %.loc9_24.4, constants.%DestroyOp.b0ebf8.3
// CHECK:STDOUT:   %DestroyOp.call.loc9_24.5: init %empty_tuple.type = call %DestroyOp.bound.loc9_24.5(%.loc9_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc8_24.6: <bound method> = bound_method %.loc8_24.4, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc8_24.6: init %empty_tuple.type = call %DestroyOp.bound.loc8_24.6(%.loc8_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc7_24.7: <bound method> = bound_method %.loc7_24.4, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc7_24.7: init %empty_tuple.type = call %DestroyOp.bound.loc7_24.7(%.loc7_24.4)
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
// CHECK:STDOUT:   %impl.elem0.loc14_34: %.863 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc14_34.1: <bound method> = bound_method %int_7, %impl.elem0.loc14_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1e0]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.2: <bound method> = bound_method %int_7, %specific_fn.loc14 [concrete = constants.%bound_method.bf2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14: init %i32 = call %bound_method.loc14_34.2(%int_7) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc14_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc14_34.2: %i32 = converted %int_7, %.loc14_34.1 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %C.loc14: type = class_type @C, @C(constants.%int_7.0b1) [concrete = constants.%C.304]
// CHECK:STDOUT:   %.loc14_24.2: ref %C.304 = temporary_storage
// CHECK:STDOUT:   %.loc14_24.3: init %C.304 = class_init (), %.loc14_24.2 [concrete = constants.%C.val.f9f]
// CHECK:STDOUT:   %.loc14_24.4: ref %C.304 = temporary %.loc14_24.2, %.loc14_24.3
// CHECK:STDOUT:   %.loc14_26.1: ref %C.304 = converted %.loc14_24.1, %.loc14_24.4
// CHECK:STDOUT:   %impl.elem0.loc14_35: %.dad = impl_witness_access constants.%ImplicitAs.impl_witness.356, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.5ea]
// CHECK:STDOUT:   %bound_method.loc14_35: <bound method> = bound_method %.loc14_26.1, %impl.elem0.loc14_35
// CHECK:STDOUT:   %.loc6_19.8: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc14_26.2: %C.304 = acquire_value %.loc14_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc14: init %D = call %bound_method.loc14_35(%.loc14_26.2) to %.loc6_19.8
// CHECK:STDOUT:   %.loc14_35: init %D = converted %.loc14_26.1, %C.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %DestroyOp.bound.loc14_24.1: <bound method> = bound_method %.loc14_24.4, constants.%DestroyOp.b0ebf8.8
// CHECK:STDOUT:   %DestroyOp.call.loc14_24.1: init %empty_tuple.type = call %DestroyOp.bound.loc14_24.1(%.loc14_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc13_24.2: <bound method> = bound_method %.loc13_24.4, constants.%DestroyOp.b0ebf8.7
// CHECK:STDOUT:   %DestroyOp.call.loc13_24.2: init %empty_tuple.type = call %DestroyOp.bound.loc13_24.2(%.loc13_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc12_24.3: <bound method> = bound_method %.loc12_24.4, constants.%DestroyOp.b0ebf8.6
// CHECK:STDOUT:   %DestroyOp.call.loc12_24.3: init %empty_tuple.type = call %DestroyOp.bound.loc12_24.3(%.loc12_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc11_24.4: <bound method> = bound_method %.loc11_24.4, constants.%DestroyOp.b0ebf8.5
// CHECK:STDOUT:   %DestroyOp.call.loc11_24.4: init %empty_tuple.type = call %DestroyOp.bound.loc11_24.4(%.loc11_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_24.5: <bound method> = bound_method %.loc10_24.4, constants.%DestroyOp.b0ebf8.4
// CHECK:STDOUT:   %DestroyOp.call.loc10_24.5: init %empty_tuple.type = call %DestroyOp.bound.loc10_24.5(%.loc10_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc9_24.6: <bound method> = bound_method %.loc9_24.4, constants.%DestroyOp.b0ebf8.3
// CHECK:STDOUT:   %DestroyOp.call.loc9_24.6: init %empty_tuple.type = call %DestroyOp.bound.loc9_24.6(%.loc9_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc8_24.7: <bound method> = bound_method %.loc8_24.4, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc8_24.7: init %empty_tuple.type = call %DestroyOp.bound.loc8_24.7(%.loc8_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc7_24.8: <bound method> = bound_method %.loc7_24.4, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc7_24.8: init %empty_tuple.type = call %DestroyOp.bound.loc7_24.8(%.loc7_24.4)
// CHECK:STDOUT:   return %.loc14_35 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc14:
// CHECK:STDOUT:   %P.ref.loc15: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%P.Make [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc6_19.9: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc6_19.9
// CHECK:STDOUT:   %DestroyOp.bound.loc14_24.2: <bound method> = bound_method %.loc14_24.4, constants.%DestroyOp.b0ebf8.8
// CHECK:STDOUT:   %DestroyOp.call.loc14_24.2: init %empty_tuple.type = call %DestroyOp.bound.loc14_24.2(%.loc14_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc13_24.3: <bound method> = bound_method %.loc13_24.4, constants.%DestroyOp.b0ebf8.7
// CHECK:STDOUT:   %DestroyOp.call.loc13_24.3: init %empty_tuple.type = call %DestroyOp.bound.loc13_24.3(%.loc13_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc12_24.4: <bound method> = bound_method %.loc12_24.4, constants.%DestroyOp.b0ebf8.6
// CHECK:STDOUT:   %DestroyOp.call.loc12_24.4: init %empty_tuple.type = call %DestroyOp.bound.loc12_24.4(%.loc12_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc11_24.5: <bound method> = bound_method %.loc11_24.4, constants.%DestroyOp.b0ebf8.5
// CHECK:STDOUT:   %DestroyOp.call.loc11_24.5: init %empty_tuple.type = call %DestroyOp.bound.loc11_24.5(%.loc11_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc10_24.6: <bound method> = bound_method %.loc10_24.4, constants.%DestroyOp.b0ebf8.4
// CHECK:STDOUT:   %DestroyOp.call.loc10_24.6: init %empty_tuple.type = call %DestroyOp.bound.loc10_24.6(%.loc10_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc9_24.7: <bound method> = bound_method %.loc9_24.4, constants.%DestroyOp.b0ebf8.3
// CHECK:STDOUT:   %DestroyOp.call.loc9_24.7: init %empty_tuple.type = call %DestroyOp.bound.loc9_24.7(%.loc9_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc8_24.8: <bound method> = bound_method %.loc8_24.4, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call.loc8_24.8: init %empty_tuple.type = call %DestroyOp.bound.loc8_24.8(%.loc8_24.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc7_24.9: <bound method> = bound_method %.loc7_24.4, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:   %DestroyOp.call.loc7_24.9: init %empty_tuple.type = call %DestroyOp.bound.loc7_24.9(%.loc7_24.4)
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.1 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc7(%self.param: %C.b00) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.2 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc8(%self.param: %C.674) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.3 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc9(%self.param: %C.681) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.4 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10(%self.param: %C.7ac) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.5 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc11(%self.param: %C.89d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.6 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc12(%self.param: %C.f0a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.7 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc13(%self.param: %C.c60) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.8 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc14(%self.param: %C.304) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.5de) {
// CHECK:STDOUT:   %N => constants.%N.5de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.6a9) {
// CHECK:STDOUT:   %N => constants.%int_0.6a9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.5d2) {
// CHECK:STDOUT:   %N => constants.%int_1.5d2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.ef8) {
// CHECK:STDOUT:   %N => constants.%int_2.ef8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.822) {
// CHECK:STDOUT:   %N => constants.%int_3.822
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.940) {
// CHECK:STDOUT:   %N => constants.%int_4.940
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.0f6) {
// CHECK:STDOUT:   %N => constants.%int_5.0f6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.e56) {
// CHECK:STDOUT:   %N => constants.%int_6.e56
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.0b1) {
// CHECK:STDOUT:   %N => constants.%int_7.0b1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
