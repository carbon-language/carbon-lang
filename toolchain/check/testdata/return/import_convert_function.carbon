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
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.5de: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.a67: type = class_type @C, @C(%N.5de) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.de8: type = struct_type {.n: %i32, .m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ea0: <witness> = complete_type_witness %struct_type.n.m.de8 [concrete]
// CHECK:STDOUT:   %.b76: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %pattern_type.c49: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.819: type = struct_type {.n: Core.IntLiteral, .m: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.n.m.819 = struct_value (%int_0.5c6, %int_0.5c6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b94: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.b94) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet.b94 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.d2e: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_0.6a9, %int_0.6a9) [concrete]
// CHECK:STDOUT:   %C.4f5: type = class_type @C, @C(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.1a9: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.78c: <witness> = impl_witness @C.as.ImplicitAs.impl.c43.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.2de: type = pattern_type %C.4f5 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.f45: type = fn_type @C.as.ImplicitAs.impl.Convert.loc75 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.72e: %C.as.ImplicitAs.impl.Convert.type.f45 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.95b: type = class_type @C, @C(%int_1.5d2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.cf2: <witness> = impl_witness @C.as.ImplicitAs.impl.7c3.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.4f8: type = pattern_type %C.95b [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.30c: type = fn_type @C.as.ImplicitAs.impl.Convert.loc108 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.330: %C.as.ImplicitAs.impl.Convert.type.30c = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.a93: type = class_type @C, @C(%int_2.ef8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.7b9: <witness> = impl_witness @C.as.ImplicitAs.impl.fa3.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.97c: type = pattern_type %C.a93 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.9c9: type = fn_type @C.as.ImplicitAs.impl.Convert.loc141 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.4d5: %C.as.ImplicitAs.impl.Convert.type.9c9 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.1d7: type = class_type @C, @C(%int_3.822) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.377: <witness> = impl_witness @C.as.ImplicitAs.impl.158.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.00b: type = pattern_type %C.1d7 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.a26: type = fn_type @C.as.ImplicitAs.impl.Convert.loc174 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.85c: %C.as.ImplicitAs.impl.Convert.type.a26 = struct_value () [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.6d7: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %C.e40: type = class_type @C, @C(%int_4.940) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a25: <witness> = impl_witness @C.as.ImplicitAs.impl.f64.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.9d7: type = pattern_type %C.e40 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.6b2: type = fn_type @C.as.ImplicitAs.impl.Convert.loc207 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.97b: %C.as.ImplicitAs.impl.Convert.type.6b2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.e9d: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %C.3f1: type = class_type @C, @C(%int_5.0f6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fff: <witness> = impl_witness @C.as.ImplicitAs.impl.d86.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.ed0: type = pattern_type %C.3f1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.6ff: type = fn_type @C.as.ImplicitAs.impl.Convert.loc240 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.d99: %C.as.ImplicitAs.impl.Convert.type.6ff = struct_value () [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.502: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.bc2: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.e56: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %C.0fa: type = class_type @C, @C(%int_6.e56) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.178: <witness> = impl_witness @C.as.ImplicitAs.impl.f13.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.6f3: type = pattern_type %C.0fa [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.969: type = fn_type @C.as.ImplicitAs.impl.Convert.loc273 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.b4c: %C.as.ImplicitAs.impl.Convert.type.969 = struct_value () [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1e0: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.bf2: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.0b1: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %C.2b7: type = class_type @C, @C(%int_7.0b1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ad8: <witness> = impl_witness @C.as.ImplicitAs.impl.83b.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.aa6: type = pattern_type %C.2b7 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.bb0: type = fn_type @C.as.ImplicitAs.impl.Convert.loc306 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
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
// CHECK:STDOUT:     %.loc11: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc9.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc9.1 (constants.%N.5de)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc34: Core.Form = init_form %D.ref [concrete = constants.%.b76]
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.c43 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc54.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc54.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc54.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc54.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc54.2: %i32 = converted %int_0, %.loc54.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_0.6a9) [concrete = constants.%C.4f5]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.1a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.7c3 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc87.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc87.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc87.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc87.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc87.2: %i32 = converted %int_1, %.loc87.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_1.5d2) [concrete = constants.%C.95b]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.1a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.fa3 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc120.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc120.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.646]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc120.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc120.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc120.2: %i32 = converted %int_2, %.loc120.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_2.ef8) [concrete = constants.%C.a93]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.1a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.158 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc153.1: <bound method> = bound_method %int_3, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc153.2: <bound method> = bound_method %int_3, %specific_fn [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc153.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc153.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc153.2: %i32 = converted %int_3, %.loc153.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_3.822) [concrete = constants.%C.1d7]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.1a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.f64 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc186.1: <bound method> = bound_method %int_4, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc186.2: <bound method> = bound_method %int_4, %specific_fn [concrete = constants.%bound_method.6d7]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc186.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %.loc186.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %.loc186.2: %i32 = converted %int_4, %.loc186.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_4.940) [concrete = constants.%C.e40]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.1a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.d86 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:     %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc219.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc219.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method.e9d]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc219.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %.loc219.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %.loc219.2: %i32 = converted %int_5, %.loc219.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_5.0f6) [concrete = constants.%C.3f1]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.1a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.f13 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:     %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc252.1: <bound method> = bound_method %int_6, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.502]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc252.2: <bound method> = bound_method %int_6, %specific_fn [concrete = constants.%bound_method.bc2]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc252.2(%int_6) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:     %.loc252.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_6.e56]
// CHECK:STDOUT:     %.loc252.2: %i32 = converted %int_6, %.loc252.1 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_6.e56) [concrete = constants.%C.0fa]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.1a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.83b [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:     %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:     %bound_method.loc285.1: <bound method> = bound_method %int_7, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1e0]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc285.2: <bound method> = bound_method %int_7, %specific_fn [concrete = constants.%bound_method.bf2]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc285.2(%int_7) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:     %.loc285.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:     %.loc285.2: %i32 = converted %int_7, %.loc285.1 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_7.0b1) [concrete = constants.%C.2b7]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.1a9]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.c43: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.f45 = fn_decl @C.as.ImplicitAs.impl.Convert.loc75 [concrete = constants.%C.as.ImplicitAs.impl.Convert.72e] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2de = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.2de = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc74.2: Core.Form = init_form %D.ref [concrete = constants.%.b76]
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
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.30c = fn_decl @C.as.ImplicitAs.impl.Convert.loc108 [concrete = constants.%C.as.ImplicitAs.impl.Convert.330] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.4f8 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.4f8 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc107.2: Core.Form = init_form %D.ref [concrete = constants.%.b76]
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
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.9c9 = fn_decl @C.as.ImplicitAs.impl.Convert.loc141 [concrete = constants.%C.as.ImplicitAs.impl.Convert.4d5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.97c = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.97c = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc140.2: Core.Form = init_form %D.ref [concrete = constants.%.b76]
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
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.a26 = fn_decl @C.as.ImplicitAs.impl.Convert.loc174 [concrete = constants.%C.as.ImplicitAs.impl.Convert.85c] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.00b = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.00b = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc173.2: Core.Form = init_form %D.ref [concrete = constants.%.b76]
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
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.6b2 = fn_decl @C.as.ImplicitAs.impl.Convert.loc207 [concrete = constants.%C.as.ImplicitAs.impl.Convert.97b] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d7 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d7 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc206.2: Core.Form = init_form %D.ref [concrete = constants.%.b76]
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
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.6ff = fn_decl @C.as.ImplicitAs.impl.Convert.loc240 [concrete = constants.%C.as.ImplicitAs.impl.Convert.d99] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ed0 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.ed0 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc239.2: Core.Form = init_form %D.ref [concrete = constants.%.b76]
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
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.969 = fn_decl @C.as.ImplicitAs.impl.Convert.loc273 [concrete = constants.%C.as.ImplicitAs.impl.Convert.b4c] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6f3 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.6f3 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc272.2: Core.Form = init_form %D.ref [concrete = constants.%.b76]
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
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.bb0 = fn_decl @C.as.ImplicitAs.impl.Convert.loc306 [concrete = constants.%C.as.ImplicitAs.impl.Convert.aa7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.aa6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.aa6 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c49 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c49 = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc305.2: Core.Form = init_form %D.ref [concrete = constants.%.b76]
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
// CHECK:STDOUT: generic class @C(%N.loc9.2: %i32) {
// CHECK:STDOUT:   %N.loc9.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc9.1 (constants.%N.5de)]
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
// CHECK:STDOUT:   %i32.loc21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc20: %D.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %i32.loc26: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc25: %D.elem = field_decl m, element1 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n.m.de8 [concrete = constants.%complete_type.ea0]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .n = %.loc20
// CHECK:STDOUT:   .m = %.loc25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc41: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_0.loc46: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc47.1: %struct_type.n.m.819 = struct_literal (%int_0.loc41, %int_0.loc46) [concrete = constants.%struct]
// CHECK:STDOUT:   %impl.elem0.loc47.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc47.1: <bound method> = bound_method %int_0.loc41, %impl.elem0.loc47.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc47.1: <specific function> = specific_function %impl.elem0.loc47.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc47.2: <bound method> = bound_method %int_0.loc41, %specific_fn.loc47.1 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc47.1: init %i32 = call %bound_method.loc47.2(%int_0.loc41) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc47.2: init %i32 = converted %int_0.loc41, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc47.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc47.3: ref %i32 = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc47.4: init %i32 to %.loc47.3 = in_place_init %.loc47.2 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %impl.elem0.loc47.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc47.3: <bound method> = bound_method %int_0.loc46, %impl.elem0.loc47.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc47.2: <specific function> = specific_function %impl.elem0.loc47.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc47.4: <bound method> = bound_method %int_0.loc46, %specific_fn.loc47.2 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc47.2: init %i32 = call %bound_method.loc47.4(%int_0.loc46) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc47.5: init %i32 = converted %int_0.loc46, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc47.2 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc47.6: ref %i32 = class_element_access %return.param, element1
// CHECK:STDOUT:   %.loc47.7: init %i32 to %.loc47.6 = in_place_init %.loc47.5 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc47.8: init %D to %return.param = class_init (%.loc47.4, %.loc47.7) [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc48: init %D = converted %.loc47.1, %.loc47.8 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc48 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc75(%self.param: %C.4f5) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc74.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc74.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc108(%self.param: %C.95b) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc107.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc107.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc141(%self.param: %C.a93) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc140.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc140.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc174(%self.param: %C.1d7) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc173.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc173.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc207(%self.param: %C.e40) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc206.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc206.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc240(%self.param: %C.3f1) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc239.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc239.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc273(%self.param: %C.0fa) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc272.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc272.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc306(%self.param: %C.2b7) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc305.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc305.1 = call %Make.ref()
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.5de) {
// CHECK:STDOUT:   %N.loc9.1 => constants.%N.5de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_0.6a9) {
// CHECK:STDOUT:   %N.loc9.1 => constants.%int_0.6a9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_1.5d2) {
// CHECK:STDOUT:   %N.loc9.1 => constants.%int_1.5d2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_2.ef8) {
// CHECK:STDOUT:   %N.loc9.1 => constants.%int_2.ef8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_3.822) {
// CHECK:STDOUT:   %N.loc9.1 => constants.%int_3.822
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_4.940) {
// CHECK:STDOUT:   %N.loc9.1 => constants.%int_4.940
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_5.0f6) {
// CHECK:STDOUT:   %N.loc9.1 => constants.%int_5.0f6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_6.e56) {
// CHECK:STDOUT:   %N.loc9.1 => constants.%int_6.e56
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%int_7.0b1) {
// CHECK:STDOUT:   %N.loc9.1 => constants.%int_7.0b1
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
// CHECK:STDOUT:   %.852: Core.Form = init_form %D [concrete]
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
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b94: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.b94) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet.b94 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.d2e: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %C.b00: type = class_type @C, @C(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %C.val.452: %C.b00 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.899: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a31: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.774) [concrete]
// CHECK:STDOUT:   %.866: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a31, %ImplicitAs.facet.774 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.994: type = fn_type @C.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.78d: %C.as.ImplicitAs.impl.Convert.type.994 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.cfc: <bound method> = bound_method %C.val.452, %C.as.ImplicitAs.impl.Convert.78d [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.loc27 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.678: %C.674 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.638: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c5c [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.0f6: %ImplicitAs.type.899 = facet_value %C.674, (%ImplicitAs.impl_witness.638) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.cbc: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.0f6) [concrete]
// CHECK:STDOUT:   %.86b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.cbc, %ImplicitAs.facet.0f6 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.900: type = fn_type @C.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.b44: %C.as.ImplicitAs.impl.Convert.type.900 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.530: <bound method> = bound_method %C.val.678, %C.as.ImplicitAs.impl.Convert.b44 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc44 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.fb5: %C.681 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a8d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5db [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.53b: %ImplicitAs.type.899 = facet_value %C.681, (%ImplicitAs.impl_witness.a8d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.308: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.53b) [concrete]
// CHECK:STDOUT:   %.bfa: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.308, %ImplicitAs.facet.53b [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae1: type = fn_type @C.as.ImplicitAs.impl.Convert.3 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.dd6: %C.as.ImplicitAs.impl.Convert.type.ae1 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.fc6: <bound method> = bound_method %C.val.fb5, %C.as.ImplicitAs.impl.Convert.dd6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc61 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.fe7: %C.7ac = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.56b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.126 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fe0: %ImplicitAs.type.899 = facet_value %C.7ac, (%ImplicitAs.impl_witness.56b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.4d0: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.fe0) [concrete]
// CHECK:STDOUT:   %.a2c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.4d0, %ImplicitAs.facet.fe0 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.cbe: type = fn_type @C.as.ImplicitAs.impl.Convert.4 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.d1c: %C.as.ImplicitAs.impl.Convert.type.cbe = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.ffe: <bound method> = bound_method %C.val.fe7, %C.as.ImplicitAs.impl.Convert.d1c [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.4: type = fn_type @Destroy.Op.loc78 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.4: %Destroy.Op.type.bae255.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.6d7: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.1dd: %C.89d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e05: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.92c [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fd8: %ImplicitAs.type.899 = facet_value %C.89d, (%ImplicitAs.impl_witness.e05) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.46a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.fd8) [concrete]
// CHECK:STDOUT:   %.c32: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.46a, %ImplicitAs.facet.fd8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.876: type = fn_type @C.as.ImplicitAs.impl.Convert.5 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.db5: %C.as.ImplicitAs.impl.Convert.type.876 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.7cc: <bound method> = bound_method %C.val.1dd, %C.as.ImplicitAs.impl.Convert.db5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.5: type = fn_type @Destroy.Op.loc95 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.5: %Destroy.Op.type.bae255.5 = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.e9d: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.a4a: %C.f0a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.262: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d8f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.357: %ImplicitAs.type.899 = facet_value %C.f0a, (%ImplicitAs.impl_witness.262) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.3d2: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.357) [concrete]
// CHECK:STDOUT:   %.df3: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.3d2, %ImplicitAs.facet.357 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.043: type = fn_type @C.as.ImplicitAs.impl.Convert.6 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.a63: %C.as.ImplicitAs.impl.Convert.type.043 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.4ca: <bound method> = bound_method %C.val.a4a, %C.as.ImplicitAs.impl.Convert.a63 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.6: type = fn_type @Destroy.Op.loc112 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.6: %Destroy.Op.type.bae255.6 = struct_value () [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.502: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.bc2: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.a4b: %C.c60 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.7d1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.70a [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.510: %ImplicitAs.type.899 = facet_value %C.c60, (%ImplicitAs.impl_witness.7d1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a01: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.510) [concrete]
// CHECK:STDOUT:   %.4a2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a01, %ImplicitAs.facet.510 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.7be: type = fn_type @C.as.ImplicitAs.impl.Convert.7 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.e2f: %C.as.ImplicitAs.impl.Convert.type.7be = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.f82: <bound method> = bound_method %C.val.a4b, %C.as.ImplicitAs.impl.Convert.e2f [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.7: type = fn_type @Destroy.Op.loc129 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.7: %Destroy.Op.type.bae255.7 = struct_value () [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1e0: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.bf2: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.f9f: %C.304 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.356: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.575 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.965: %ImplicitAs.type.899 = facet_value %C.304, (%ImplicitAs.impl_witness.356) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.063: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%D, %ImplicitAs.facet.965) [concrete]
// CHECK:STDOUT:   %.ad8: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.063, %ImplicitAs.facet.965 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.c30: type = fn_type @C.as.ImplicitAs.impl.Convert.8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.5ea: %C.as.ImplicitAs.impl.Convert.type.c30 = struct_value () [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.bound.622: <bound method> = bound_method %C.val.f9f, %C.as.ImplicitAs.impl.Convert.5ea [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.8: type = fn_type @Destroy.Op.loc146 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.8: %Destroy.Op.type.bae255.8 = struct_value () [concrete]
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
// CHECK:STDOUT:   %P.import_ref.a7d: <witness> = import_ref P//library, loc28, loaded [concrete = constants.%complete_type.ea0]
// CHECK:STDOUT:   %P.import_ref.a4d = import_ref P//library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.f32 = import_ref P//library, loc20, unloaded
// CHECK:STDOUT:   %P.import_ref.eba = import_ref P//library, loc25, unloaded
// CHECK:STDOUT:   %P.C: %C.type = import_ref P//library, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %P.import_ref.8f2: <witness> = import_ref P//library, loc14, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %P.import_ref.01d = import_ref P//library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.67b: %i32 = import_ref P//library, loc9, loaded [symbolic = @C.%N (constants.%N.5de)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %P.import_ref.baa: <witness> = import_ref P//library, loc62, loaded [concrete = constants.%ImplicitAs.impl_witness.01b]
// CHECK:STDOUT:   %P.import_ref.a2e: type = import_ref P//library, loc54, loaded [concrete = constants.%C.b00]
// CHECK:STDOUT:   %P.import_ref.68d5e2.1: type = import_ref P//library, loc61, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.367: <witness> = import_ref P//library, loc95, loaded [concrete = constants.%ImplicitAs.impl_witness.638]
// CHECK:STDOUT:   %P.import_ref.203: type = import_ref P//library, loc87, loaded [concrete = constants.%C.674]
// CHECK:STDOUT:   %P.import_ref.68d5e2.2: type = import_ref P//library, loc94, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.9fc: <witness> = import_ref P//library, loc128, loaded [concrete = constants.%ImplicitAs.impl_witness.a8d]
// CHECK:STDOUT:   %P.import_ref.956: type = import_ref P//library, loc120, loaded [concrete = constants.%C.681]
// CHECK:STDOUT:   %P.import_ref.68d5e2.3: type = import_ref P//library, loc127, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.b266: <witness> = import_ref P//library, loc161, loaded [concrete = constants.%ImplicitAs.impl_witness.56b]
// CHECK:STDOUT:   %P.import_ref.51c: type = import_ref P//library, loc153, loaded [concrete = constants.%C.7ac]
// CHECK:STDOUT:   %P.import_ref.68d5e2.4: type = import_ref P//library, loc160, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.442: <witness> = import_ref P//library, loc194, loaded [concrete = constants.%ImplicitAs.impl_witness.e05]
// CHECK:STDOUT:   %P.import_ref.883: type = import_ref P//library, loc186, loaded [concrete = constants.%C.89d]
// CHECK:STDOUT:   %P.import_ref.68d5e2.5: type = import_ref P//library, loc193, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.766: <witness> = import_ref P//library, loc227, loaded [concrete = constants.%ImplicitAs.impl_witness.262]
// CHECK:STDOUT:   %P.import_ref.0c8: type = import_ref P//library, loc219, loaded [concrete = constants.%C.f0a]
// CHECK:STDOUT:   %P.import_ref.68d5e2.6: type = import_ref P//library, loc226, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.63e: <witness> = import_ref P//library, loc260, loaded [concrete = constants.%ImplicitAs.impl_witness.7d1]
// CHECK:STDOUT:   %P.import_ref.92b: type = import_ref P//library, loc252, loaded [concrete = constants.%C.c60]
// CHECK:STDOUT:   %P.import_ref.68d5e2.7: type = import_ref P//library, loc259, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.24c: <witness> = import_ref P//library, loc293, loaded [concrete = constants.%ImplicitAs.impl_witness.356]
// CHECK:STDOUT:   %P.import_ref.854: type = import_ref P//library, loc285, loaded [concrete = constants.%C.304]
// CHECK:STDOUT:   %P.import_ref.68d5e2.8: type = import_ref P//library, loc292, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.623: %C.as.ImplicitAs.impl.Convert.type.994 = import_ref P//library, loc75, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.78d]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1e8 = impl_witness_table (%P.import_ref.623), @C.as.ImplicitAs.impl.87c [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %P.import_ref.a4a: %C.as.ImplicitAs.impl.Convert.type.900 = import_ref P//library, loc108, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.b44]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c5c = impl_witness_table (%P.import_ref.a4a), @C.as.ImplicitAs.impl.a01 [concrete]
// CHECK:STDOUT:   %P.import_ref.e4a: %C.as.ImplicitAs.impl.Convert.type.ae1 = import_ref P//library, loc141, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.dd6]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5db = impl_witness_table (%P.import_ref.e4a), @C.as.ImplicitAs.impl.390 [concrete]
// CHECK:STDOUT:   %P.import_ref.b26b: %C.as.ImplicitAs.impl.Convert.type.cbe = import_ref P//library, loc174, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.d1c]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.126 = impl_witness_table (%P.import_ref.b26b), @C.as.ImplicitAs.impl.071 [concrete]
// CHECK:STDOUT:   %P.import_ref.6da: %C.as.ImplicitAs.impl.Convert.type.876 = import_ref P//library, loc207, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.db5]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.92c = impl_witness_table (%P.import_ref.6da), @C.as.ImplicitAs.impl.891 [concrete]
// CHECK:STDOUT:   %P.import_ref.200: %C.as.ImplicitAs.impl.Convert.type.043 = import_ref P//library, loc240, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.a63]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d8f = impl_witness_table (%P.import_ref.200), @C.as.ImplicitAs.impl.882 [concrete]
// CHECK:STDOUT:   %P.import_ref.010: %C.as.ImplicitAs.impl.Convert.type.7be = import_ref P//library, loc273, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.e2f]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.70a = impl_witness_table (%P.import_ref.010), @C.as.ImplicitAs.impl.ad9 [concrete]
// CHECK:STDOUT:   %P.import_ref.624: %C.as.ImplicitAs.impl.Convert.type.c30 = import_ref P//library, loc306, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.5ea]
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
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.f8a = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.f8a = return_slot_pattern %return.param_patt, %D.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %P.ref.loc15: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:     %D.ref: type = name_ref D, imports.%P.D [concrete = constants.%D]
// CHECK:STDOUT:     %.loc16.10: Core.Form = init_form %D.ref [concrete = constants.%.852]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
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
// CHECK:STDOUT: fn @F0(%n.param: %i32) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %false.loc21: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc21 br !if.then.loc22 else br !if.else.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc22:
// CHECK:STDOUT:   %.loc26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc28: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc29: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc33: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc33.1: <bound method> = bound_method %int_0, %impl.elem0.loc33 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.897]
// CHECK:STDOUT:   %specific_fn.loc33: <specific function> = specific_function %impl.elem0.loc33, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33.2: <bound method> = bound_method %int_0, %specific_fn.loc33 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc33: init %i32 = call %bound_method.loc33.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc33.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc33 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc33.2: %i32 = converted %int_0, %.loc33.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %C.loc33: type = class_type @C, @C(constants.%int_0.6a9) [concrete = constants.%C.b00]
// CHECK:STDOUT:   %.loc26.2: ref %C.b00 = temporary_storage
// CHECK:STDOUT:   %.loc26.3: init %C.b00 to %.loc26.2 = class_init () [concrete = constants.%C.val.452]
// CHECK:STDOUT:   %.loc27.1: init %C.b00 = converted %.loc26.1, %.loc26.3 [concrete = constants.%C.val.452]
// CHECK:STDOUT:   %impl.elem0.loc34: %.866 = impl_witness_access constants.%ImplicitAs.impl_witness.01b, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.78d]
// CHECK:STDOUT:   %bound_method.loc34: <bound method> = bound_method %.loc27.1, %impl.elem0.loc34 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.cfc]
// CHECK:STDOUT:   %.loc16.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc27.2: ref %C.b00 = temporary %.loc26.2, %.loc27.1
// CHECK:STDOUT:   %.loc27.3: %C.b00 = acquire_value %.loc27.2
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc34: init %D to %.loc16.1 = call %bound_method.loc34(%.loc27.3)
// CHECK:STDOUT:   %.loc34: init %D = converted %.loc27.1, %C.as.ImplicitAs.impl.Convert.call.loc34
// CHECK:STDOUT:   %Destroy.Op.bound.loc27.1: <bound method> = bound_method %.loc27.2, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc27.1: init %empty_tuple.type = call %Destroy.Op.bound.loc27.1(%.loc27.2)
// CHECK:STDOUT:   return %.loc34 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc22:
// CHECK:STDOUT:   %false.loc38: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc38 br !if.then.loc39 else br !if.else.loc39
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc39:
// CHECK:STDOUT:   %.loc43.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc45: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc46: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc50: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc50.1: <bound method> = bound_method %int_1, %impl.elem0.loc50 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc50: <specific function> = specific_function %impl.elem0.loc50, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc50.2: <bound method> = bound_method %int_1, %specific_fn.loc50 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc50: init %i32 = call %bound_method.loc50.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc50.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc50 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc50.2: %i32 = converted %int_1, %.loc50.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %C.loc50: type = class_type @C, @C(constants.%int_1.5d2) [concrete = constants.%C.674]
// CHECK:STDOUT:   %.loc43.2: ref %C.674 = temporary_storage
// CHECK:STDOUT:   %.loc43.3: init %C.674 to %.loc43.2 = class_init () [concrete = constants.%C.val.678]
// CHECK:STDOUT:   %.loc44.1: init %C.674 = converted %.loc43.1, %.loc43.3 [concrete = constants.%C.val.678]
// CHECK:STDOUT:   %impl.elem0.loc51: %.86b = impl_witness_access constants.%ImplicitAs.impl_witness.638, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.b44]
// CHECK:STDOUT:   %bound_method.loc51: <bound method> = bound_method %.loc44.1, %impl.elem0.loc51 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.530]
// CHECK:STDOUT:   %.loc16.2: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc44.2: ref %C.674 = temporary %.loc43.2, %.loc44.1
// CHECK:STDOUT:   %.loc44.3: %C.674 = acquire_value %.loc44.2
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc51: init %D to %.loc16.2 = call %bound_method.loc51(%.loc44.3)
// CHECK:STDOUT:   %.loc51: init %D = converted %.loc44.1, %C.as.ImplicitAs.impl.Convert.call.loc51
// CHECK:STDOUT:   %Destroy.Op.bound.loc44.1: <bound method> = bound_method %.loc44.2, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc44.1: init %empty_tuple.type = call %Destroy.Op.bound.loc44.1(%.loc44.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc27.2: <bound method> = bound_method %.loc27.2, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc27.2: init %empty_tuple.type = call %Destroy.Op.bound.loc27.2(%.loc27.2)
// CHECK:STDOUT:   return %.loc51 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc39:
// CHECK:STDOUT:   %false.loc55: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc55 br !if.then.loc56 else br !if.else.loc56
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc56:
// CHECK:STDOUT:   %.loc60.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc62: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc63: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc67: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc67.1: <bound method> = bound_method %int_2, %impl.elem0.loc67 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc67: <specific function> = specific_function %impl.elem0.loc67, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc67.2: <bound method> = bound_method %int_2, %specific_fn.loc67 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc67: init %i32 = call %bound_method.loc67.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc67.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc67 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc67.2: %i32 = converted %int_2, %.loc67.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %C.loc67: type = class_type @C, @C(constants.%int_2.ef8) [concrete = constants.%C.681]
// CHECK:STDOUT:   %.loc60.2: ref %C.681 = temporary_storage
// CHECK:STDOUT:   %.loc60.3: init %C.681 to %.loc60.2 = class_init () [concrete = constants.%C.val.fb5]
// CHECK:STDOUT:   %.loc61.1: init %C.681 = converted %.loc60.1, %.loc60.3 [concrete = constants.%C.val.fb5]
// CHECK:STDOUT:   %impl.elem0.loc68: %.bfa = impl_witness_access constants.%ImplicitAs.impl_witness.a8d, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.dd6]
// CHECK:STDOUT:   %bound_method.loc68: <bound method> = bound_method %.loc61.1, %impl.elem0.loc68 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.fc6]
// CHECK:STDOUT:   %.loc16.3: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc61.2: ref %C.681 = temporary %.loc60.2, %.loc61.1
// CHECK:STDOUT:   %.loc61.3: %C.681 = acquire_value %.loc61.2
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc68: init %D to %.loc16.3 = call %bound_method.loc68(%.loc61.3)
// CHECK:STDOUT:   %.loc68: init %D = converted %.loc61.1, %C.as.ImplicitAs.impl.Convert.call.loc68
// CHECK:STDOUT:   %Destroy.Op.bound.loc61.1: <bound method> = bound_method %.loc61.2, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc61.1: init %empty_tuple.type = call %Destroy.Op.bound.loc61.1(%.loc61.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc44.2: <bound method> = bound_method %.loc44.2, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc44.2: init %empty_tuple.type = call %Destroy.Op.bound.loc44.2(%.loc44.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc27.3: <bound method> = bound_method %.loc27.2, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc27.3: init %empty_tuple.type = call %Destroy.Op.bound.loc27.3(%.loc27.2)
// CHECK:STDOUT:   return %.loc68 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc56:
// CHECK:STDOUT:   %false.loc72: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc72 br !if.then.loc73 else br !if.else.loc73
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc73:
// CHECK:STDOUT:   %.loc77.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc79: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc80: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc84: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc84.1: <bound method> = bound_method %int_3, %impl.elem0.loc84 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc84: <specific function> = specific_function %impl.elem0.loc84, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc84.2: <bound method> = bound_method %int_3, %specific_fn.loc84 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc84: init %i32 = call %bound_method.loc84.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc84.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc84 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc84.2: %i32 = converted %int_3, %.loc84.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %C.loc84: type = class_type @C, @C(constants.%int_3.822) [concrete = constants.%C.7ac]
// CHECK:STDOUT:   %.loc77.2: ref %C.7ac = temporary_storage
// CHECK:STDOUT:   %.loc77.3: init %C.7ac to %.loc77.2 = class_init () [concrete = constants.%C.val.fe7]
// CHECK:STDOUT:   %.loc78.1: init %C.7ac = converted %.loc77.1, %.loc77.3 [concrete = constants.%C.val.fe7]
// CHECK:STDOUT:   %impl.elem0.loc85: %.a2c = impl_witness_access constants.%ImplicitAs.impl_witness.56b, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.d1c]
// CHECK:STDOUT:   %bound_method.loc85: <bound method> = bound_method %.loc78.1, %impl.elem0.loc85 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.ffe]
// CHECK:STDOUT:   %.loc16.4: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc78.2: ref %C.7ac = temporary %.loc77.2, %.loc78.1
// CHECK:STDOUT:   %.loc78.3: %C.7ac = acquire_value %.loc78.2
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc85: init %D to %.loc16.4 = call %bound_method.loc85(%.loc78.3)
// CHECK:STDOUT:   %.loc85: init %D = converted %.loc78.1, %C.as.ImplicitAs.impl.Convert.call.loc85
// CHECK:STDOUT:   %Destroy.Op.bound.loc78.1: <bound method> = bound_method %.loc78.2, constants.%Destroy.Op.651ba6.4
// CHECK:STDOUT:   %Destroy.Op.call.loc78.1: init %empty_tuple.type = call %Destroy.Op.bound.loc78.1(%.loc78.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc61.2: <bound method> = bound_method %.loc61.2, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc61.2: init %empty_tuple.type = call %Destroy.Op.bound.loc61.2(%.loc61.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc44.3: <bound method> = bound_method %.loc44.2, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc44.3: init %empty_tuple.type = call %Destroy.Op.bound.loc44.3(%.loc44.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc27.4: <bound method> = bound_method %.loc27.2, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc27.4: init %empty_tuple.type = call %Destroy.Op.bound.loc27.4(%.loc27.2)
// CHECK:STDOUT:   return %.loc85 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc73:
// CHECK:STDOUT:   %false.loc89: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc89 br !if.then.loc90 else br !if.else.loc90
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc90:
// CHECK:STDOUT:   %.loc94.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc96: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc97: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc101: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc101.1: <bound method> = bound_method %int_4, %impl.elem0.loc101 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c]
// CHECK:STDOUT:   %specific_fn.loc101: <specific function> = specific_function %impl.elem0.loc101, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc101.2: <bound method> = bound_method %int_4, %specific_fn.loc101 [concrete = constants.%bound_method.6d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc101: init %i32 = call %bound_method.loc101.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc101.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc101 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc101.2: %i32 = converted %int_4, %.loc101.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %C.loc101: type = class_type @C, @C(constants.%int_4.940) [concrete = constants.%C.89d]
// CHECK:STDOUT:   %.loc94.2: ref %C.89d = temporary_storage
// CHECK:STDOUT:   %.loc94.3: init %C.89d to %.loc94.2 = class_init () [concrete = constants.%C.val.1dd]
// CHECK:STDOUT:   %.loc95.1: init %C.89d = converted %.loc94.1, %.loc94.3 [concrete = constants.%C.val.1dd]
// CHECK:STDOUT:   %impl.elem0.loc102: %.c32 = impl_witness_access constants.%ImplicitAs.impl_witness.e05, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.db5]
// CHECK:STDOUT:   %bound_method.loc102: <bound method> = bound_method %.loc95.1, %impl.elem0.loc102 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.7cc]
// CHECK:STDOUT:   %.loc16.5: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc95.2: ref %C.89d = temporary %.loc94.2, %.loc95.1
// CHECK:STDOUT:   %.loc95.3: %C.89d = acquire_value %.loc95.2
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc102: init %D to %.loc16.5 = call %bound_method.loc102(%.loc95.3)
// CHECK:STDOUT:   %.loc102: init %D = converted %.loc95.1, %C.as.ImplicitAs.impl.Convert.call.loc102
// CHECK:STDOUT:   %Destroy.Op.bound.loc95.1: <bound method> = bound_method %.loc95.2, constants.%Destroy.Op.651ba6.5
// CHECK:STDOUT:   %Destroy.Op.call.loc95.1: init %empty_tuple.type = call %Destroy.Op.bound.loc95.1(%.loc95.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc78.2: <bound method> = bound_method %.loc78.2, constants.%Destroy.Op.651ba6.4
// CHECK:STDOUT:   %Destroy.Op.call.loc78.2: init %empty_tuple.type = call %Destroy.Op.bound.loc78.2(%.loc78.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc61.3: <bound method> = bound_method %.loc61.2, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc61.3: init %empty_tuple.type = call %Destroy.Op.bound.loc61.3(%.loc61.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc44.4: <bound method> = bound_method %.loc44.2, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc44.4: init %empty_tuple.type = call %Destroy.Op.bound.loc44.4(%.loc44.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc27.5: <bound method> = bound_method %.loc27.2, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc27.5: init %empty_tuple.type = call %Destroy.Op.bound.loc27.5(%.loc27.2)
// CHECK:STDOUT:   return %.loc102 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc90:
// CHECK:STDOUT:   %false.loc106: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc106 br !if.then.loc107 else br !if.else.loc107
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc107:
// CHECK:STDOUT:   %.loc111.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc113: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc114: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc118: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc118.1: <bound method> = bound_method %int_5, %impl.elem0.loc118 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005]
// CHECK:STDOUT:   %specific_fn.loc118: <specific function> = specific_function %impl.elem0.loc118, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc118.2: <bound method> = bound_method %int_5, %specific_fn.loc118 [concrete = constants.%bound_method.e9d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc118: init %i32 = call %bound_method.loc118.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc118.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc118 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc118.2: %i32 = converted %int_5, %.loc118.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %C.loc118: type = class_type @C, @C(constants.%int_5.0f6) [concrete = constants.%C.f0a]
// CHECK:STDOUT:   %.loc111.2: ref %C.f0a = temporary_storage
// CHECK:STDOUT:   %.loc111.3: init %C.f0a to %.loc111.2 = class_init () [concrete = constants.%C.val.a4a]
// CHECK:STDOUT:   %.loc112.1: init %C.f0a = converted %.loc111.1, %.loc111.3 [concrete = constants.%C.val.a4a]
// CHECK:STDOUT:   %impl.elem0.loc119: %.df3 = impl_witness_access constants.%ImplicitAs.impl_witness.262, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.a63]
// CHECK:STDOUT:   %bound_method.loc119: <bound method> = bound_method %.loc112.1, %impl.elem0.loc119 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.4ca]
// CHECK:STDOUT:   %.loc16.6: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc112.2: ref %C.f0a = temporary %.loc111.2, %.loc112.1
// CHECK:STDOUT:   %.loc112.3: %C.f0a = acquire_value %.loc112.2
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc119: init %D to %.loc16.6 = call %bound_method.loc119(%.loc112.3)
// CHECK:STDOUT:   %.loc119: init %D = converted %.loc112.1, %C.as.ImplicitAs.impl.Convert.call.loc119
// CHECK:STDOUT:   %Destroy.Op.bound.loc112.1: <bound method> = bound_method %.loc112.2, constants.%Destroy.Op.651ba6.6
// CHECK:STDOUT:   %Destroy.Op.call.loc112.1: init %empty_tuple.type = call %Destroy.Op.bound.loc112.1(%.loc112.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc95.2: <bound method> = bound_method %.loc95.2, constants.%Destroy.Op.651ba6.5
// CHECK:STDOUT:   %Destroy.Op.call.loc95.2: init %empty_tuple.type = call %Destroy.Op.bound.loc95.2(%.loc95.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc78.3: <bound method> = bound_method %.loc78.2, constants.%Destroy.Op.651ba6.4
// CHECK:STDOUT:   %Destroy.Op.call.loc78.3: init %empty_tuple.type = call %Destroy.Op.bound.loc78.3(%.loc78.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc61.4: <bound method> = bound_method %.loc61.2, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc61.4: init %empty_tuple.type = call %Destroy.Op.bound.loc61.4(%.loc61.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc44.5: <bound method> = bound_method %.loc44.2, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc44.5: init %empty_tuple.type = call %Destroy.Op.bound.loc44.5(%.loc44.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc27.6: <bound method> = bound_method %.loc27.2, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc27.6: init %empty_tuple.type = call %Destroy.Op.bound.loc27.6(%.loc27.2)
// CHECK:STDOUT:   return %.loc119 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc107:
// CHECK:STDOUT:   %false.loc123: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc123 br !if.then.loc124 else br !if.else.loc124
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc124:
// CHECK:STDOUT:   %.loc128.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc130: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc131: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:   %impl.elem0.loc135: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc135.1: <bound method> = bound_method %int_6, %impl.elem0.loc135 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.502]
// CHECK:STDOUT:   %specific_fn.loc135: <specific function> = specific_function %impl.elem0.loc135, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc135.2: <bound method> = bound_method %int_6, %specific_fn.loc135 [concrete = constants.%bound_method.bc2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc135: init %i32 = call %bound_method.loc135.2(%int_6) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc135.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc135 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc135.2: %i32 = converted %int_6, %.loc135.1 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %C.loc135: type = class_type @C, @C(constants.%int_6.e56) [concrete = constants.%C.c60]
// CHECK:STDOUT:   %.loc128.2: ref %C.c60 = temporary_storage
// CHECK:STDOUT:   %.loc128.3: init %C.c60 to %.loc128.2 = class_init () [concrete = constants.%C.val.a4b]
// CHECK:STDOUT:   %.loc129.1: init %C.c60 = converted %.loc128.1, %.loc128.3 [concrete = constants.%C.val.a4b]
// CHECK:STDOUT:   %impl.elem0.loc136: %.4a2 = impl_witness_access constants.%ImplicitAs.impl_witness.7d1, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.e2f]
// CHECK:STDOUT:   %bound_method.loc136: <bound method> = bound_method %.loc129.1, %impl.elem0.loc136 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.f82]
// CHECK:STDOUT:   %.loc16.7: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc129.2: ref %C.c60 = temporary %.loc128.2, %.loc129.1
// CHECK:STDOUT:   %.loc129.3: %C.c60 = acquire_value %.loc129.2
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc136: init %D to %.loc16.7 = call %bound_method.loc136(%.loc129.3)
// CHECK:STDOUT:   %.loc136: init %D = converted %.loc129.1, %C.as.ImplicitAs.impl.Convert.call.loc136
// CHECK:STDOUT:   %Destroy.Op.bound.loc129.1: <bound method> = bound_method %.loc129.2, constants.%Destroy.Op.651ba6.7
// CHECK:STDOUT:   %Destroy.Op.call.loc129.1: init %empty_tuple.type = call %Destroy.Op.bound.loc129.1(%.loc129.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc112.2: <bound method> = bound_method %.loc112.2, constants.%Destroy.Op.651ba6.6
// CHECK:STDOUT:   %Destroy.Op.call.loc112.2: init %empty_tuple.type = call %Destroy.Op.bound.loc112.2(%.loc112.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc95.3: <bound method> = bound_method %.loc95.2, constants.%Destroy.Op.651ba6.5
// CHECK:STDOUT:   %Destroy.Op.call.loc95.3: init %empty_tuple.type = call %Destroy.Op.bound.loc95.3(%.loc95.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc78.4: <bound method> = bound_method %.loc78.2, constants.%Destroy.Op.651ba6.4
// CHECK:STDOUT:   %Destroy.Op.call.loc78.4: init %empty_tuple.type = call %Destroy.Op.bound.loc78.4(%.loc78.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc61.5: <bound method> = bound_method %.loc61.2, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc61.5: init %empty_tuple.type = call %Destroy.Op.bound.loc61.5(%.loc61.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc44.6: <bound method> = bound_method %.loc44.2, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc44.6: init %empty_tuple.type = call %Destroy.Op.bound.loc44.6(%.loc44.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc27.7: <bound method> = bound_method %.loc27.2, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc27.7: init %empty_tuple.type = call %Destroy.Op.bound.loc27.7(%.loc27.2)
// CHECK:STDOUT:   return %.loc136 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc124:
// CHECK:STDOUT:   %false.loc140: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc140 br !if.then.loc141 else br !if.else.loc141
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc141:
// CHECK:STDOUT:   %.loc145.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %P.ref.loc147: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc148: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:   %impl.elem0.loc152: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc152.1: <bound method> = bound_method %int_7, %impl.elem0.loc152 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1e0]
// CHECK:STDOUT:   %specific_fn.loc152: <specific function> = specific_function %impl.elem0.loc152, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc152.2: <bound method> = bound_method %int_7, %specific_fn.loc152 [concrete = constants.%bound_method.bf2]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc152: init %i32 = call %bound_method.loc152.2(%int_7) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc152.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc152 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc152.2: %i32 = converted %int_7, %.loc152.1 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %C.loc152: type = class_type @C, @C(constants.%int_7.0b1) [concrete = constants.%C.304]
// CHECK:STDOUT:   %.loc145.2: ref %C.304 = temporary_storage
// CHECK:STDOUT:   %.loc145.3: init %C.304 to %.loc145.2 = class_init () [concrete = constants.%C.val.f9f]
// CHECK:STDOUT:   %.loc146.1: init %C.304 = converted %.loc145.1, %.loc145.3 [concrete = constants.%C.val.f9f]
// CHECK:STDOUT:   %impl.elem0.loc153: %.ad8 = impl_witness_access constants.%ImplicitAs.impl_witness.356, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.5ea]
// CHECK:STDOUT:   %bound_method.loc153: <bound method> = bound_method %.loc146.1, %impl.elem0.loc153 [concrete = constants.%C.as.ImplicitAs.impl.Convert.bound.622]
// CHECK:STDOUT:   %.loc16.8: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %.loc146.2: ref %C.304 = temporary %.loc145.2, %.loc146.1
// CHECK:STDOUT:   %.loc146.3: %C.304 = acquire_value %.loc146.2
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc153: init %D to %.loc16.8 = call %bound_method.loc153(%.loc146.3)
// CHECK:STDOUT:   %.loc153: init %D = converted %.loc146.1, %C.as.ImplicitAs.impl.Convert.call.loc153
// CHECK:STDOUT:   %Destroy.Op.bound.loc146.1: <bound method> = bound_method %.loc146.2, constants.%Destroy.Op.651ba6.8
// CHECK:STDOUT:   %Destroy.Op.call.loc146.1: init %empty_tuple.type = call %Destroy.Op.bound.loc146.1(%.loc146.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc129.2: <bound method> = bound_method %.loc129.2, constants.%Destroy.Op.651ba6.7
// CHECK:STDOUT:   %Destroy.Op.call.loc129.2: init %empty_tuple.type = call %Destroy.Op.bound.loc129.2(%.loc129.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc112.3: <bound method> = bound_method %.loc112.2, constants.%Destroy.Op.651ba6.6
// CHECK:STDOUT:   %Destroy.Op.call.loc112.3: init %empty_tuple.type = call %Destroy.Op.bound.loc112.3(%.loc112.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc95.4: <bound method> = bound_method %.loc95.2, constants.%Destroy.Op.651ba6.5
// CHECK:STDOUT:   %Destroy.Op.call.loc95.4: init %empty_tuple.type = call %Destroy.Op.bound.loc95.4(%.loc95.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc78.5: <bound method> = bound_method %.loc78.2, constants.%Destroy.Op.651ba6.4
// CHECK:STDOUT:   %Destroy.Op.call.loc78.5: init %empty_tuple.type = call %Destroy.Op.bound.loc78.5(%.loc78.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc61.6: <bound method> = bound_method %.loc61.2, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc61.6: init %empty_tuple.type = call %Destroy.Op.bound.loc61.6(%.loc61.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc44.7: <bound method> = bound_method %.loc44.2, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc44.7: init %empty_tuple.type = call %Destroy.Op.bound.loc44.7(%.loc44.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc27.8: <bound method> = bound_method %.loc27.2, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc27.8: init %empty_tuple.type = call %Destroy.Op.bound.loc27.8(%.loc27.2)
// CHECK:STDOUT:   return %.loc153 to %return.param
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc141:
// CHECK:STDOUT:   %P.ref.loc156: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%P.Make [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc16.9: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %Make.call: init %D to %.loc16.9 = call %Make.ref()
// CHECK:STDOUT:   %Destroy.Op.bound.loc146.2: <bound method> = bound_method %.loc146.2, constants.%Destroy.Op.651ba6.8
// CHECK:STDOUT:   %Destroy.Op.call.loc146.2: init %empty_tuple.type = call %Destroy.Op.bound.loc146.2(%.loc146.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc129.3: <bound method> = bound_method %.loc129.2, constants.%Destroy.Op.651ba6.7
// CHECK:STDOUT:   %Destroy.Op.call.loc129.3: init %empty_tuple.type = call %Destroy.Op.bound.loc129.3(%.loc129.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc112.4: <bound method> = bound_method %.loc112.2, constants.%Destroy.Op.651ba6.6
// CHECK:STDOUT:   %Destroy.Op.call.loc112.4: init %empty_tuple.type = call %Destroy.Op.bound.loc112.4(%.loc112.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc95.5: <bound method> = bound_method %.loc95.2, constants.%Destroy.Op.651ba6.5
// CHECK:STDOUT:   %Destroy.Op.call.loc95.5: init %empty_tuple.type = call %Destroy.Op.bound.loc95.5(%.loc95.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc78.6: <bound method> = bound_method %.loc78.2, constants.%Destroy.Op.651ba6.4
// CHECK:STDOUT:   %Destroy.Op.call.loc78.6: init %empty_tuple.type = call %Destroy.Op.bound.loc78.6(%.loc78.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc61.7: <bound method> = bound_method %.loc61.2, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc61.7: init %empty_tuple.type = call %Destroy.Op.bound.loc61.7(%.loc61.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc44.8: <bound method> = bound_method %.loc44.2, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc44.8: init %empty_tuple.type = call %Destroy.Op.bound.loc44.8(%.loc44.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc27.9: <bound method> = bound_method %.loc27.2, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc27.9: init %empty_tuple.type = call %Destroy.Op.bound.loc27.9(%.loc27.2)
// CHECK:STDOUT:   return %Make.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.1 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc27(%self.param: ref %C.b00) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.2 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44(%self.param: ref %C.674) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.3 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc61(%self.param: ref %C.681) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.4 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc78(%self.param: ref %C.7ac) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.5 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc95(%self.param: ref %C.89d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.6 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc112(%self.param: ref %C.f0a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.7 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc129(%self.param: ref %C.c60) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.8 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc146(%self.param: ref %C.304) = "no_op";
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
