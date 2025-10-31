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
// CHECK:STDOUT:   %N.fe8: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.49b: type = class_type @C, @C(%N.fe8) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.de8: type = struct_type {.n: %i32, .m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ea0: <witness> = complete_type_witness %struct_type.n.m.de8 [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.819: type = struct_type {.n: Core.IntLiteral, .m: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3ad: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.3ad [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b3: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9be: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_0.6a9, %int_0.6a9) [concrete]
// CHECK:STDOUT:   %C.c8d: type = class_type @C, @C(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.26d: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.b7a: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc8 [concrete]
// CHECK:STDOUT:   %pattern_type.f33: type = pattern_type %C.c8d [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.478: type = fn_type @C.as.ImplicitAs.impl.Convert.loc8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.97f: %C.as.ImplicitAs.impl.Convert.type.478 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.265: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.dfc: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.c02: type = class_type @C, @C(%int_1.5d2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.358: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc9 [concrete]
// CHECK:STDOUT:   %pattern_type.07c: type = pattern_type %C.c02 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.9e5: type = fn_type @C.as.ImplicitAs.impl.Convert.loc9 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.dc1: %C.as.ImplicitAs.impl.Convert.type.9e5 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.393: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.9f7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.70b: type = class_type @C, @C(%int_2.ef8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.25e: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc10 [concrete]
// CHECK:STDOUT:   %pattern_type.538: type = pattern_type %C.70b [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.0da: type = fn_type @C.as.ImplicitAs.impl.Convert.loc10 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.fda: %C.as.ImplicitAs.impl.Convert.type.0da = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87d: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.def: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.8cb: type = class_type @C, @C(%int_3.822) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.83b: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc11 [concrete]
// CHECK:STDOUT:   %pattern_type.232: type = pattern_type %C.8cb [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.7f2: type = fn_type @C.as.ImplicitAs.impl.Convert.loc11 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.803: %C.as.ImplicitAs.impl.Convert.type.7f2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b5: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.5dd: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %C.ba2: type = class_type @C, @C(%int_4.940) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc0: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc12 [concrete]
// CHECK:STDOUT:   %pattern_type.fe0: type = pattern_type %C.ba2 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.d29: type = fn_type @C.as.ImplicitAs.impl.Convert.loc12 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.c16: %C.as.ImplicitAs.impl.Convert.type.d29 = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.451: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.c57: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %C.fcd: type = class_type @C, @C(%int_5.0f6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.b5c: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc13 [concrete]
// CHECK:STDOUT:   %pattern_type.e96: type = pattern_type %C.fcd [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.bba: type = fn_type @C.as.ImplicitAs.impl.Convert.loc13 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.b0d: %C.as.ImplicitAs.impl.Convert.type.bba = struct_value () [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a4a: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.b96: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.e56: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %C.675: type = class_type @C, @C(%int_6.e56) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5ab: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc14 [concrete]
// CHECK:STDOUT:   %pattern_type.bc0: type = pattern_type %C.675 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.84f: type = fn_type @C.as.ImplicitAs.impl.Convert.loc14 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.a90: %C.as.ImplicitAs.impl.Convert.type.84f = struct_value () [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a66: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.1db: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.0b1: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %C.884: type = class_type @C, @C(%int_7.0b1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.19e: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc15 [concrete]
// CHECK:STDOUT:   %pattern_type.c15: type = pattern_type %C.884 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.66d: type = fn_type @C.as.ImplicitAs.impl.Convert.loc15 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.be0: %C.as.ImplicitAs.impl.Convert.type.66d = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %N.loc4_9.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_9.1 (constants.%N.fe8)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param0
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.890 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:     %bound_method.loc8_9.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b3]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_9.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.9be]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_9.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc8_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc8_9.2: %i32 = converted %int_0, %.loc8_9.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_0.6a9) [concrete = constants.%C.c8d]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.26d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc8 = impl_witness_table (@C.as.ImplicitAs.impl.890.%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.890 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc8: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc8 [concrete = constants.%ImplicitAs.impl_witness.b7a]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.612 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:     %bound_method.loc9_9.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.265]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_9.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method.dfc]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_9.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc9_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc9_9.2: %i32 = converted %int_1, %.loc9_9.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_1.5d2) [concrete = constants.%C.c02]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.26d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc9 = impl_witness_table (@C.as.ImplicitAs.impl.612.%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.612 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc9: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc9 [concrete = constants.%ImplicitAs.impl_witness.358]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.39d [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:     %bound_method.loc10_9.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.393]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_9.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.9f7]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc10_9.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc10_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc10_9.2: %i32 = converted %int_2, %.loc10_9.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_2.ef8) [concrete = constants.%C.70b]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.26d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc10 = impl_witness_table (@C.as.ImplicitAs.impl.39d.%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.39d [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc10: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc10 [concrete = constants.%ImplicitAs.impl_witness.25e]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.9d2 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:     %bound_method.loc11_9.1: <bound method> = bound_method %int_3, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87d]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_9.2: <bound method> = bound_method %int_3, %specific_fn [concrete = constants.%bound_method.def]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_9.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc11_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc11_9.2: %i32 = converted %int_3, %.loc11_9.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_3.822) [concrete = constants.%C.8cb]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.26d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc11 = impl_witness_table (@C.as.ImplicitAs.impl.9d2.%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.9d2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc11: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc11 [concrete = constants.%ImplicitAs.impl_witness.83b]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.993 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:     %bound_method.loc12_9.1: <bound method> = bound_method %int_4, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b5]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc12_9.2: <bound method> = bound_method %int_4, %specific_fn [concrete = constants.%bound_method.5dd]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_9.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %.loc12_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %.loc12_9.2: %i32 = converted %int_4, %.loc12_9.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_4.940) [concrete = constants.%C.ba2]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.26d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc12 = impl_witness_table (@C.as.ImplicitAs.impl.993.%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.993 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc12: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc12 [concrete = constants.%ImplicitAs.impl_witness.bc0]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.77e [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:     %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:     %bound_method.loc13_9.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.451]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc13_9.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method.c57]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_9.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %.loc13_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %.loc13_9.2: %i32 = converted %int_5, %.loc13_9.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_5.0f6) [concrete = constants.%C.fcd]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.26d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc13 = impl_witness_table (@C.as.ImplicitAs.impl.77e.%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.77e [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc13: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc13 [concrete = constants.%ImplicitAs.impl_witness.b5c]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.1b5 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:     %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %int_6, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a4a]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %int_6, %specific_fn [concrete = constants.%bound_method.b96]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc14_9.2(%int_6) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:     %.loc14_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_6.e56]
// CHECK:STDOUT:     %.loc14_9.2: %i32 = converted %int_6, %.loc14_9.1 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_6.e56) [concrete = constants.%C.675]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.26d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc14 = impl_witness_table (@C.as.ImplicitAs.impl.1b5.%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.1b5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc14: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc14 [concrete = constants.%ImplicitAs.impl_witness.5ab]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.b0d [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:     %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:     %bound_method.loc15_9.1: <bound method> = bound_method %int_7, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a66]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc15_9.2: <bound method> = bound_method %int_7, %specific_fn [concrete = constants.%bound_method.1db]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_9.2(%int_7) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:     %.loc15_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:     %.loc15_9.2: %i32 = converted %int_7, %.loc15_9.1 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_7.0b1) [concrete = constants.%C.884]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.26d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc15 = impl_witness_table (@C.as.ImplicitAs.impl.b0d.%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.b0d [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc15: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc15 [concrete = constants.%ImplicitAs.impl_witness.19e]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.890: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.478 = fn_decl @C.as.ImplicitAs.impl.Convert.loc8 [concrete = constants.%C.as.ImplicitAs.impl.Convert.97f] {
// CHECK:STDOUT:     %self.patt: %pattern_type.f33 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f33 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.c8d = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.890.%C [concrete = constants.%C.c8d]
// CHECK:STDOUT:     %self: %C.c8d = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.612: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.9e5 = fn_decl @C.as.ImplicitAs.impl.Convert.loc9 [concrete = constants.%C.as.ImplicitAs.impl.Convert.dc1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.07c = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.07c = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.c02 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.612.%C [concrete = constants.%C.c02]
// CHECK:STDOUT:     %self: %C.c02 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.39d: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.0da = fn_decl @C.as.ImplicitAs.impl.Convert.loc10 [concrete = constants.%C.as.ImplicitAs.impl.Convert.fda] {
// CHECK:STDOUT:     %self.patt: %pattern_type.538 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.538 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.70b = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.39d.%C [concrete = constants.%C.70b]
// CHECK:STDOUT:     %self: %C.70b = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.9d2: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.7f2 = fn_decl @C.as.ImplicitAs.impl.Convert.loc11 [concrete = constants.%C.as.ImplicitAs.impl.Convert.803] {
// CHECK:STDOUT:     %self.patt: %pattern_type.232 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.232 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.8cb = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.9d2.%C [concrete = constants.%C.8cb]
// CHECK:STDOUT:     %self: %C.8cb = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.993: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.d29 = fn_decl @C.as.ImplicitAs.impl.Convert.loc12 [concrete = constants.%C.as.ImplicitAs.impl.Convert.c16] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fe0 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fe0 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.ba2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.993.%C [concrete = constants.%C.ba2]
// CHECK:STDOUT:     %self: %C.ba2 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.77e: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.bba = fn_decl @C.as.ImplicitAs.impl.Convert.loc13 [concrete = constants.%C.as.ImplicitAs.impl.Convert.b0d] {
// CHECK:STDOUT:     %self.patt: %pattern_type.e96 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e96 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.fcd = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.77e.%C [concrete = constants.%C.fcd]
// CHECK:STDOUT:     %self: %C.fcd = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.1b5: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.84f = fn_decl @C.as.ImplicitAs.impl.Convert.loc14 [concrete = constants.%C.as.ImplicitAs.impl.Convert.a90] {
// CHECK:STDOUT:     %self.patt: %pattern_type.bc0 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.bc0 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.675 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.1b5.%C [concrete = constants.%C.675]
// CHECK:STDOUT:     %self: %C.675 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.b0d: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.66d = fn_decl @C.as.ImplicitAs.impl.Convert.loc15 [concrete = constants.%C.as.ImplicitAs.impl.Convert.be0] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c15 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c15 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.884 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.b0d.%C [concrete = constants.%C.884]
// CHECK:STDOUT:     %self: %C.884 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %C.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%N.loc4_9.2: %i32) {
// CHECK:STDOUT:   %N.loc4_9.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_9.1 (constants.%N.fe8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.49b
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
// CHECK:STDOUT:   %.loc6_40.1: %struct_type.n.m.819 = struct_literal (%int_0.loc6_31, %int_0.loc6_39)
// CHECK:STDOUT:   %impl.elem0.loc6_40.1: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc6_40.1: <bound method> = bound_method %int_0.loc6_31, %impl.elem0.loc6_40.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b3]
// CHECK:STDOUT:   %specific_fn.loc6_40.1: <specific function> = specific_function %impl.elem0.loc6_40.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.2: <bound method> = bound_method %int_0.loc6_31, %specific_fn.loc6_40.1 [concrete = constants.%bound_method.9be]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.1: init %i32 = call %bound_method.loc6_40.2(%int_0.loc6_31) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.2: init %i32 = converted %int_0.loc6_31, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.3: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc6_40.4: init %i32 = initialize_from %.loc6_40.2 to %.loc6_40.3 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %impl.elem0.loc6_40.2: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc6_40.3: <bound method> = bound_method %int_0.loc6_39, %impl.elem0.loc6_40.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b3]
// CHECK:STDOUT:   %specific_fn.loc6_40.2: <specific function> = specific_function %impl.elem0.loc6_40.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.4: <bound method> = bound_method %int_0.loc6_39, %specific_fn.loc6_40.2 [concrete = constants.%bound_method.9be]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.2: init %i32 = call %bound_method.loc6_40.4(%int_0.loc6_39) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.5: init %i32 = converted %int_0.loc6_39, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.2 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.6: ref %i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc6_40.7: init %i32 = initialize_from %.loc6_40.5 to %.loc6_40.6 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.8: init %D = class_init (%.loc6_40.4, %.loc6_40.7), %return [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc6_41: init %D = converted %.loc6_40.1, %.loc6_40.8 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc6_41 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc8(%self.param: %C.c8d) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc8: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc8
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc9(%self.param: %C.c02) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc9: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc9
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc10(%self.param: %C.70b) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc10: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc10
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc11(%self.param: %C.8cb) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc11: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc11
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc12(%self.param: %C.ba2) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc12: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc12
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc13(%self.param: %C.fcd) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc13
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc14(%self.param: %C.675) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc14
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.loc15(%self.param: %C.884) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc15: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc15
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.fe8) {
// CHECK:STDOUT:   %N.loc4_9.1 => constants.%N.fe8
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
// CHECK:STDOUT:   %pattern_type.f8a: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %F0.type: type = fn_type @F0 [concrete]
// CHECK:STDOUT:   %F0: %F0.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %N.fe8: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3ad: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.3ad [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b3: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9be: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %C.b00: type = class_type @C, @C(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %C.val.452: %C.b00 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.bc5: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.541: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.2b6 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3e5: %ImplicitAs.type.bc5 = facet_value %C.b00, (%ImplicitAs.impl_witness.541) [concrete]
// CHECK:STDOUT:   %.e33: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.3e5 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.7ddb8: type = fn_type @C.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.43b: %C.as.ImplicitAs.impl.Convert.type.7ddb8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.e56: %type_where = facet_value %C.b00, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.42c: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.e56) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.9ed: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.42c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.697: type = ptr_type %C.b00 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.67c: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.9ed, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.e56) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.265: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.dfc: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.678: %C.674 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.371: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ff5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.47a: %ImplicitAs.type.bc5 = facet_value %C.674, (%ImplicitAs.impl_witness.371) [concrete]
// CHECK:STDOUT:   %.5da: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.47a [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.f4e: type = fn_type @C.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.e68: %C.as.ImplicitAs.impl.Convert.type.f4e = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.4dd: %type_where = facet_value %C.674, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.6b5: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.4dd) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.d98: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.6b5 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.625: type = ptr_type %C.674 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.466: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.d98, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.4dd) [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.393: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.9f7: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.fb5: %C.681 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2a3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a5f: %ImplicitAs.type.bc5 = facet_value %C.681, (%ImplicitAs.impl_witness.2a3) [concrete]
// CHECK:STDOUT:   %.b2b: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.a5f [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.7cc: type = fn_type @C.as.ImplicitAs.impl.Convert.3 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.cb1: %C.as.ImplicitAs.impl.Convert.type.7cc = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.7fe: %type_where = facet_value %C.681, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.6b4: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.7fe) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.ce4: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.6b4 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.3bd: type = ptr_type %C.681 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cfd: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.ce4, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.7fe) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87d: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.def: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.fe7: %C.7ac = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.29c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.056 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c0d: %ImplicitAs.type.bc5 = facet_value %C.7ac, (%ImplicitAs.impl_witness.29c) [concrete]
// CHECK:STDOUT:   %.11b: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.c0d [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.cba: type = fn_type @C.as.ImplicitAs.impl.Convert.4 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.67e: %C.as.ImplicitAs.impl.Convert.type.cba = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.99b: %type_where = facet_value %C.7ac, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.22a: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.99b) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.931: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.22a = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.2b1: type = ptr_type %C.7ac [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.20a: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.931, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.99b) [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b5: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.5dd: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.1dd: %C.89d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.565: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5ff [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.5f1: %ImplicitAs.type.bc5 = facet_value %C.89d, (%ImplicitAs.impl_witness.565) [concrete]
// CHECK:STDOUT:   %.0dc: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.5f1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.a2c: type = fn_type @C.as.ImplicitAs.impl.Convert.5 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.58d: %C.as.ImplicitAs.impl.Convert.type.a2c = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.b4c: %type_where = facet_value %C.89d, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.dc5: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.b4c) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.d47: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.dc5 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28b: type = ptr_type %C.89d [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.fe8: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.d47, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.b4c) [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.451: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.c57: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.a4a: %C.f0a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c1a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c8d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.489: %ImplicitAs.type.bc5 = facet_value %C.f0a, (%ImplicitAs.impl_witness.c1a) [concrete]
// CHECK:STDOUT:   %.332: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.489 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.bf6: type = fn_type @C.as.ImplicitAs.impl.Convert.6 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.a45: %C.as.ImplicitAs.impl.Convert.type.bf6 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.430: %type_where = facet_value %C.f0a, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.3a3: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.430) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.217: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.3a3 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.24c: type = ptr_type %C.f0a [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.bcb: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.217, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.430) [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a4a: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.b96: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.a4b: %C.c60 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ebf: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a9c [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d57: %ImplicitAs.type.bc5 = facet_value %C.c60, (%ImplicitAs.impl_witness.ebf) [concrete]
// CHECK:STDOUT:   %.1bb: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.d57 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.e95: type = fn_type @C.as.ImplicitAs.impl.Convert.7 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.ad0: %C.as.ImplicitAs.impl.Convert.type.e95 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.14a: %type_where = facet_value %C.c60, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.748: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.14a) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bf7: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.748 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b5e: type = ptr_type %C.c60 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.5e9: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.bf7, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.14a) [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a66: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %bound_method.1db: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.f9f: %C.304 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.b93: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.820 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.faa: %ImplicitAs.type.bc5 = facet_value %C.304, (%ImplicitAs.impl_witness.b93) [concrete]
// CHECK:STDOUT:   %.bc1: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.faa [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.7ddb0: type = fn_type @C.as.ImplicitAs.impl.Convert.8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.9be: %C.as.ImplicitAs.impl.Convert.type.7ddb0 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.488: %type_where = facet_value %C.304, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b0c: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.488) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.85d: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b0c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.dc3: type = ptr_type %C.304 [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.eb6: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.85d, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.488) [concrete]
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
// CHECK:STDOUT:   %P.import_ref.cab = import_ref P//library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.a525 = import_ref P//library, loc5_16, unloaded
// CHECK:STDOUT:   %P.import_ref.b4a = import_ref P//library, loc5_28, unloaded
// CHECK:STDOUT:   %P.C: %C.type = import_ref P//library, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %P.import_ref.797: %i32 = import_ref P//library, loc4_9, loaded [symbolic = @C.%N (constants.%N.fe8)]
// CHECK:STDOUT:   %P.import_ref.8f2: <witness> = import_ref P//library, loc4_19, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %P.import_ref.9b2 = import_ref P//library, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %P.import_ref.cc2: <witness> = import_ref P//library, loc8_33, loaded [concrete = constants.%ImplicitAs.impl_witness.541]
// CHECK:STDOUT:   %P.import_ref.d2c: type = import_ref P//library, loc8_9, loaded [concrete = constants.%C.b00]
// CHECK:STDOUT:   %P.import_ref.daf1e3.1: type = import_ref P//library, loc8_31, loaded [concrete = constants.%ImplicitAs.type.bc5]
// CHECK:STDOUT:   %P.import_ref.755: <witness> = import_ref P//library, loc9_33, loaded [concrete = constants.%ImplicitAs.impl_witness.371]
// CHECK:STDOUT:   %P.import_ref.cf3: type = import_ref P//library, loc9_9, loaded [concrete = constants.%C.674]
// CHECK:STDOUT:   %P.import_ref.daf1e3.2: type = import_ref P//library, loc9_31, loaded [concrete = constants.%ImplicitAs.type.bc5]
// CHECK:STDOUT:   %P.import_ref.2b2: <witness> = import_ref P//library, loc10_33, loaded [concrete = constants.%ImplicitAs.impl_witness.2a3]
// CHECK:STDOUT:   %P.import_ref.095: type = import_ref P//library, loc10_9, loaded [concrete = constants.%C.681]
// CHECK:STDOUT:   %P.import_ref.daf1e3.3: type = import_ref P//library, loc10_31, loaded [concrete = constants.%ImplicitAs.type.bc5]
// CHECK:STDOUT:   %P.import_ref.f08: <witness> = import_ref P//library, loc11_33, loaded [concrete = constants.%ImplicitAs.impl_witness.29c]
// CHECK:STDOUT:   %P.import_ref.82b: type = import_ref P//library, loc11_9, loaded [concrete = constants.%C.7ac]
// CHECK:STDOUT:   %P.import_ref.daf1e3.4: type = import_ref P//library, loc11_31, loaded [concrete = constants.%ImplicitAs.type.bc5]
// CHECK:STDOUT:   %P.import_ref.eb0: <witness> = import_ref P//library, loc12_33, loaded [concrete = constants.%ImplicitAs.impl_witness.565]
// CHECK:STDOUT:   %P.import_ref.f7d: type = import_ref P//library, loc12_9, loaded [concrete = constants.%C.89d]
// CHECK:STDOUT:   %P.import_ref.daf1e3.5: type = import_ref P//library, loc12_31, loaded [concrete = constants.%ImplicitAs.type.bc5]
// CHECK:STDOUT:   %P.import_ref.d8a: <witness> = import_ref P//library, loc13_33, loaded [concrete = constants.%ImplicitAs.impl_witness.c1a]
// CHECK:STDOUT:   %P.import_ref.a43: type = import_ref P//library, loc13_9, loaded [concrete = constants.%C.f0a]
// CHECK:STDOUT:   %P.import_ref.daf1e3.6: type = import_ref P//library, loc13_31, loaded [concrete = constants.%ImplicitAs.type.bc5]
// CHECK:STDOUT:   %P.import_ref.a523: <witness> = import_ref P//library, loc14_33, loaded [concrete = constants.%ImplicitAs.impl_witness.ebf]
// CHECK:STDOUT:   %P.import_ref.654: type = import_ref P//library, loc14_9, loaded [concrete = constants.%C.c60]
// CHECK:STDOUT:   %P.import_ref.daf1e3.7: type = import_ref P//library, loc14_31, loaded [concrete = constants.%ImplicitAs.type.bc5]
// CHECK:STDOUT:   %P.import_ref.b1e: <witness> = import_ref P//library, loc15_33, loaded [concrete = constants.%ImplicitAs.impl_witness.b93]
// CHECK:STDOUT:   %P.import_ref.dae: type = import_ref P//library, loc15_9, loaded [concrete = constants.%C.304]
// CHECK:STDOUT:   %P.import_ref.daf1e3.8: type = import_ref P//library, loc15_31, loaded [concrete = constants.%ImplicitAs.type.bc5]
// CHECK:STDOUT:   %P.import_ref.381: %C.as.ImplicitAs.impl.Convert.type.7ddb8 = import_ref P//library, loc8_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.43b]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.2b6 = impl_witness_table (%P.import_ref.381), @C.as.ImplicitAs.impl.c36 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %P.import_ref.aa6: %C.as.ImplicitAs.impl.Convert.type.f4e = import_ref P//library, loc9_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.e68]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ff5 = impl_witness_table (%P.import_ref.aa6), @C.as.ImplicitAs.impl.4a1 [concrete]
// CHECK:STDOUT:   %P.import_ref.c8c: %C.as.ImplicitAs.impl.Convert.type.7cc = import_ref P//library, loc10_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.cb1]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d32 = impl_witness_table (%P.import_ref.c8c), @C.as.ImplicitAs.impl.2ab [concrete]
// CHECK:STDOUT:   %P.import_ref.472: %C.as.ImplicitAs.impl.Convert.type.cba = import_ref P//library, loc11_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.67e]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.056 = impl_witness_table (%P.import_ref.472), @C.as.ImplicitAs.impl.cce [concrete]
// CHECK:STDOUT:   %P.import_ref.707: %C.as.ImplicitAs.impl.Convert.type.a2c = import_ref P//library, loc12_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.58d]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5ff = impl_witness_table (%P.import_ref.707), @C.as.ImplicitAs.impl.d94 [concrete]
// CHECK:STDOUT:   %P.import_ref.8ab: %C.as.ImplicitAs.impl.Convert.type.bf6 = import_ref P//library, loc13_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.a45]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c8d = impl_witness_table (%P.import_ref.8ab), @C.as.ImplicitAs.impl.695 [concrete]
// CHECK:STDOUT:   %P.import_ref.d8d: %C.as.ImplicitAs.impl.Convert.type.e95 = import_ref P//library, loc14_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.ad0]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a9c = impl_witness_table (%P.import_ref.d8d), @C.as.ImplicitAs.impl.7d7 [concrete]
// CHECK:STDOUT:   %P.import_ref.bc0: %C.as.ImplicitAs.impl.Convert.type.7ddb0 = import_ref P//library, loc15_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.9be]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.820 = impl_witness_table (%P.import_ref.bc0), @C.as.ImplicitAs.impl.86a [concrete]
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
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.c36: imports.%P.import_ref.d2c as imports.%P.import_ref.daf1e3.1 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.cc2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.4a1: imports.%P.import_ref.cf3 as imports.%P.import_ref.daf1e3.2 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.755
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.2ab: imports.%P.import_ref.095 as imports.%P.import_ref.daf1e3.3 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.2b2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.cce: imports.%P.import_ref.82b as imports.%P.import_ref.daf1e3.4 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.f08
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.d94: imports.%P.import_ref.f7d as imports.%P.import_ref.daf1e3.5 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.eb0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.695: imports.%P.import_ref.a43 as imports.%P.import_ref.daf1e3.6 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.d8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.7d7: imports.%P.import_ref.654 as imports.%P.import_ref.daf1e3.7 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.a523
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.86a: imports.%P.import_ref.dae as imports.%P.import_ref.daf1e3.8 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.b1e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "library.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%P.import_ref.a7d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%P.import_ref.cab
// CHECK:STDOUT:   .n = imports.%P.import_ref.a525
// CHECK:STDOUT:   .m = imports.%P.import_ref.b4a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%P.import_ref.797: %i32) [from "library.carbon"] {
// CHECK:STDOUT:   %N: %i32 = symbolic_binding N, 0 [symbolic = %N (constants.%N.fe8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%P.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%P.import_ref.9b2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F0(%n.param: %i32) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %false.loc7: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc7 br !if.then.loc7 else br !if.else.loc7
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc7:
// CHECK:STDOUT:   %.loc7_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc7: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc7: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc7_34: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc7_34.1: <bound method> = bound_method %int_0, %impl.elem0.loc7_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b3]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_34.2: <bound method> = bound_method %int_0, %specific_fn.loc7 [concrete = constants.%bound_method.9be]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_34.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc7_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc7_34.2: %i32 = converted %int_0, %.loc7_34.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %C.loc7: type = class_type @C, @C(constants.%int_0.6a9) [concrete = constants.%C.b00]
// CHECK:STDOUT:   %.loc7_24.2: ref %C.b00 = temporary_storage
// CHECK:STDOUT:   %.loc7_24.3: init %C.b00 = class_init (), %.loc7_24.2 [concrete = constants.%C.val.452]
// CHECK:STDOUT:   %.loc7_24.4: ref %C.b00 = temporary %.loc7_24.2, %.loc7_24.3
// CHECK:STDOUT:   %.loc7_26.1: ref %C.b00 = converted %.loc7_24.1, %.loc7_24.4
// CHECK:STDOUT:   %impl.elem0.loc7_35: %.e33 = impl_witness_access constants.%ImplicitAs.impl_witness.541, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.43b]
// CHECK:STDOUT:   %bound_method.loc7_35: <bound method> = bound_method %.loc7_26.1, %impl.elem0.loc7_35
// CHECK:STDOUT:   %.loc7_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc7_26.2: %C.b00 = acquire_value %.loc7_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc7: init %D = call %bound_method.loc7_35(%.loc7_26.2) to %.loc7_35.1
// CHECK:STDOUT:   %.loc7_35.2: init %D = converted %.loc7_26.1, %C.as.ImplicitAs.impl.Convert.call.loc7
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc7_24.1: <bound method> = bound_method %.loc7_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.e56) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.67c]
// CHECK:STDOUT:   %bound_method.loc7_24.1: <bound method> = bound_method %.loc7_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc7_24.1: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc7_24.1: init %empty_tuple.type = call %bound_method.loc7_24.1(%addr.loc7_24.1)
// CHECK:STDOUT:   return %.loc7_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc7:
// CHECK:STDOUT:   %false.loc8: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc8 br !if.then.loc8 else br !if.else.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc8:
// CHECK:STDOUT:   %.loc8_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc8: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc8: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8_34: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.265]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_1, %specific_fn.loc8 [concrete = constants.%bound_method.dfc]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_34.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_1, %.loc8_34.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %C.loc8: type = class_type @C, @C(constants.%int_1.5d2) [concrete = constants.%C.674]
// CHECK:STDOUT:   %.loc8_24.2: ref %C.674 = temporary_storage
// CHECK:STDOUT:   %.loc8_24.3: init %C.674 = class_init (), %.loc8_24.2 [concrete = constants.%C.val.678]
// CHECK:STDOUT:   %.loc8_24.4: ref %C.674 = temporary %.loc8_24.2, %.loc8_24.3
// CHECK:STDOUT:   %.loc8_26.1: ref %C.674 = converted %.loc8_24.1, %.loc8_24.4
// CHECK:STDOUT:   %impl.elem0.loc8_35: %.5da = impl_witness_access constants.%ImplicitAs.impl_witness.371, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.e68]
// CHECK:STDOUT:   %bound_method.loc8_35: <bound method> = bound_method %.loc8_26.1, %impl.elem0.loc8_35
// CHECK:STDOUT:   %.loc8_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc8_26.2: %C.674 = acquire_value %.loc8_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc8: init %D = call %bound_method.loc8_35(%.loc8_26.2) to %.loc8_35.1
// CHECK:STDOUT:   %.loc8_35.2: init %D = converted %.loc8_26.1, %C.as.ImplicitAs.impl.Convert.call.loc8
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8_24.1: <bound method> = bound_method %.loc8_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d98
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d98, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4dd) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.466]
// CHECK:STDOUT:   %bound_method.loc8_24.1: <bound method> = bound_method %.loc8_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc8_24.1: %ptr.625 = addr_of %.loc8_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8_24.1: init %empty_tuple.type = call %bound_method.loc8_24.1(%addr.loc8_24.1)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc7_24.2: <bound method> = bound_method %.loc7_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.e56) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.67c]
// CHECK:STDOUT:   %bound_method.loc7_24.2: <bound method> = bound_method %.loc7_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc7_24.2: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc7_24.2: init %empty_tuple.type = call %bound_method.loc7_24.2(%addr.loc7_24.2)
// CHECK:STDOUT:   return %.loc8_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc8:
// CHECK:STDOUT:   %false.loc9: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc9 br !if.then.loc9 else br !if.else.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc9:
// CHECK:STDOUT:   %.loc9_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc9: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc9: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc9_34: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %int_2, %impl.elem0.loc9_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.393]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %int_2, %specific_fn.loc9 [concrete = constants.%bound_method.9f7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_34.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_34.2: %i32 = converted %int_2, %.loc9_34.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %C.loc9: type = class_type @C, @C(constants.%int_2.ef8) [concrete = constants.%C.681]
// CHECK:STDOUT:   %.loc9_24.2: ref %C.681 = temporary_storage
// CHECK:STDOUT:   %.loc9_24.3: init %C.681 = class_init (), %.loc9_24.2 [concrete = constants.%C.val.fb5]
// CHECK:STDOUT:   %.loc9_24.4: ref %C.681 = temporary %.loc9_24.2, %.loc9_24.3
// CHECK:STDOUT:   %.loc9_26.1: ref %C.681 = converted %.loc9_24.1, %.loc9_24.4
// CHECK:STDOUT:   %impl.elem0.loc9_35: %.b2b = impl_witness_access constants.%ImplicitAs.impl_witness.2a3, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.cb1]
// CHECK:STDOUT:   %bound_method.loc9_35: <bound method> = bound_method %.loc9_26.1, %impl.elem0.loc9_35
// CHECK:STDOUT:   %.loc9_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc9_26.2: %C.681 = acquire_value %.loc9_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc9: init %D = call %bound_method.loc9_35(%.loc9_26.2) to %.loc9_35.1
// CHECK:STDOUT:   %.loc9_35.2: init %D = converted %.loc9_26.1, %C.as.ImplicitAs.impl.Convert.call.loc9
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9_24.1: <bound method> = bound_method %.loc9_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ce4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ce4, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.7fe) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cfd]
// CHECK:STDOUT:   %bound_method.loc9_24.1: <bound method> = bound_method %.loc9_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %addr.loc9_24.1: %ptr.3bd = addr_of %.loc9_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9_24.1: init %empty_tuple.type = call %bound_method.loc9_24.1(%addr.loc9_24.1)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8_24.2: <bound method> = bound_method %.loc8_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d98
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.5: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d98, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4dd) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.466]
// CHECK:STDOUT:   %bound_method.loc8_24.2: <bound method> = bound_method %.loc8_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.5
// CHECK:STDOUT:   %addr.loc8_24.2: %ptr.625 = addr_of %.loc8_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8_24.2: init %empty_tuple.type = call %bound_method.loc8_24.2(%addr.loc8_24.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc7_24.3: <bound method> = bound_method %.loc7_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.6: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.e56) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.67c]
// CHECK:STDOUT:   %bound_method.loc7_24.3: <bound method> = bound_method %.loc7_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.6
// CHECK:STDOUT:   %addr.loc7_24.3: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc7_24.3: init %empty_tuple.type = call %bound_method.loc7_24.3(%addr.loc7_24.3)
// CHECK:STDOUT:   return %.loc9_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc9:
// CHECK:STDOUT:   %false.loc10: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc10 br !if.then.loc10 else br !if.else.loc10
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc10:
// CHECK:STDOUT:   %.loc10_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc10: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc10: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc10_34: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc10_34.1: <bound method> = bound_method %int_3, %impl.elem0.loc10_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.87d]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_34.2: <bound method> = bound_method %int_3, %specific_fn.loc10 [concrete = constants.%bound_method.def]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %i32 = call %bound_method.loc10_34.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc10_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc10_34.2: %i32 = converted %int_3, %.loc10_34.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %C.loc10: type = class_type @C, @C(constants.%int_3.822) [concrete = constants.%C.7ac]
// CHECK:STDOUT:   %.loc10_24.2: ref %C.7ac = temporary_storage
// CHECK:STDOUT:   %.loc10_24.3: init %C.7ac = class_init (), %.loc10_24.2 [concrete = constants.%C.val.fe7]
// CHECK:STDOUT:   %.loc10_24.4: ref %C.7ac = temporary %.loc10_24.2, %.loc10_24.3
// CHECK:STDOUT:   %.loc10_26.1: ref %C.7ac = converted %.loc10_24.1, %.loc10_24.4
// CHECK:STDOUT:   %impl.elem0.loc10_35: %.11b = impl_witness_access constants.%ImplicitAs.impl_witness.29c, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.67e]
// CHECK:STDOUT:   %bound_method.loc10_35: <bound method> = bound_method %.loc10_26.1, %impl.elem0.loc10_35
// CHECK:STDOUT:   %.loc10_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc10_26.2: %C.7ac = acquire_value %.loc10_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc10: init %D = call %bound_method.loc10_35(%.loc10_26.2) to %.loc10_35.1
// CHECK:STDOUT:   %.loc10_35.2: init %D = converted %.loc10_26.1, %C.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_24.1: <bound method> = bound_method %.loc10_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.931
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.7: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.931, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.99b) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.20a]
// CHECK:STDOUT:   %bound_method.loc10_24.1: <bound method> = bound_method %.loc10_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.7
// CHECK:STDOUT:   %addr.loc10_24.1: %ptr.2b1 = addr_of %.loc10_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_24.1: init %empty_tuple.type = call %bound_method.loc10_24.1(%addr.loc10_24.1)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9_24.2: <bound method> = bound_method %.loc9_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ce4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.8: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ce4, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.7fe) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cfd]
// CHECK:STDOUT:   %bound_method.loc9_24.2: <bound method> = bound_method %.loc9_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.8
// CHECK:STDOUT:   %addr.loc9_24.2: %ptr.3bd = addr_of %.loc9_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9_24.2: init %empty_tuple.type = call %bound_method.loc9_24.2(%addr.loc9_24.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8_24.3: <bound method> = bound_method %.loc8_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d98
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.9: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d98, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4dd) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.466]
// CHECK:STDOUT:   %bound_method.loc8_24.3: <bound method> = bound_method %.loc8_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.9
// CHECK:STDOUT:   %addr.loc8_24.3: %ptr.625 = addr_of %.loc8_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8_24.3: init %empty_tuple.type = call %bound_method.loc8_24.3(%addr.loc8_24.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc7_24.4: <bound method> = bound_method %.loc7_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.10: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.e56) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.67c]
// CHECK:STDOUT:   %bound_method.loc7_24.4: <bound method> = bound_method %.loc7_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.10
// CHECK:STDOUT:   %addr.loc7_24.4: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc7_24.4: init %empty_tuple.type = call %bound_method.loc7_24.4(%addr.loc7_24.4)
// CHECK:STDOUT:   return %.loc10_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc10:
// CHECK:STDOUT:   %false.loc11: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc11 br !if.then.loc11 else br !if.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11:
// CHECK:STDOUT:   %.loc11_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc11: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc11: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc11_34: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc11_34.1: <bound method> = bound_method %int_4, %impl.elem0.loc11_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b5]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_34.2: <bound method> = bound_method %int_4, %specific_fn.loc11 [concrete = constants.%bound_method.5dd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %i32 = call %bound_method.loc11_34.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc11_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc11_34.2: %i32 = converted %int_4, %.loc11_34.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %C.loc11: type = class_type @C, @C(constants.%int_4.940) [concrete = constants.%C.89d]
// CHECK:STDOUT:   %.loc11_24.2: ref %C.89d = temporary_storage
// CHECK:STDOUT:   %.loc11_24.3: init %C.89d = class_init (), %.loc11_24.2 [concrete = constants.%C.val.1dd]
// CHECK:STDOUT:   %.loc11_24.4: ref %C.89d = temporary %.loc11_24.2, %.loc11_24.3
// CHECK:STDOUT:   %.loc11_26.1: ref %C.89d = converted %.loc11_24.1, %.loc11_24.4
// CHECK:STDOUT:   %impl.elem0.loc11_35: %.0dc = impl_witness_access constants.%ImplicitAs.impl_witness.565, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.58d]
// CHECK:STDOUT:   %bound_method.loc11_35: <bound method> = bound_method %.loc11_26.1, %impl.elem0.loc11_35
// CHECK:STDOUT:   %.loc11_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc11_26.2: %C.89d = acquire_value %.loc11_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc11: init %D = call %bound_method.loc11_35(%.loc11_26.2) to %.loc11_35.1
// CHECK:STDOUT:   %.loc11_35.2: init %D = converted %.loc11_26.1, %C.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc11_24.1: <bound method> = bound_method %.loc11_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d47
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.11: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d47, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.b4c) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.fe8]
// CHECK:STDOUT:   %bound_method.loc11_24.1: <bound method> = bound_method %.loc11_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.11
// CHECK:STDOUT:   %addr.loc11_24.1: %ptr.c28b = addr_of %.loc11_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc11_24.1: init %empty_tuple.type = call %bound_method.loc11_24.1(%addr.loc11_24.1)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_24.2: <bound method> = bound_method %.loc10_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.931
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.12: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.931, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.99b) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.20a]
// CHECK:STDOUT:   %bound_method.loc10_24.2: <bound method> = bound_method %.loc10_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.12
// CHECK:STDOUT:   %addr.loc10_24.2: %ptr.2b1 = addr_of %.loc10_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_24.2: init %empty_tuple.type = call %bound_method.loc10_24.2(%addr.loc10_24.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9_24.3: <bound method> = bound_method %.loc9_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ce4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.13: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ce4, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.7fe) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cfd]
// CHECK:STDOUT:   %bound_method.loc9_24.3: <bound method> = bound_method %.loc9_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.13
// CHECK:STDOUT:   %addr.loc9_24.3: %ptr.3bd = addr_of %.loc9_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9_24.3: init %empty_tuple.type = call %bound_method.loc9_24.3(%addr.loc9_24.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8_24.4: <bound method> = bound_method %.loc8_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d98
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.14: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d98, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4dd) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.466]
// CHECK:STDOUT:   %bound_method.loc8_24.4: <bound method> = bound_method %.loc8_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.14
// CHECK:STDOUT:   %addr.loc8_24.4: %ptr.625 = addr_of %.loc8_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8_24.4: init %empty_tuple.type = call %bound_method.loc8_24.4(%addr.loc8_24.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc7_24.5: <bound method> = bound_method %.loc7_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.15: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.e56) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.67c]
// CHECK:STDOUT:   %bound_method.loc7_24.5: <bound method> = bound_method %.loc7_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.15
// CHECK:STDOUT:   %addr.loc7_24.5: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc7_24.5: init %empty_tuple.type = call %bound_method.loc7_24.5(%addr.loc7_24.5)
// CHECK:STDOUT:   return %.loc11_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11:
// CHECK:STDOUT:   %false.loc12: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc12 br !if.then.loc12 else br !if.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc12:
// CHECK:STDOUT:   %.loc12_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc12: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc12: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc12_34: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc12_34.1: <bound method> = bound_method %int_5, %impl.elem0.loc12_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.451]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_34.2: <bound method> = bound_method %int_5, %specific_fn.loc12 [concrete = constants.%bound_method.c57]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %i32 = call %bound_method.loc12_34.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc12_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc12_34.2: %i32 = converted %int_5, %.loc12_34.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %C.loc12: type = class_type @C, @C(constants.%int_5.0f6) [concrete = constants.%C.f0a]
// CHECK:STDOUT:   %.loc12_24.2: ref %C.f0a = temporary_storage
// CHECK:STDOUT:   %.loc12_24.3: init %C.f0a = class_init (), %.loc12_24.2 [concrete = constants.%C.val.a4a]
// CHECK:STDOUT:   %.loc12_24.4: ref %C.f0a = temporary %.loc12_24.2, %.loc12_24.3
// CHECK:STDOUT:   %.loc12_26.1: ref %C.f0a = converted %.loc12_24.1, %.loc12_24.4
// CHECK:STDOUT:   %impl.elem0.loc12_35: %.332 = impl_witness_access constants.%ImplicitAs.impl_witness.c1a, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.a45]
// CHECK:STDOUT:   %bound_method.loc12_35: <bound method> = bound_method %.loc12_26.1, %impl.elem0.loc12_35
// CHECK:STDOUT:   %.loc12_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc12_26.2: %C.f0a = acquire_value %.loc12_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc12: init %D = call %bound_method.loc12_35(%.loc12_26.2) to %.loc12_35.1
// CHECK:STDOUT:   %.loc12_35.2: init %D = converted %.loc12_26.1, %C.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc12_24.1: <bound method> = bound_method %.loc12_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.217
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.16: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.217, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.430) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.bcb]
// CHECK:STDOUT:   %bound_method.loc12_24.1: <bound method> = bound_method %.loc12_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.16
// CHECK:STDOUT:   %addr.loc12_24.1: %ptr.24c = addr_of %.loc12_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc12_24.1: init %empty_tuple.type = call %bound_method.loc12_24.1(%addr.loc12_24.1)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc11_24.2: <bound method> = bound_method %.loc11_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d47
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.17: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d47, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.b4c) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.fe8]
// CHECK:STDOUT:   %bound_method.loc11_24.2: <bound method> = bound_method %.loc11_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.17
// CHECK:STDOUT:   %addr.loc11_24.2: %ptr.c28b = addr_of %.loc11_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc11_24.2: init %empty_tuple.type = call %bound_method.loc11_24.2(%addr.loc11_24.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_24.3: <bound method> = bound_method %.loc10_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.931
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.18: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.931, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.99b) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.20a]
// CHECK:STDOUT:   %bound_method.loc10_24.3: <bound method> = bound_method %.loc10_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.18
// CHECK:STDOUT:   %addr.loc10_24.3: %ptr.2b1 = addr_of %.loc10_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_24.3: init %empty_tuple.type = call %bound_method.loc10_24.3(%addr.loc10_24.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9_24.4: <bound method> = bound_method %.loc9_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ce4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.19: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ce4, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.7fe) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cfd]
// CHECK:STDOUT:   %bound_method.loc9_24.4: <bound method> = bound_method %.loc9_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.19
// CHECK:STDOUT:   %addr.loc9_24.4: %ptr.3bd = addr_of %.loc9_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9_24.4: init %empty_tuple.type = call %bound_method.loc9_24.4(%addr.loc9_24.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8_24.5: <bound method> = bound_method %.loc8_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d98
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.20: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d98, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4dd) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.466]
// CHECK:STDOUT:   %bound_method.loc8_24.5: <bound method> = bound_method %.loc8_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.20
// CHECK:STDOUT:   %addr.loc8_24.5: %ptr.625 = addr_of %.loc8_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8_24.5: init %empty_tuple.type = call %bound_method.loc8_24.5(%addr.loc8_24.5)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc7_24.6: <bound method> = bound_method %.loc7_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.21: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.e56) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.67c]
// CHECK:STDOUT:   %bound_method.loc7_24.6: <bound method> = bound_method %.loc7_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.21
// CHECK:STDOUT:   %addr.loc7_24.6: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc7_24.6: init %empty_tuple.type = call %bound_method.loc7_24.6(%addr.loc7_24.6)
// CHECK:STDOUT:   return %.loc12_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc12:
// CHECK:STDOUT:   %false.loc13: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc13 br !if.then.loc13 else br !if.else.loc13
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc13:
// CHECK:STDOUT:   %.loc13_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc13: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc13: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:   %impl.elem0.loc13_34: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc13_34.1: <bound method> = bound_method %int_6, %impl.elem0.loc13_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a4a]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_34.2: <bound method> = bound_method %int_6, %specific_fn.loc13 [concrete = constants.%bound_method.b96]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %i32 = call %bound_method.loc13_34.2(%int_6) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc13_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc13_34.2: %i32 = converted %int_6, %.loc13_34.1 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %C.loc13: type = class_type @C, @C(constants.%int_6.e56) [concrete = constants.%C.c60]
// CHECK:STDOUT:   %.loc13_24.2: ref %C.c60 = temporary_storage
// CHECK:STDOUT:   %.loc13_24.3: init %C.c60 = class_init (), %.loc13_24.2 [concrete = constants.%C.val.a4b]
// CHECK:STDOUT:   %.loc13_24.4: ref %C.c60 = temporary %.loc13_24.2, %.loc13_24.3
// CHECK:STDOUT:   %.loc13_26.1: ref %C.c60 = converted %.loc13_24.1, %.loc13_24.4
// CHECK:STDOUT:   %impl.elem0.loc13_35: %.1bb = impl_witness_access constants.%ImplicitAs.impl_witness.ebf, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.ad0]
// CHECK:STDOUT:   %bound_method.loc13_35: <bound method> = bound_method %.loc13_26.1, %impl.elem0.loc13_35
// CHECK:STDOUT:   %.loc13_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc13_26.2: %C.c60 = acquire_value %.loc13_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc13: init %D = call %bound_method.loc13_35(%.loc13_26.2) to %.loc13_35.1
// CHECK:STDOUT:   %.loc13_35.2: init %D = converted %.loc13_26.1, %C.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc13_24.1: <bound method> = bound_method %.loc13_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.bf7
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.22: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.bf7, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.14a) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.5e9]
// CHECK:STDOUT:   %bound_method.loc13_24.1: <bound method> = bound_method %.loc13_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.22
// CHECK:STDOUT:   %addr.loc13_24.1: %ptr.b5e = addr_of %.loc13_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc13_24.1: init %empty_tuple.type = call %bound_method.loc13_24.1(%addr.loc13_24.1)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc12_24.2: <bound method> = bound_method %.loc12_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.217
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.23: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.217, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.430) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.bcb]
// CHECK:STDOUT:   %bound_method.loc12_24.2: <bound method> = bound_method %.loc12_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.23
// CHECK:STDOUT:   %addr.loc12_24.2: %ptr.24c = addr_of %.loc12_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc12_24.2: init %empty_tuple.type = call %bound_method.loc12_24.2(%addr.loc12_24.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc11_24.3: <bound method> = bound_method %.loc11_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d47
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.24: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d47, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.b4c) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.fe8]
// CHECK:STDOUT:   %bound_method.loc11_24.3: <bound method> = bound_method %.loc11_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.24
// CHECK:STDOUT:   %addr.loc11_24.3: %ptr.c28b = addr_of %.loc11_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc11_24.3: init %empty_tuple.type = call %bound_method.loc11_24.3(%addr.loc11_24.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_24.4: <bound method> = bound_method %.loc10_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.931
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.25: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.931, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.99b) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.20a]
// CHECK:STDOUT:   %bound_method.loc10_24.4: <bound method> = bound_method %.loc10_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.25
// CHECK:STDOUT:   %addr.loc10_24.4: %ptr.2b1 = addr_of %.loc10_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_24.4: init %empty_tuple.type = call %bound_method.loc10_24.4(%addr.loc10_24.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9_24.5: <bound method> = bound_method %.loc9_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ce4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.26: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ce4, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.7fe) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cfd]
// CHECK:STDOUT:   %bound_method.loc9_24.5: <bound method> = bound_method %.loc9_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.26
// CHECK:STDOUT:   %addr.loc9_24.5: %ptr.3bd = addr_of %.loc9_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9_24.5: init %empty_tuple.type = call %bound_method.loc9_24.5(%addr.loc9_24.5)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8_24.6: <bound method> = bound_method %.loc8_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d98
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.27: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d98, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4dd) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.466]
// CHECK:STDOUT:   %bound_method.loc8_24.6: <bound method> = bound_method %.loc8_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.27
// CHECK:STDOUT:   %addr.loc8_24.6: %ptr.625 = addr_of %.loc8_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8_24.6: init %empty_tuple.type = call %bound_method.loc8_24.6(%addr.loc8_24.6)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc7_24.7: <bound method> = bound_method %.loc7_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.28: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.e56) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.67c]
// CHECK:STDOUT:   %bound_method.loc7_24.7: <bound method> = bound_method %.loc7_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.28
// CHECK:STDOUT:   %addr.loc7_24.7: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc7_24.7: init %empty_tuple.type = call %bound_method.loc7_24.7(%addr.loc7_24.7)
// CHECK:STDOUT:   return %.loc13_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc13:
// CHECK:STDOUT:   %false.loc14: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc14 br !if.then.loc14 else br !if.else.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc14:
// CHECK:STDOUT:   %.loc14_24.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %P.ref.loc14: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %C.ref.loc14: %C.type = name_ref C, imports.%P.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:   %impl.elem0.loc14_34: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc14_34.1: <bound method> = bound_method %int_7, %impl.elem0.loc14_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a66]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.2: <bound method> = bound_method %int_7, %specific_fn.loc14 [concrete = constants.%bound_method.1db]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14: init %i32 = call %bound_method.loc14_34.2(%int_7) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc14_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc14_34.2: %i32 = converted %int_7, %.loc14_34.1 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %C.loc14: type = class_type @C, @C(constants.%int_7.0b1) [concrete = constants.%C.304]
// CHECK:STDOUT:   %.loc14_24.2: ref %C.304 = temporary_storage
// CHECK:STDOUT:   %.loc14_24.3: init %C.304 = class_init (), %.loc14_24.2 [concrete = constants.%C.val.f9f]
// CHECK:STDOUT:   %.loc14_24.4: ref %C.304 = temporary %.loc14_24.2, %.loc14_24.3
// CHECK:STDOUT:   %.loc14_26.1: ref %C.304 = converted %.loc14_24.1, %.loc14_24.4
// CHECK:STDOUT:   %impl.elem0.loc14_35: %.bc1 = impl_witness_access constants.%ImplicitAs.impl_witness.b93, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.9be]
// CHECK:STDOUT:   %bound_method.loc14_35: <bound method> = bound_method %.loc14_26.1, %impl.elem0.loc14_35
// CHECK:STDOUT:   %.loc14_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc14_26.2: %C.304 = acquire_value %.loc14_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc14: init %D = call %bound_method.loc14_35(%.loc14_26.2) to %.loc14_35.1
// CHECK:STDOUT:   %.loc14_35.2: init %D = converted %.loc14_26.1, %C.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc14_24.1: <bound method> = bound_method %.loc14_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.85d
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.29: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.85d, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.488) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.eb6]
// CHECK:STDOUT:   %bound_method.loc14_24.1: <bound method> = bound_method %.loc14_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.29
// CHECK:STDOUT:   %addr.loc14_24.1: %ptr.dc3 = addr_of %.loc14_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc14_24.1: init %empty_tuple.type = call %bound_method.loc14_24.1(%addr.loc14_24.1)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc13_24.2: <bound method> = bound_method %.loc13_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.bf7
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.30: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.bf7, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.14a) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.5e9]
// CHECK:STDOUT:   %bound_method.loc13_24.2: <bound method> = bound_method %.loc13_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.30
// CHECK:STDOUT:   %addr.loc13_24.2: %ptr.b5e = addr_of %.loc13_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc13_24.2: init %empty_tuple.type = call %bound_method.loc13_24.2(%addr.loc13_24.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc12_24.3: <bound method> = bound_method %.loc12_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.217
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.31: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.217, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.430) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.bcb]
// CHECK:STDOUT:   %bound_method.loc12_24.3: <bound method> = bound_method %.loc12_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.31
// CHECK:STDOUT:   %addr.loc12_24.3: %ptr.24c = addr_of %.loc12_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc12_24.3: init %empty_tuple.type = call %bound_method.loc12_24.3(%addr.loc12_24.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc11_24.4: <bound method> = bound_method %.loc11_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d47
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.32: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d47, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.b4c) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.fe8]
// CHECK:STDOUT:   %bound_method.loc11_24.4: <bound method> = bound_method %.loc11_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.32
// CHECK:STDOUT:   %addr.loc11_24.4: %ptr.c28b = addr_of %.loc11_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc11_24.4: init %empty_tuple.type = call %bound_method.loc11_24.4(%addr.loc11_24.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_24.5: <bound method> = bound_method %.loc10_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.931
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.33: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.931, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.99b) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.20a]
// CHECK:STDOUT:   %bound_method.loc10_24.5: <bound method> = bound_method %.loc10_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.33
// CHECK:STDOUT:   %addr.loc10_24.5: %ptr.2b1 = addr_of %.loc10_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_24.5: init %empty_tuple.type = call %bound_method.loc10_24.5(%addr.loc10_24.5)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9_24.6: <bound method> = bound_method %.loc9_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ce4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.34: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ce4, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.7fe) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cfd]
// CHECK:STDOUT:   %bound_method.loc9_24.6: <bound method> = bound_method %.loc9_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.34
// CHECK:STDOUT:   %addr.loc9_24.6: %ptr.3bd = addr_of %.loc9_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9_24.6: init %empty_tuple.type = call %bound_method.loc9_24.6(%addr.loc9_24.6)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8_24.7: <bound method> = bound_method %.loc8_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d98
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.35: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d98, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4dd) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.466]
// CHECK:STDOUT:   %bound_method.loc8_24.7: <bound method> = bound_method %.loc8_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.35
// CHECK:STDOUT:   %addr.loc8_24.7: %ptr.625 = addr_of %.loc8_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8_24.7: init %empty_tuple.type = call %bound_method.loc8_24.7(%addr.loc8_24.7)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc7_24.8: <bound method> = bound_method %.loc7_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.36: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.e56) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.67c]
// CHECK:STDOUT:   %bound_method.loc7_24.8: <bound method> = bound_method %.loc7_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.36
// CHECK:STDOUT:   %addr.loc7_24.8: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc7_24.8: init %empty_tuple.type = call %bound_method.loc7_24.8(%addr.loc7_24.8)
// CHECK:STDOUT:   return %.loc14_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc14:
// CHECK:STDOUT:   %P.ref.loc15: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%P.Make [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc6_15: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc6_15
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc14_24.2: <bound method> = bound_method %.loc14_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.85d
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.37: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.85d, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.488) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.eb6]
// CHECK:STDOUT:   %bound_method.loc14_24.2: <bound method> = bound_method %.loc14_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.37
// CHECK:STDOUT:   %addr.loc14_24.2: %ptr.dc3 = addr_of %.loc14_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc14_24.2: init %empty_tuple.type = call %bound_method.loc14_24.2(%addr.loc14_24.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc13_24.3: <bound method> = bound_method %.loc13_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.bf7
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.38: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.bf7, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.14a) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.5e9]
// CHECK:STDOUT:   %bound_method.loc13_24.3: <bound method> = bound_method %.loc13_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.38
// CHECK:STDOUT:   %addr.loc13_24.3: %ptr.b5e = addr_of %.loc13_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc13_24.3: init %empty_tuple.type = call %bound_method.loc13_24.3(%addr.loc13_24.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc12_24.4: <bound method> = bound_method %.loc12_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.217
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.39: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.217, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.430) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.bcb]
// CHECK:STDOUT:   %bound_method.loc12_24.4: <bound method> = bound_method %.loc12_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.39
// CHECK:STDOUT:   %addr.loc12_24.4: %ptr.24c = addr_of %.loc12_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc12_24.4: init %empty_tuple.type = call %bound_method.loc12_24.4(%addr.loc12_24.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc11_24.5: <bound method> = bound_method %.loc11_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d47
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.40: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d47, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.b4c) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.fe8]
// CHECK:STDOUT:   %bound_method.loc11_24.5: <bound method> = bound_method %.loc11_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.40
// CHECK:STDOUT:   %addr.loc11_24.5: %ptr.c28b = addr_of %.loc11_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc11_24.5: init %empty_tuple.type = call %bound_method.loc11_24.5(%addr.loc11_24.5)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_24.6: <bound method> = bound_method %.loc10_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.931
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.41: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.931, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.99b) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.20a]
// CHECK:STDOUT:   %bound_method.loc10_24.6: <bound method> = bound_method %.loc10_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.41
// CHECK:STDOUT:   %addr.loc10_24.6: %ptr.2b1 = addr_of %.loc10_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_24.6: init %empty_tuple.type = call %bound_method.loc10_24.6(%addr.loc10_24.6)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc9_24.7: <bound method> = bound_method %.loc9_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ce4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.42: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ce4, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.7fe) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.cfd]
// CHECK:STDOUT:   %bound_method.loc9_24.7: <bound method> = bound_method %.loc9_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.42
// CHECK:STDOUT:   %addr.loc9_24.7: %ptr.3bd = addr_of %.loc9_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc9_24.7: init %empty_tuple.type = call %bound_method.loc9_24.7(%addr.loc9_24.7)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8_24.8: <bound method> = bound_method %.loc8_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d98
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.43: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.d98, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.4dd) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.466]
// CHECK:STDOUT:   %bound_method.loc8_24.8: <bound method> = bound_method %.loc8_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.43
// CHECK:STDOUT:   %addr.loc8_24.8: %ptr.625 = addr_of %.loc8_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8_24.8: init %empty_tuple.type = call %bound_method.loc8_24.8(%addr.loc8_24.8)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc7_24.9: <bound method> = bound_method %.loc7_24.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.44: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ed, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.e56) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.67c]
// CHECK:STDOUT:   %bound_method.loc7_24.9: <bound method> = bound_method %.loc7_24.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.44
// CHECK:STDOUT:   %addr.loc7_24.9: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc7_24.9: init %empty_tuple.type = call %bound_method.loc7_24.9(%addr.loc7_24.9)
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.1 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.2 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.3 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.4 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.5 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.6 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.7 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.8 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.fe8) {
// CHECK:STDOUT:   %N => constants.%N.fe8
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
