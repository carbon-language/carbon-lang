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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.506: type = class_type @C, @C(%N.51e) [symbolic]
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
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.2, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_0.6a9, %int_0.6a9) [concrete]
// CHECK:STDOUT:   %C.c8d: type = class_type @C, @C(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.94e: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.00c: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc8 [concrete]
// CHECK:STDOUT:   %pattern_type.f33: type = pattern_type %C.c8d [concrete]
// CHECK:STDOUT:   %Convert.type.cd0: type = fn_type @Convert.loc8 [concrete]
// CHECK:STDOUT:   %Convert.b95: %Convert.type.cd0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.548: type = fn_type @Op.2, @Destroy.impl(%D) [concrete]
// CHECK:STDOUT:   %Op.2d4: %Op.type.548 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.2d4, @Op.2(%D) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.c02: type = class_type @C, @C(%int_1.5d2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c66: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc9 [concrete]
// CHECK:STDOUT:   %pattern_type.07c: type = pattern_type %C.c02 [concrete]
// CHECK:STDOUT:   %Convert.type.af9: type = fn_type @Convert.loc9 [concrete]
// CHECK:STDOUT:   %Convert.9a4: %Convert.type.af9 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.70b: type = class_type @C, @C(%int_2.ef8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f9c: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc10 [concrete]
// CHECK:STDOUT:   %pattern_type.538: type = pattern_type %C.70b [concrete]
// CHECK:STDOUT:   %Convert.type.5f9: type = fn_type @Convert.loc10 [concrete]
// CHECK:STDOUT:   %Convert.aad: %Convert.type.5f9 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.8cb: type = class_type @C, @C(%int_3.822) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c1f: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc11 [concrete]
// CHECK:STDOUT:   %pattern_type.232: type = pattern_type %C.8cb [concrete]
// CHECK:STDOUT:   %Convert.type.d96: type = fn_type @Convert.loc11 [concrete]
// CHECK:STDOUT:   %Convert.9b7: %Convert.type.d96 = struct_value () [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %C.ba2: type = class_type @C, @C(%int_4.940) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.097: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc12 [concrete]
// CHECK:STDOUT:   %pattern_type.fe0: type = pattern_type %C.ba2 [concrete]
// CHECK:STDOUT:   %Convert.type.33c: type = fn_type @Convert.loc12 [concrete]
// CHECK:STDOUT:   %Convert.b7b: %Convert.type.33c = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Convert.bound.4e6: <bound method> = bound_method %int_5.64b, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.a25: <bound method> = bound_method %int_5.64b, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %C.fcd: type = class_type @C, @C(%int_5.0f6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.03f: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc13 [concrete]
// CHECK:STDOUT:   %pattern_type.e96: type = pattern_type %C.fcd [concrete]
// CHECK:STDOUT:   %Convert.type.193: type = fn_type @Convert.loc13 [concrete]
// CHECK:STDOUT:   %Convert.a6f: %Convert.type.193 = struct_value () [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Convert.bound.ce9: <bound method> = bound_method %int_6.462, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.efa: <bound method> = bound_method %int_6.462, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.e56: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %C.675: type = class_type @C, @C(%int_6.e56) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.89c: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc14 [concrete]
// CHECK:STDOUT:   %pattern_type.bc0: type = pattern_type %C.675 [concrete]
// CHECK:STDOUT:   %Convert.type.7d3: type = fn_type @Convert.loc14 [concrete]
// CHECK:STDOUT:   %Convert.f7f: %Convert.type.7d3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Convert.bound.208: <bound method> = bound_method %int_7.29f, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_7.29f, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.0b1: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %C.884: type = class_type @C, @C(%int_7.0b1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.217: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc15 [concrete]
// CHECK:STDOUT:   %pattern_type.c15: type = pattern_type %C.884 [concrete]
// CHECK:STDOUT:   %Convert.type.167: type = fn_type @Convert.loc15 [concrete]
// CHECK:STDOUT:   %Convert.e51: %Convert.type.167 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_9.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_9.1 (constants.%N.51e)]
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
// CHECK:STDOUT:   impl_decl @ImplicitAs.impl.6d8 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc8_9.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_9.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc8_9.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc8_9.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc8_9.2: %i32 = converted %int_0, %.loc8_9.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_0.6a9) [concrete = constants.%C.c8d]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc8 = impl_witness_table (@ImplicitAs.impl.6d8.%Convert.decl), @ImplicitAs.impl.6d8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc8: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc8 [concrete = constants.%ImplicitAs.impl_witness.00c]
// CHECK:STDOUT:   impl_decl @ImplicitAs.impl.b66 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc9_9.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_9.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc9_9.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc9_9.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc9_9.2: %i32 = converted %int_1, %.loc9_9.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_1.5d2) [concrete = constants.%C.c02]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc9 = impl_witness_table (@ImplicitAs.impl.b66.%Convert.decl), @ImplicitAs.impl.b66 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc9: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc9 [concrete = constants.%ImplicitAs.impl_witness.c66]
// CHECK:STDOUT:   impl_decl @ImplicitAs.impl.9de [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc10_9.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_9.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc10_9.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc10_9.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc10_9.2: %i32 = converted %int_2, %.loc10_9.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_2.ef8) [concrete = constants.%C.70b]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc10 = impl_witness_table (@ImplicitAs.impl.9de.%Convert.decl), @ImplicitAs.impl.9de [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc10: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc10 [concrete = constants.%ImplicitAs.impl_witness.f9c]
// CHECK:STDOUT:   impl_decl @ImplicitAs.impl.649 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc11_9.1: <bound method> = bound_method %int_3, %impl.elem0 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_9.2: <bound method> = bound_method %int_3, %specific_fn [concrete = constants.%bound_method.047]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc11_9.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc11_9.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc11_9.2: %i32 = converted %int_3, %.loc11_9.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_3.822) [concrete = constants.%C.8cb]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc11 = impl_witness_table (@ImplicitAs.impl.649.%Convert.decl), @ImplicitAs.impl.649 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc11: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc11 [concrete = constants.%ImplicitAs.impl_witness.c1f]
// CHECK:STDOUT:   impl_decl @ImplicitAs.impl.49b [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc12_9.1: <bound method> = bound_method %int_4, %impl.elem0 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc12_9.2: <bound method> = bound_method %int_4, %specific_fn [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc12_9.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %.loc12_9.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %.loc12_9.2: %i32 = converted %int_4, %.loc12_9.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_4.940) [concrete = constants.%C.ba2]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc12 = impl_witness_table (@ImplicitAs.impl.49b.%Convert.decl), @ImplicitAs.impl.49b [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc12: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc12 [concrete = constants.%ImplicitAs.impl_witness.097]
// CHECK:STDOUT:   impl_decl @ImplicitAs.impl.75f [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:     %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc13_9.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Convert.bound.4e6]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc13_9.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method.a25]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc13_9.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %.loc13_9.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %.loc13_9.2: %i32 = converted %int_5, %.loc13_9.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_5.0f6) [concrete = constants.%C.fcd]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc13 = impl_witness_table (@ImplicitAs.impl.75f.%Convert.decl), @ImplicitAs.impl.75f [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc13: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc13 [concrete = constants.%ImplicitAs.impl_witness.03f]
// CHECK:STDOUT:   impl_decl @ImplicitAs.impl.98f [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:     %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %int_6, %impl.elem0 [concrete = constants.%Convert.bound.ce9]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %int_6, %specific_fn [concrete = constants.%bound_method.efa]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc14_9.2(%int_6) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:     %.loc14_9.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_6.e56]
// CHECK:STDOUT:     %.loc14_9.2: %i32 = converted %int_6, %.loc14_9.1 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_6.e56) [concrete = constants.%C.675]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc14 = impl_witness_table (@ImplicitAs.impl.98f.%Convert.decl), @ImplicitAs.impl.98f [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc14: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc14 [concrete = constants.%ImplicitAs.impl_witness.89c]
// CHECK:STDOUT:   impl_decl @ImplicitAs.impl.9fd [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:     %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc15_9.1: <bound method> = bound_method %int_7, %impl.elem0 [concrete = constants.%Convert.bound.208]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc15_9.2: <bound method> = bound_method %int_7, %specific_fn [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc15_9.2(%int_7) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:     %.loc15_9.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:     %.loc15_9.2: %i32 = converted %int_7, %.loc15_9.1 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_7.0b1) [concrete = constants.%C.884]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.94e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc15 = impl_witness_table (@ImplicitAs.impl.9fd.%Convert.decl), @ImplicitAs.impl.9fd [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc15: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc15 [concrete = constants.%ImplicitAs.impl_witness.217]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl.6d8: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.cd0 = fn_decl @Convert.loc8 [concrete = constants.%Convert.b95] {
// CHECK:STDOUT:     %self.patt: %pattern_type.f33 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f33 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.c8d = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @ImplicitAs.impl.6d8.%C [concrete = constants.%C.c8d]
// CHECK:STDOUT:     %self: %C.c8d = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl.b66: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.af9 = fn_decl @Convert.loc9 [concrete = constants.%Convert.9a4] {
// CHECK:STDOUT:     %self.patt: %pattern_type.07c = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.07c = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.c02 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @ImplicitAs.impl.b66.%C [concrete = constants.%C.c02]
// CHECK:STDOUT:     %self: %C.c02 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl.9de: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.5f9 = fn_decl @Convert.loc10 [concrete = constants.%Convert.aad] {
// CHECK:STDOUT:     %self.patt: %pattern_type.538 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.538 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.70b = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @ImplicitAs.impl.9de.%C [concrete = constants.%C.70b]
// CHECK:STDOUT:     %self: %C.70b = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl.649: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.d96 = fn_decl @Convert.loc11 [concrete = constants.%Convert.9b7] {
// CHECK:STDOUT:     %self.patt: %pattern_type.232 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.232 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.8cb = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @ImplicitAs.impl.649.%C [concrete = constants.%C.8cb]
// CHECK:STDOUT:     %self: %C.8cb = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl.49b: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.33c = fn_decl @Convert.loc12 [concrete = constants.%Convert.b7b] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fe0 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fe0 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.ba2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @ImplicitAs.impl.49b.%C [concrete = constants.%C.ba2]
// CHECK:STDOUT:     %self: %C.ba2 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl.75f: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.193 = fn_decl @Convert.loc13 [concrete = constants.%Convert.a6f] {
// CHECK:STDOUT:     %self.patt: %pattern_type.e96 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e96 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.fcd = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @ImplicitAs.impl.75f.%C [concrete = constants.%C.fcd]
// CHECK:STDOUT:     %self: %C.fcd = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl.98f: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.7d3 = fn_decl @Convert.loc14 [concrete = constants.%Convert.f7f] {
// CHECK:STDOUT:     %self.patt: %pattern_type.bc0 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.bc0 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.675 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @ImplicitAs.impl.98f.%C [concrete = constants.%C.675]
// CHECK:STDOUT:     %self: %C.675 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl.9fd: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.167 = fn_decl @Convert.loc15 [concrete = constants.%Convert.e51] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c15 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c15 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.884 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @ImplicitAs.impl.9fd.%C [concrete = constants.%C.884]
// CHECK:STDOUT:     %self: %C.884 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   .Make = <poisoned>
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%N.loc4_9.2: %i32) {
// CHECK:STDOUT:   %N.loc4_9.1: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_9.1 (constants.%N.51e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.506
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
// CHECK:STDOUT:   %struct_type.n.m: type = struct_type {.n: %i32, .m: %i32} [concrete = constants.%struct_type.n.m.de8]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n.m [concrete = constants.%complete_type.ea0]
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
// CHECK:STDOUT:   %impl.elem0.loc6_40.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc6_40.1: <bound method> = bound_method %int_0.loc6_31, %impl.elem0.loc6_40.1 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc6_40.1: <specific function> = specific_function %impl.elem0.loc6_40.1, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.2: <bound method> = bound_method %int_0.loc6_31, %specific_fn.loc6_40.1 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc6_40.1: init %i32 = call %bound_method.loc6_40.2(%int_0.loc6_31) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.2: init %i32 = converted %int_0.loc6_31, %int.convert_checked.loc6_40.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.3: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc6_40.4: init %i32 = initialize_from %.loc6_40.2 to %.loc6_40.3 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %impl.elem0.loc6_40.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc6_40.3: <bound method> = bound_method %int_0.loc6_39, %impl.elem0.loc6_40.2 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc6_40.2: <specific function> = specific_function %impl.elem0.loc6_40.2, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.4: <bound method> = bound_method %int_0.loc6_39, %specific_fn.loc6_40.2 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc6_40.2: init %i32 = call %bound_method.loc6_40.4(%int_0.loc6_39) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.5: init %i32 = converted %int_0.loc6_39, %int.convert_checked.loc6_40.2 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.6: ref %i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc6_40.7: init %i32 = initialize_from %.loc6_40.5 to %.loc6_40.6 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.8: init %D = class_init (%.loc6_40.4, %.loc6_40.7), %return [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc6_41: init %D = converted %.loc6_40.1, %.loc6_40.8 [concrete = constants.%D.val]
// CHECK:STDOUT:   return %.loc6_41 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.loc8(%self.param: %C.c8d) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc8: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc8
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc8, constants.%Op.2d4
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.2d4, @Op.2(constants.%D) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.19c = addr_of %.loc8
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.loc9(%self.param: %C.c02) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc9: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc9
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc9, constants.%Op.2d4
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.2d4, @Op.2(constants.%D) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc9, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.19c = addr_of %.loc9
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.loc10(%self.param: %C.70b) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc10: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc10
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc10, constants.%Op.2d4
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.2d4, @Op.2(constants.%D) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc10, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.19c = addr_of %.loc10
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.loc11(%self.param: %C.8cb) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc11: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc11
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc11, constants.%Op.2d4
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.2d4, @Op.2(constants.%D) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc11, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.19c = addr_of %.loc11
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.loc12(%self.param: %C.ba2) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc12: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc12
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc12, constants.%Op.2d4
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.2d4, @Op.2(constants.%D) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc12, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.19c = addr_of %.loc12
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.loc13(%self.param: %C.fcd) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc13: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc13
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc13, constants.%Op.2d4
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.2d4, @Op.2(constants.%D) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc13, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.19c = addr_of %.loc13
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.loc14(%self.param: %C.675) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc14: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc14
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc14, constants.%Op.2d4
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.2d4, @Op.2(constants.%D) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc14, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.19c = addr_of %.loc14
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.loc15(%self.param: %C.884) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc15: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc15
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.loc15, constants.%Op.2d4
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.2d4, @Op.2(constants.%D) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc15, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.19c = addr_of %.loc15
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc4_9.1 => constants.%N.51e
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
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.10, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %C.b00: type = class_type @C, @C(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %C.val.452: %C.b00 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5f9: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %Convert.type.334: type = fn_type @Convert.10, @ImplicitAs(%D) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f53: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f4b [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.014: %ImplicitAs.type.5f9 = facet_value %C.b00, (%ImplicitAs.impl_witness.f53) [concrete]
// CHECK:STDOUT:   %.a9c: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.014 [concrete]
// CHECK:STDOUT:   %Convert.type.971: type = fn_type @Convert.2 [concrete]
// CHECK:STDOUT:   %Convert.bf9: %Convert.type.971 = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.69c: type = fn_type @Op.1, @Destroy.impl(%D) [concrete]
// CHECK:STDOUT:   %Op.f5a: %Op.type.69c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.f29: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %Op.specific_fn.bd1: <specific function> = specific_function %Op.f5a, @Op.1(%D) [concrete]
// CHECK:STDOUT:   %Op.type.94c: type = fn_type @Op.1, @Destroy.impl(%C.b00) [concrete]
// CHECK:STDOUT:   %Op.ae6: %Op.type.94c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.697: type = ptr_type %C.b00 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.e42: <specific function> = specific_function %Op.ae6, @Op.1(%C.b00) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.678: %C.674 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fcb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.297 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b8a: %ImplicitAs.type.5f9 = facet_value %C.674, (%ImplicitAs.impl_witness.fcb) [concrete]
// CHECK:STDOUT:   %.7d6: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.b8a [concrete]
// CHECK:STDOUT:   %Convert.type.bdd: type = fn_type @Convert.3 [concrete]
// CHECK:STDOUT:   %Convert.c82: %Convert.type.bdd = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.b6b: type = fn_type @Op.1, @Destroy.impl(%C.674) [concrete]
// CHECK:STDOUT:   %Op.ad8: %Op.type.b6b = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.625: type = ptr_type %C.674 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.6b6: <specific function> = specific_function %Op.ad8, @Op.1(%C.674) [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.fb5: %C.681 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.746: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8b3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d55: %ImplicitAs.type.5f9 = facet_value %C.681, (%ImplicitAs.impl_witness.746) [concrete]
// CHECK:STDOUT:   %.987: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.d55 [concrete]
// CHECK:STDOUT:   %Convert.type.4c1: type = fn_type @Convert.4 [concrete]
// CHECK:STDOUT:   %Convert.08c: %Convert.type.4c1 = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.a31: type = fn_type @Op.1, @Destroy.impl(%C.681) [concrete]
// CHECK:STDOUT:   %Op.404: %Op.type.a31 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.3bd: type = ptr_type %C.681 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.822: <specific function> = specific_function %Op.404, @Op.1(%C.681) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.fe7: %C.7ac = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f79: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.db0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.aa1: %ImplicitAs.type.5f9 = facet_value %C.7ac, (%ImplicitAs.impl_witness.f79) [concrete]
// CHECK:STDOUT:   %.efa: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.aa1 [concrete]
// CHECK:STDOUT:   %Convert.type.f89: type = fn_type @Convert.5 [concrete]
// CHECK:STDOUT:   %Convert.2dd: %Convert.type.f89 = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.100: type = fn_type @Op.1, @Destroy.impl(%C.7ac) [concrete]
// CHECK:STDOUT:   %Op.9cf: %Op.type.100 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.2b1: type = ptr_type %C.7ac [concrete]
// CHECK:STDOUT:   %Op.specific_fn.694: <specific function> = specific_function %Op.9cf, @Op.1(%C.7ac) [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.1dd: %C.89d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.16c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.54d [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.750: %ImplicitAs.type.5f9 = facet_value %C.89d, (%ImplicitAs.impl_witness.16c) [concrete]
// CHECK:STDOUT:   %.26e: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.750 [concrete]
// CHECK:STDOUT:   %Convert.type.2ce: type = fn_type @Convert.6 [concrete]
// CHECK:STDOUT:   %Convert.009: %Convert.type.2ce = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.86c: type = fn_type @Op.1, @Destroy.impl(%C.89d) [concrete]
// CHECK:STDOUT:   %Op.6b4: %Op.type.86c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28b: type = ptr_type %C.89d [concrete]
// CHECK:STDOUT:   %Op.specific_fn.357: <specific function> = specific_function %Op.6b4, @Op.1(%C.89d) [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Convert.bound.4e6: <bound method> = bound_method %int_5.64b, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.a25: <bound method> = bound_method %int_5.64b, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.a4a: %C.f0a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.aa9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e3f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.8cc: %ImplicitAs.type.5f9 = facet_value %C.f0a, (%ImplicitAs.impl_witness.aa9) [concrete]
// CHECK:STDOUT:   %.eaf: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.8cc [concrete]
// CHECK:STDOUT:   %Convert.type.e9d: type = fn_type @Convert.7 [concrete]
// CHECK:STDOUT:   %Convert.b66: %Convert.type.e9d = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.4df: type = fn_type @Op.1, @Destroy.impl(%C.f0a) [concrete]
// CHECK:STDOUT:   %Op.985: %Op.type.4df = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.24c: type = ptr_type %C.f0a [concrete]
// CHECK:STDOUT:   %Op.specific_fn.a66: <specific function> = specific_function %Op.985, @Op.1(%C.f0a) [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Convert.bound.ce9: <bound method> = bound_method %int_6.462, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.efa: <bound method> = bound_method %int_6.462, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.a4b: %C.c60 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e4d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b31 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a69: %ImplicitAs.type.5f9 = facet_value %C.c60, (%ImplicitAs.impl_witness.e4d) [concrete]
// CHECK:STDOUT:   %.874: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.a69 [concrete]
// CHECK:STDOUT:   %Convert.type.4b8: type = fn_type @Convert.8 [concrete]
// CHECK:STDOUT:   %Convert.f59: %Convert.type.4b8 = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.32d: type = fn_type @Op.1, @Destroy.impl(%C.c60) [concrete]
// CHECK:STDOUT:   %Op.ca5: %Op.type.32d = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b5e: type = ptr_type %C.c60 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.8d0: <specific function> = specific_function %Op.ca5, @Op.1(%C.c60) [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Convert.bound.208: <bound method> = bound_method %int_7.29f, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_7.29f, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.f9f: %C.304 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.0ca: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.056 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.cca: %ImplicitAs.type.5f9 = facet_value %C.304, (%ImplicitAs.impl_witness.0ca) [concrete]
// CHECK:STDOUT:   %.79b: type = fn_type_with_self_type %Convert.type.334, %ImplicitAs.facet.cca [concrete]
// CHECK:STDOUT:   %Convert.type.c7e: type = fn_type @Convert.9 [concrete]
// CHECK:STDOUT:   %Convert.6c1: %Convert.type.c7e = struct_value () [concrete]
// CHECK:STDOUT:   %Op.type.986: type = fn_type @Op.1, @Destroy.impl(%C.304) [concrete]
// CHECK:STDOUT:   %Op.6fc: %Op.type.986 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.dc3: type = ptr_type %C.304 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.a08: <specific function> = specific_function %Op.6fc, @Op.1(%C.304) [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
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
// CHECK:STDOUT:   %P.import_ref.cab = import_ref P//library, inst57 [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.a52 = import_ref P//library, loc5_16, unloaded
// CHECK:STDOUT:   %P.import_ref.b4a = import_ref P//library, loc5_28, unloaded
// CHECK:STDOUT:   %P.C: %C.type = import_ref P//library, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %P.import_ref.1b7: %i32 = import_ref P//library, loc4_9, loaded [symbolic = @C.%N (constants.%N.51e)]
// CHECK:STDOUT:   %P.import_ref.8f2: <witness> = import_ref P//library, loc4_19, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %P.import_ref.f65 = import_ref P//library, inst51 [no loc], unloaded
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
// CHECK:STDOUT:   %P.import_ref.708: <witness> = import_ref P//library, loc8_33, loaded [concrete = constants.%ImplicitAs.impl_witness.f53]
// CHECK:STDOUT:   %P.import_ref.d2c: type = import_ref P//library, loc8_9, loaded [concrete = constants.%C.b00]
// CHECK:STDOUT:   %P.import_ref.b769fa.1: type = import_ref P//library, loc8_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.ae8: <witness> = import_ref P//library, loc9_33, loaded [concrete = constants.%ImplicitAs.impl_witness.fcb]
// CHECK:STDOUT:   %P.import_ref.cf3: type = import_ref P//library, loc9_9, loaded [concrete = constants.%C.674]
// CHECK:STDOUT:   %P.import_ref.b769fa.2: type = import_ref P//library, loc9_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.2fc: <witness> = import_ref P//library, loc10_33, loaded [concrete = constants.%ImplicitAs.impl_witness.746]
// CHECK:STDOUT:   %P.import_ref.095: type = import_ref P//library, loc10_9, loaded [concrete = constants.%C.681]
// CHECK:STDOUT:   %P.import_ref.b769fa.3: type = import_ref P//library, loc10_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.d75: <witness> = import_ref P//library, loc11_33, loaded [concrete = constants.%ImplicitAs.impl_witness.f79]
// CHECK:STDOUT:   %P.import_ref.82b: type = import_ref P//library, loc11_9, loaded [concrete = constants.%C.7ac]
// CHECK:STDOUT:   %P.import_ref.b769fa.4: type = import_ref P//library, loc11_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.6c2: <witness> = import_ref P//library, loc12_33, loaded [concrete = constants.%ImplicitAs.impl_witness.16c]
// CHECK:STDOUT:   %P.import_ref.f7d: type = import_ref P//library, loc12_9, loaded [concrete = constants.%C.89d]
// CHECK:STDOUT:   %P.import_ref.b769fa.5: type = import_ref P//library, loc12_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.0e0: <witness> = import_ref P//library, loc13_33, loaded [concrete = constants.%ImplicitAs.impl_witness.aa9]
// CHECK:STDOUT:   %P.import_ref.a43: type = import_ref P//library, loc13_9, loaded [concrete = constants.%C.f0a]
// CHECK:STDOUT:   %P.import_ref.b769fa.6: type = import_ref P//library, loc13_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.e6a: <witness> = import_ref P//library, loc14_33, loaded [concrete = constants.%ImplicitAs.impl_witness.e4d]
// CHECK:STDOUT:   %P.import_ref.654: type = import_ref P//library, loc14_9, loaded [concrete = constants.%C.c60]
// CHECK:STDOUT:   %P.import_ref.b769fa.7: type = import_ref P//library, loc14_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.5f0: <witness> = import_ref P//library, loc15_33, loaded [concrete = constants.%ImplicitAs.impl_witness.0ca]
// CHECK:STDOUT:   %P.import_ref.dae: type = import_ref P//library, loc15_9, loaded [concrete = constants.%C.304]
// CHECK:STDOUT:   %P.import_ref.b769fa.8: type = import_ref P//library, loc15_31, loaded [concrete = constants.%ImplicitAs.type.5f9]
// CHECK:STDOUT:   %P.import_ref.e36: %Convert.type.971 = import_ref P//library, loc8_65, loaded [concrete = constants.%Convert.bf9]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f4b = impl_witness_table (%P.import_ref.e36), @ImplicitAs.impl.284 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %P.import_ref.38e: %Convert.type.bdd = import_ref P//library, loc9_65, loaded [concrete = constants.%Convert.c82]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.297 = impl_witness_table (%P.import_ref.38e), @ImplicitAs.impl.628 [concrete]
// CHECK:STDOUT:   %P.import_ref.5ab: %Convert.type.4c1 = import_ref P//library, loc10_65, loaded [concrete = constants.%Convert.08c]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8b3 = impl_witness_table (%P.import_ref.5ab), @ImplicitAs.impl.88c [concrete]
// CHECK:STDOUT:   %P.import_ref.db1: %Convert.type.f89 = import_ref P//library, loc11_65, loaded [concrete = constants.%Convert.2dd]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.db0 = impl_witness_table (%P.import_ref.db1), @ImplicitAs.impl.f76 [concrete]
// CHECK:STDOUT:   %P.import_ref.134: %Convert.type.2ce = import_ref P//library, loc12_65, loaded [concrete = constants.%Convert.009]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.54d = impl_witness_table (%P.import_ref.134), @ImplicitAs.impl.521 [concrete]
// CHECK:STDOUT:   %P.import_ref.9a0: %Convert.type.e9d = import_ref P//library, loc13_65, loaded [concrete = constants.%Convert.b66]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e3f = impl_witness_table (%P.import_ref.9a0), @ImplicitAs.impl.ac4 [concrete]
// CHECK:STDOUT:   %P.import_ref.b13: %Convert.type.4b8 = import_ref P//library, loc14_65, loaded [concrete = constants.%Convert.f59]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b31 = impl_witness_table (%P.import_ref.b13), @ImplicitAs.impl.3bf [concrete]
// CHECK:STDOUT:   %P.import_ref.216: %Convert.type.c7e = import_ref P//library, loc15_65, loaded [concrete = constants.%Convert.6c1]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.056 = impl_witness_table (%P.import_ref.216), @ImplicitAs.impl.935 [concrete]
// CHECK:STDOUT:   %P.Make: %Make.type = import_ref P//library, Make, loaded [concrete = constants.%Make]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .P = imports.%P
// CHECK:STDOUT:     .F0 = %F0.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %P.import = import P
// CHECK:STDOUT:   %F0.decl: %F0.type = fn_decl @F0 [concrete = constants.%F0] {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = binding_pattern n [concrete]
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
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl.284: imports.%P.import_ref.d2c as imports.%P.import_ref.b769fa.1 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.708
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl.628: imports.%P.import_ref.cf3 as imports.%P.import_ref.b769fa.2 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.ae8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl.88c: imports.%P.import_ref.095 as imports.%P.import_ref.b769fa.3 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.2fc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl.f76: imports.%P.import_ref.82b as imports.%P.import_ref.b769fa.4 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.d75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl.521: imports.%P.import_ref.f7d as imports.%P.import_ref.b769fa.5 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.6c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl.ac4: imports.%P.import_ref.a43 as imports.%P.import_ref.b769fa.6 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.0e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl.3bf: imports.%P.import_ref.654 as imports.%P.import_ref.b769fa.7 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.e6a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl.935: imports.%P.import_ref.dae as imports.%P.import_ref.b769fa.8 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.5f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D [from "library.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%P.import_ref.a7d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%P.import_ref.cab
// CHECK:STDOUT:   .n = imports.%P.import_ref.a52
// CHECK:STDOUT:   .m = imports.%P.import_ref.b4a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%P.import_ref.1b7: %i32) [from "library.carbon"] {
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.51e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%P.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%P.import_ref.f65
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
// CHECK:STDOUT:   %impl.elem0.loc7_34: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc7_34.1: <bound method> = bound_method %int_0, %impl.elem0.loc7_34 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7_34, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_34.2: <bound method> = bound_method %int_0, %specific_fn.loc7 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc7: init %i32 = call %bound_method.loc7_34.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc7_34.1: %i32 = value_of_initializer %int.convert_checked.loc7 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc7_34.2: %i32 = converted %int_0, %.loc7_34.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %C.loc7: type = class_type @C, @C(constants.%int_0.6a9) [concrete = constants.%C.b00]
// CHECK:STDOUT:   %.loc7_24.2: ref %C.b00 = temporary_storage
// CHECK:STDOUT:   %.loc7_24.3: init %C.b00 = class_init (), %.loc7_24.2 [concrete = constants.%C.val.452]
// CHECK:STDOUT:   %.loc7_24.4: ref %C.b00 = temporary %.loc7_24.2, %.loc7_24.3
// CHECK:STDOUT:   %.loc7_26.1: ref %C.b00 = converted %.loc7_24.1, %.loc7_24.4
// CHECK:STDOUT:   %impl.elem0.loc7_35: %.a9c = impl_witness_access constants.%ImplicitAs.impl_witness.f53, element0 [concrete = constants.%Convert.bf9]
// CHECK:STDOUT:   %bound_method.loc7_35.1: <bound method> = bound_method %.loc7_26.1, %impl.elem0.loc7_35
// CHECK:STDOUT:   %.loc7_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc7_26.2: %C.b00 = bind_value %.loc7_26.1
// CHECK:STDOUT:   %Convert.call.loc7: init %D = call %bound_method.loc7_35.1(%.loc7_26.2) to %.loc7_35.1
// CHECK:STDOUT:   %.loc7_35.2: init %D = converted %.loc7_26.1, %Convert.call.loc7
// CHECK:STDOUT:   %Op.bound.loc7_35.1: <bound method> = bound_method %.loc7_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc7_35.2: <bound method> = bound_method %.loc7_35.1, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc7_35.1: %ptr.f29 = addr_of %.loc7_35.1
// CHECK:STDOUT:   %no_op.loc7_35.1: init %empty_tuple.type = call %bound_method.loc7_35.2(%addr.loc7_35.1)
// CHECK:STDOUT:   %Op.bound.loc7_24.1: <bound method> = bound_method %.loc7_24.2, constants.%Op.ae6
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.ae6, @Op.1(constants.%C.b00) [concrete = constants.%Op.specific_fn.e42]
// CHECK:STDOUT:   %bound_method.loc7_24.1: <bound method> = bound_method %.loc7_24.2, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc7_24.1: %ptr.697 = addr_of %.loc7_24.2
// CHECK:STDOUT:   %no_op.loc7_24.1: init %empty_tuple.type = call %bound_method.loc7_24.1(%addr.loc7_24.1)
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
// CHECK:STDOUT:   %impl.elem0.loc8_34: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_34 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8_34, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_1, %specific_fn.loc8 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %i32 = call %bound_method.loc8_34.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %int.convert_checked.loc8 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_1, %.loc8_34.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %C.loc8: type = class_type @C, @C(constants.%int_1.5d2) [concrete = constants.%C.674]
// CHECK:STDOUT:   %.loc8_24.2: ref %C.674 = temporary_storage
// CHECK:STDOUT:   %.loc8_24.3: init %C.674 = class_init (), %.loc8_24.2 [concrete = constants.%C.val.678]
// CHECK:STDOUT:   %.loc8_24.4: ref %C.674 = temporary %.loc8_24.2, %.loc8_24.3
// CHECK:STDOUT:   %.loc8_26.1: ref %C.674 = converted %.loc8_24.1, %.loc8_24.4
// CHECK:STDOUT:   %impl.elem0.loc8_35: %.7d6 = impl_witness_access constants.%ImplicitAs.impl_witness.fcb, element0 [concrete = constants.%Convert.c82]
// CHECK:STDOUT:   %bound_method.loc8_35.1: <bound method> = bound_method %.loc8_26.1, %impl.elem0.loc8_35
// CHECK:STDOUT:   %.loc8_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc8_26.2: %C.674 = bind_value %.loc8_26.1
// CHECK:STDOUT:   %Convert.call.loc8: init %D = call %bound_method.loc8_35.1(%.loc8_26.2) to %.loc8_35.1
// CHECK:STDOUT:   %.loc8_35.2: init %D = converted %.loc8_26.1, %Convert.call.loc8
// CHECK:STDOUT:   %Op.bound.loc8_35.1: <bound method> = bound_method %.loc8_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.3: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc8_35.2: <bound method> = bound_method %.loc8_35.1, %Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc8_35.1: %ptr.f29 = addr_of %.loc8_35.1
// CHECK:STDOUT:   %no_op.loc8_35.1: init %empty_tuple.type = call %bound_method.loc8_35.2(%addr.loc8_35.1)
// CHECK:STDOUT:   %Op.bound.loc8_24.1: <bound method> = bound_method %.loc8_24.2, constants.%Op.ad8
// CHECK:STDOUT:   %Op.specific_fn.4: <specific function> = specific_function constants.%Op.ad8, @Op.1(constants.%C.674) [concrete = constants.%Op.specific_fn.6b6]
// CHECK:STDOUT:   %bound_method.loc8_24.1: <bound method> = bound_method %.loc8_24.2, %Op.specific_fn.4
// CHECK:STDOUT:   %addr.loc8_24.1: %ptr.625 = addr_of %.loc8_24.2
// CHECK:STDOUT:   %no_op.loc8_24.1: init %empty_tuple.type = call %bound_method.loc8_24.1(%addr.loc8_24.1)
// CHECK:STDOUT:   %Op.bound.loc7_35.2: <bound method> = bound_method %.loc7_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.5: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc7_35.3: <bound method> = bound_method %.loc7_35.1, %Op.specific_fn.5
// CHECK:STDOUT:   %addr.loc7_35.2: %ptr.f29 = addr_of %.loc7_35.1
// CHECK:STDOUT:   %no_op.loc7_35.2: init %empty_tuple.type = call %bound_method.loc7_35.3(%addr.loc7_35.2)
// CHECK:STDOUT:   %Op.bound.loc7_24.2: <bound method> = bound_method %.loc7_24.2, constants.%Op.ae6
// CHECK:STDOUT:   %Op.specific_fn.6: <specific function> = specific_function constants.%Op.ae6, @Op.1(constants.%C.b00) [concrete = constants.%Op.specific_fn.e42]
// CHECK:STDOUT:   %bound_method.loc7_24.2: <bound method> = bound_method %.loc7_24.2, %Op.specific_fn.6
// CHECK:STDOUT:   %addr.loc7_24.2: %ptr.697 = addr_of %.loc7_24.2
// CHECK:STDOUT:   %no_op.loc7_24.2: init %empty_tuple.type = call %bound_method.loc7_24.2(%addr.loc7_24.2)
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
// CHECK:STDOUT:   %impl.elem0.loc9_34: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %int_2, %impl.elem0.loc9_34 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9_34, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %int_2, %specific_fn.loc9 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc9: init %i32 = call %bound_method.loc9_34.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_34.1: %i32 = value_of_initializer %int.convert_checked.loc9 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_34.2: %i32 = converted %int_2, %.loc9_34.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %C.loc9: type = class_type @C, @C(constants.%int_2.ef8) [concrete = constants.%C.681]
// CHECK:STDOUT:   %.loc9_24.2: ref %C.681 = temporary_storage
// CHECK:STDOUT:   %.loc9_24.3: init %C.681 = class_init (), %.loc9_24.2 [concrete = constants.%C.val.fb5]
// CHECK:STDOUT:   %.loc9_24.4: ref %C.681 = temporary %.loc9_24.2, %.loc9_24.3
// CHECK:STDOUT:   %.loc9_26.1: ref %C.681 = converted %.loc9_24.1, %.loc9_24.4
// CHECK:STDOUT:   %impl.elem0.loc9_35: %.987 = impl_witness_access constants.%ImplicitAs.impl_witness.746, element0 [concrete = constants.%Convert.08c]
// CHECK:STDOUT:   %bound_method.loc9_35.1: <bound method> = bound_method %.loc9_26.1, %impl.elem0.loc9_35
// CHECK:STDOUT:   %.loc9_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc9_26.2: %C.681 = bind_value %.loc9_26.1
// CHECK:STDOUT:   %Convert.call.loc9: init %D = call %bound_method.loc9_35.1(%.loc9_26.2) to %.loc9_35.1
// CHECK:STDOUT:   %.loc9_35.2: init %D = converted %.loc9_26.1, %Convert.call.loc9
// CHECK:STDOUT:   %Op.bound.loc9_35.1: <bound method> = bound_method %.loc9_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.7: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc9_35.2: <bound method> = bound_method %.loc9_35.1, %Op.specific_fn.7
// CHECK:STDOUT:   %addr.loc9_35.1: %ptr.f29 = addr_of %.loc9_35.1
// CHECK:STDOUT:   %no_op.loc9_35.1: init %empty_tuple.type = call %bound_method.loc9_35.2(%addr.loc9_35.1)
// CHECK:STDOUT:   %Op.bound.loc9_24.1: <bound method> = bound_method %.loc9_24.2, constants.%Op.404
// CHECK:STDOUT:   %Op.specific_fn.8: <specific function> = specific_function constants.%Op.404, @Op.1(constants.%C.681) [concrete = constants.%Op.specific_fn.822]
// CHECK:STDOUT:   %bound_method.loc9_24.1: <bound method> = bound_method %.loc9_24.2, %Op.specific_fn.8
// CHECK:STDOUT:   %addr.loc9_24.1: %ptr.3bd = addr_of %.loc9_24.2
// CHECK:STDOUT:   %no_op.loc9_24.1: init %empty_tuple.type = call %bound_method.loc9_24.1(%addr.loc9_24.1)
// CHECK:STDOUT:   %Op.bound.loc8_35.2: <bound method> = bound_method %.loc8_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.9: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc8_35.3: <bound method> = bound_method %.loc8_35.1, %Op.specific_fn.9
// CHECK:STDOUT:   %addr.loc8_35.2: %ptr.f29 = addr_of %.loc8_35.1
// CHECK:STDOUT:   %no_op.loc8_35.2: init %empty_tuple.type = call %bound_method.loc8_35.3(%addr.loc8_35.2)
// CHECK:STDOUT:   %Op.bound.loc8_24.2: <bound method> = bound_method %.loc8_24.2, constants.%Op.ad8
// CHECK:STDOUT:   %Op.specific_fn.10: <specific function> = specific_function constants.%Op.ad8, @Op.1(constants.%C.674) [concrete = constants.%Op.specific_fn.6b6]
// CHECK:STDOUT:   %bound_method.loc8_24.2: <bound method> = bound_method %.loc8_24.2, %Op.specific_fn.10
// CHECK:STDOUT:   %addr.loc8_24.2: %ptr.625 = addr_of %.loc8_24.2
// CHECK:STDOUT:   %no_op.loc8_24.2: init %empty_tuple.type = call %bound_method.loc8_24.2(%addr.loc8_24.2)
// CHECK:STDOUT:   %Op.bound.loc7_35.3: <bound method> = bound_method %.loc7_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.11: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc7_35.4: <bound method> = bound_method %.loc7_35.1, %Op.specific_fn.11
// CHECK:STDOUT:   %addr.loc7_35.3: %ptr.f29 = addr_of %.loc7_35.1
// CHECK:STDOUT:   %no_op.loc7_35.3: init %empty_tuple.type = call %bound_method.loc7_35.4(%addr.loc7_35.3)
// CHECK:STDOUT:   %Op.bound.loc7_24.3: <bound method> = bound_method %.loc7_24.2, constants.%Op.ae6
// CHECK:STDOUT:   %Op.specific_fn.12: <specific function> = specific_function constants.%Op.ae6, @Op.1(constants.%C.b00) [concrete = constants.%Op.specific_fn.e42]
// CHECK:STDOUT:   %bound_method.loc7_24.3: <bound method> = bound_method %.loc7_24.2, %Op.specific_fn.12
// CHECK:STDOUT:   %addr.loc7_24.3: %ptr.697 = addr_of %.loc7_24.2
// CHECK:STDOUT:   %no_op.loc7_24.3: init %empty_tuple.type = call %bound_method.loc7_24.3(%addr.loc7_24.3)
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
// CHECK:STDOUT:   %impl.elem0.loc10_34: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc10_34.1: <bound method> = bound_method %int_3, %impl.elem0.loc10_34 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_34, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_34.2: <bound method> = bound_method %int_3, %specific_fn.loc10 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc10: init %i32 = call %bound_method.loc10_34.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc10_34.1: %i32 = value_of_initializer %int.convert_checked.loc10 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc10_34.2: %i32 = converted %int_3, %.loc10_34.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %C.loc10: type = class_type @C, @C(constants.%int_3.822) [concrete = constants.%C.7ac]
// CHECK:STDOUT:   %.loc10_24.2: ref %C.7ac = temporary_storage
// CHECK:STDOUT:   %.loc10_24.3: init %C.7ac = class_init (), %.loc10_24.2 [concrete = constants.%C.val.fe7]
// CHECK:STDOUT:   %.loc10_24.4: ref %C.7ac = temporary %.loc10_24.2, %.loc10_24.3
// CHECK:STDOUT:   %.loc10_26.1: ref %C.7ac = converted %.loc10_24.1, %.loc10_24.4
// CHECK:STDOUT:   %impl.elem0.loc10_35: %.efa = impl_witness_access constants.%ImplicitAs.impl_witness.f79, element0 [concrete = constants.%Convert.2dd]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %.loc10_26.1, %impl.elem0.loc10_35
// CHECK:STDOUT:   %.loc10_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc10_26.2: %C.7ac = bind_value %.loc10_26.1
// CHECK:STDOUT:   %Convert.call.loc10: init %D = call %bound_method.loc10_35.1(%.loc10_26.2) to %.loc10_35.1
// CHECK:STDOUT:   %.loc10_35.2: init %D = converted %.loc10_26.1, %Convert.call.loc10
// CHECK:STDOUT:   %Op.bound.loc10_35.1: <bound method> = bound_method %.loc10_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.13: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %.loc10_35.1, %Op.specific_fn.13
// CHECK:STDOUT:   %addr.loc10_35.1: %ptr.f29 = addr_of %.loc10_35.1
// CHECK:STDOUT:   %no_op.loc10_35.1: init %empty_tuple.type = call %bound_method.loc10_35.2(%addr.loc10_35.1)
// CHECK:STDOUT:   %Op.bound.loc10_24.1: <bound method> = bound_method %.loc10_24.2, constants.%Op.9cf
// CHECK:STDOUT:   %Op.specific_fn.14: <specific function> = specific_function constants.%Op.9cf, @Op.1(constants.%C.7ac) [concrete = constants.%Op.specific_fn.694]
// CHECK:STDOUT:   %bound_method.loc10_24.1: <bound method> = bound_method %.loc10_24.2, %Op.specific_fn.14
// CHECK:STDOUT:   %addr.loc10_24.1: %ptr.2b1 = addr_of %.loc10_24.2
// CHECK:STDOUT:   %no_op.loc10_24.1: init %empty_tuple.type = call %bound_method.loc10_24.1(%addr.loc10_24.1)
// CHECK:STDOUT:   %Op.bound.loc9_35.2: <bound method> = bound_method %.loc9_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.15: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc9_35.3: <bound method> = bound_method %.loc9_35.1, %Op.specific_fn.15
// CHECK:STDOUT:   %addr.loc9_35.2: %ptr.f29 = addr_of %.loc9_35.1
// CHECK:STDOUT:   %no_op.loc9_35.2: init %empty_tuple.type = call %bound_method.loc9_35.3(%addr.loc9_35.2)
// CHECK:STDOUT:   %Op.bound.loc9_24.2: <bound method> = bound_method %.loc9_24.2, constants.%Op.404
// CHECK:STDOUT:   %Op.specific_fn.16: <specific function> = specific_function constants.%Op.404, @Op.1(constants.%C.681) [concrete = constants.%Op.specific_fn.822]
// CHECK:STDOUT:   %bound_method.loc9_24.2: <bound method> = bound_method %.loc9_24.2, %Op.specific_fn.16
// CHECK:STDOUT:   %addr.loc9_24.2: %ptr.3bd = addr_of %.loc9_24.2
// CHECK:STDOUT:   %no_op.loc9_24.2: init %empty_tuple.type = call %bound_method.loc9_24.2(%addr.loc9_24.2)
// CHECK:STDOUT:   %Op.bound.loc8_35.3: <bound method> = bound_method %.loc8_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.17: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc8_35.4: <bound method> = bound_method %.loc8_35.1, %Op.specific_fn.17
// CHECK:STDOUT:   %addr.loc8_35.3: %ptr.f29 = addr_of %.loc8_35.1
// CHECK:STDOUT:   %no_op.loc8_35.3: init %empty_tuple.type = call %bound_method.loc8_35.4(%addr.loc8_35.3)
// CHECK:STDOUT:   %Op.bound.loc8_24.3: <bound method> = bound_method %.loc8_24.2, constants.%Op.ad8
// CHECK:STDOUT:   %Op.specific_fn.18: <specific function> = specific_function constants.%Op.ad8, @Op.1(constants.%C.674) [concrete = constants.%Op.specific_fn.6b6]
// CHECK:STDOUT:   %bound_method.loc8_24.3: <bound method> = bound_method %.loc8_24.2, %Op.specific_fn.18
// CHECK:STDOUT:   %addr.loc8_24.3: %ptr.625 = addr_of %.loc8_24.2
// CHECK:STDOUT:   %no_op.loc8_24.3: init %empty_tuple.type = call %bound_method.loc8_24.3(%addr.loc8_24.3)
// CHECK:STDOUT:   %Op.bound.loc7_35.4: <bound method> = bound_method %.loc7_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.19: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc7_35.5: <bound method> = bound_method %.loc7_35.1, %Op.specific_fn.19
// CHECK:STDOUT:   %addr.loc7_35.4: %ptr.f29 = addr_of %.loc7_35.1
// CHECK:STDOUT:   %no_op.loc7_35.4: init %empty_tuple.type = call %bound_method.loc7_35.5(%addr.loc7_35.4)
// CHECK:STDOUT:   %Op.bound.loc7_24.4: <bound method> = bound_method %.loc7_24.2, constants.%Op.ae6
// CHECK:STDOUT:   %Op.specific_fn.20: <specific function> = specific_function constants.%Op.ae6, @Op.1(constants.%C.b00) [concrete = constants.%Op.specific_fn.e42]
// CHECK:STDOUT:   %bound_method.loc7_24.4: <bound method> = bound_method %.loc7_24.2, %Op.specific_fn.20
// CHECK:STDOUT:   %addr.loc7_24.4: %ptr.697 = addr_of %.loc7_24.2
// CHECK:STDOUT:   %no_op.loc7_24.4: init %empty_tuple.type = call %bound_method.loc7_24.4(%addr.loc7_24.4)
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
// CHECK:STDOUT:   %impl.elem0.loc11_34: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_34.1: <bound method> = bound_method %int_4, %impl.elem0.loc11_34 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_34, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_34.2: <bound method> = bound_method %int_4, %specific_fn.loc11 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %bound_method.loc11_34.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc11_34.1: %i32 = value_of_initializer %int.convert_checked.loc11 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc11_34.2: %i32 = converted %int_4, %.loc11_34.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %C.loc11: type = class_type @C, @C(constants.%int_4.940) [concrete = constants.%C.89d]
// CHECK:STDOUT:   %.loc11_24.2: ref %C.89d = temporary_storage
// CHECK:STDOUT:   %.loc11_24.3: init %C.89d = class_init (), %.loc11_24.2 [concrete = constants.%C.val.1dd]
// CHECK:STDOUT:   %.loc11_24.4: ref %C.89d = temporary %.loc11_24.2, %.loc11_24.3
// CHECK:STDOUT:   %.loc11_26.1: ref %C.89d = converted %.loc11_24.1, %.loc11_24.4
// CHECK:STDOUT:   %impl.elem0.loc11_35: %.26e = impl_witness_access constants.%ImplicitAs.impl_witness.16c, element0 [concrete = constants.%Convert.009]
// CHECK:STDOUT:   %bound_method.loc11_35.1: <bound method> = bound_method %.loc11_26.1, %impl.elem0.loc11_35
// CHECK:STDOUT:   %.loc11_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc11_26.2: %C.89d = bind_value %.loc11_26.1
// CHECK:STDOUT:   %Convert.call.loc11: init %D = call %bound_method.loc11_35.1(%.loc11_26.2) to %.loc11_35.1
// CHECK:STDOUT:   %.loc11_35.2: init %D = converted %.loc11_26.1, %Convert.call.loc11
// CHECK:STDOUT:   %Op.bound.loc11_35.1: <bound method> = bound_method %.loc11_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.21: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc11_35.2: <bound method> = bound_method %.loc11_35.1, %Op.specific_fn.21
// CHECK:STDOUT:   %addr.loc11_35.1: %ptr.f29 = addr_of %.loc11_35.1
// CHECK:STDOUT:   %no_op.loc11_35.1: init %empty_tuple.type = call %bound_method.loc11_35.2(%addr.loc11_35.1)
// CHECK:STDOUT:   %Op.bound.loc11_24.1: <bound method> = bound_method %.loc11_24.2, constants.%Op.6b4
// CHECK:STDOUT:   %Op.specific_fn.22: <specific function> = specific_function constants.%Op.6b4, @Op.1(constants.%C.89d) [concrete = constants.%Op.specific_fn.357]
// CHECK:STDOUT:   %bound_method.loc11_24.1: <bound method> = bound_method %.loc11_24.2, %Op.specific_fn.22
// CHECK:STDOUT:   %addr.loc11_24.1: %ptr.c28b = addr_of %.loc11_24.2
// CHECK:STDOUT:   %no_op.loc11_24.1: init %empty_tuple.type = call %bound_method.loc11_24.1(%addr.loc11_24.1)
// CHECK:STDOUT:   %Op.bound.loc10_35.2: <bound method> = bound_method %.loc10_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.23: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc10_35.3: <bound method> = bound_method %.loc10_35.1, %Op.specific_fn.23
// CHECK:STDOUT:   %addr.loc10_35.2: %ptr.f29 = addr_of %.loc10_35.1
// CHECK:STDOUT:   %no_op.loc10_35.2: init %empty_tuple.type = call %bound_method.loc10_35.3(%addr.loc10_35.2)
// CHECK:STDOUT:   %Op.bound.loc10_24.2: <bound method> = bound_method %.loc10_24.2, constants.%Op.9cf
// CHECK:STDOUT:   %Op.specific_fn.24: <specific function> = specific_function constants.%Op.9cf, @Op.1(constants.%C.7ac) [concrete = constants.%Op.specific_fn.694]
// CHECK:STDOUT:   %bound_method.loc10_24.2: <bound method> = bound_method %.loc10_24.2, %Op.specific_fn.24
// CHECK:STDOUT:   %addr.loc10_24.2: %ptr.2b1 = addr_of %.loc10_24.2
// CHECK:STDOUT:   %no_op.loc10_24.2: init %empty_tuple.type = call %bound_method.loc10_24.2(%addr.loc10_24.2)
// CHECK:STDOUT:   %Op.bound.loc9_35.3: <bound method> = bound_method %.loc9_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.25: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc9_35.4: <bound method> = bound_method %.loc9_35.1, %Op.specific_fn.25
// CHECK:STDOUT:   %addr.loc9_35.3: %ptr.f29 = addr_of %.loc9_35.1
// CHECK:STDOUT:   %no_op.loc9_35.3: init %empty_tuple.type = call %bound_method.loc9_35.4(%addr.loc9_35.3)
// CHECK:STDOUT:   %Op.bound.loc9_24.3: <bound method> = bound_method %.loc9_24.2, constants.%Op.404
// CHECK:STDOUT:   %Op.specific_fn.26: <specific function> = specific_function constants.%Op.404, @Op.1(constants.%C.681) [concrete = constants.%Op.specific_fn.822]
// CHECK:STDOUT:   %bound_method.loc9_24.3: <bound method> = bound_method %.loc9_24.2, %Op.specific_fn.26
// CHECK:STDOUT:   %addr.loc9_24.3: %ptr.3bd = addr_of %.loc9_24.2
// CHECK:STDOUT:   %no_op.loc9_24.3: init %empty_tuple.type = call %bound_method.loc9_24.3(%addr.loc9_24.3)
// CHECK:STDOUT:   %Op.bound.loc8_35.4: <bound method> = bound_method %.loc8_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.27: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc8_35.5: <bound method> = bound_method %.loc8_35.1, %Op.specific_fn.27
// CHECK:STDOUT:   %addr.loc8_35.4: %ptr.f29 = addr_of %.loc8_35.1
// CHECK:STDOUT:   %no_op.loc8_35.4: init %empty_tuple.type = call %bound_method.loc8_35.5(%addr.loc8_35.4)
// CHECK:STDOUT:   %Op.bound.loc8_24.4: <bound method> = bound_method %.loc8_24.2, constants.%Op.ad8
// CHECK:STDOUT:   %Op.specific_fn.28: <specific function> = specific_function constants.%Op.ad8, @Op.1(constants.%C.674) [concrete = constants.%Op.specific_fn.6b6]
// CHECK:STDOUT:   %bound_method.loc8_24.4: <bound method> = bound_method %.loc8_24.2, %Op.specific_fn.28
// CHECK:STDOUT:   %addr.loc8_24.4: %ptr.625 = addr_of %.loc8_24.2
// CHECK:STDOUT:   %no_op.loc8_24.4: init %empty_tuple.type = call %bound_method.loc8_24.4(%addr.loc8_24.4)
// CHECK:STDOUT:   %Op.bound.loc7_35.5: <bound method> = bound_method %.loc7_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.29: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc7_35.6: <bound method> = bound_method %.loc7_35.1, %Op.specific_fn.29
// CHECK:STDOUT:   %addr.loc7_35.5: %ptr.f29 = addr_of %.loc7_35.1
// CHECK:STDOUT:   %no_op.loc7_35.5: init %empty_tuple.type = call %bound_method.loc7_35.6(%addr.loc7_35.5)
// CHECK:STDOUT:   %Op.bound.loc7_24.5: <bound method> = bound_method %.loc7_24.2, constants.%Op.ae6
// CHECK:STDOUT:   %Op.specific_fn.30: <specific function> = specific_function constants.%Op.ae6, @Op.1(constants.%C.b00) [concrete = constants.%Op.specific_fn.e42]
// CHECK:STDOUT:   %bound_method.loc7_24.5: <bound method> = bound_method %.loc7_24.2, %Op.specific_fn.30
// CHECK:STDOUT:   %addr.loc7_24.5: %ptr.697 = addr_of %.loc7_24.2
// CHECK:STDOUT:   %no_op.loc7_24.5: init %empty_tuple.type = call %bound_method.loc7_24.5(%addr.loc7_24.5)
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
// CHECK:STDOUT:   %impl.elem0.loc12_34: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc12_34.1: <bound method> = bound_method %int_5, %impl.elem0.loc12_34 [concrete = constants.%Convert.bound.4e6]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_34, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_34.2: <bound method> = bound_method %int_5, %specific_fn.loc12 [concrete = constants.%bound_method.a25]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %bound_method.loc12_34.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc12_34.1: %i32 = value_of_initializer %int.convert_checked.loc12 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc12_34.2: %i32 = converted %int_5, %.loc12_34.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %C.loc12: type = class_type @C, @C(constants.%int_5.0f6) [concrete = constants.%C.f0a]
// CHECK:STDOUT:   %.loc12_24.2: ref %C.f0a = temporary_storage
// CHECK:STDOUT:   %.loc12_24.3: init %C.f0a = class_init (), %.loc12_24.2 [concrete = constants.%C.val.a4a]
// CHECK:STDOUT:   %.loc12_24.4: ref %C.f0a = temporary %.loc12_24.2, %.loc12_24.3
// CHECK:STDOUT:   %.loc12_26.1: ref %C.f0a = converted %.loc12_24.1, %.loc12_24.4
// CHECK:STDOUT:   %impl.elem0.loc12_35: %.eaf = impl_witness_access constants.%ImplicitAs.impl_witness.aa9, element0 [concrete = constants.%Convert.b66]
// CHECK:STDOUT:   %bound_method.loc12_35.1: <bound method> = bound_method %.loc12_26.1, %impl.elem0.loc12_35
// CHECK:STDOUT:   %.loc12_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc12_26.2: %C.f0a = bind_value %.loc12_26.1
// CHECK:STDOUT:   %Convert.call.loc12: init %D = call %bound_method.loc12_35.1(%.loc12_26.2) to %.loc12_35.1
// CHECK:STDOUT:   %.loc12_35.2: init %D = converted %.loc12_26.1, %Convert.call.loc12
// CHECK:STDOUT:   %Op.bound.loc12_35.1: <bound method> = bound_method %.loc12_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.31: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc12_35.2: <bound method> = bound_method %.loc12_35.1, %Op.specific_fn.31
// CHECK:STDOUT:   %addr.loc12_35.1: %ptr.f29 = addr_of %.loc12_35.1
// CHECK:STDOUT:   %no_op.loc12_35.1: init %empty_tuple.type = call %bound_method.loc12_35.2(%addr.loc12_35.1)
// CHECK:STDOUT:   %Op.bound.loc12_24.1: <bound method> = bound_method %.loc12_24.2, constants.%Op.985
// CHECK:STDOUT:   %Op.specific_fn.32: <specific function> = specific_function constants.%Op.985, @Op.1(constants.%C.f0a) [concrete = constants.%Op.specific_fn.a66]
// CHECK:STDOUT:   %bound_method.loc12_24.1: <bound method> = bound_method %.loc12_24.2, %Op.specific_fn.32
// CHECK:STDOUT:   %addr.loc12_24.1: %ptr.24c = addr_of %.loc12_24.2
// CHECK:STDOUT:   %no_op.loc12_24.1: init %empty_tuple.type = call %bound_method.loc12_24.1(%addr.loc12_24.1)
// CHECK:STDOUT:   %Op.bound.loc11_35.2: <bound method> = bound_method %.loc11_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.33: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc11_35.3: <bound method> = bound_method %.loc11_35.1, %Op.specific_fn.33
// CHECK:STDOUT:   %addr.loc11_35.2: %ptr.f29 = addr_of %.loc11_35.1
// CHECK:STDOUT:   %no_op.loc11_35.2: init %empty_tuple.type = call %bound_method.loc11_35.3(%addr.loc11_35.2)
// CHECK:STDOUT:   %Op.bound.loc11_24.2: <bound method> = bound_method %.loc11_24.2, constants.%Op.6b4
// CHECK:STDOUT:   %Op.specific_fn.34: <specific function> = specific_function constants.%Op.6b4, @Op.1(constants.%C.89d) [concrete = constants.%Op.specific_fn.357]
// CHECK:STDOUT:   %bound_method.loc11_24.2: <bound method> = bound_method %.loc11_24.2, %Op.specific_fn.34
// CHECK:STDOUT:   %addr.loc11_24.2: %ptr.c28b = addr_of %.loc11_24.2
// CHECK:STDOUT:   %no_op.loc11_24.2: init %empty_tuple.type = call %bound_method.loc11_24.2(%addr.loc11_24.2)
// CHECK:STDOUT:   %Op.bound.loc10_35.3: <bound method> = bound_method %.loc10_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.35: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc10_35.4: <bound method> = bound_method %.loc10_35.1, %Op.specific_fn.35
// CHECK:STDOUT:   %addr.loc10_35.3: %ptr.f29 = addr_of %.loc10_35.1
// CHECK:STDOUT:   %no_op.loc10_35.3: init %empty_tuple.type = call %bound_method.loc10_35.4(%addr.loc10_35.3)
// CHECK:STDOUT:   %Op.bound.loc10_24.3: <bound method> = bound_method %.loc10_24.2, constants.%Op.9cf
// CHECK:STDOUT:   %Op.specific_fn.36: <specific function> = specific_function constants.%Op.9cf, @Op.1(constants.%C.7ac) [concrete = constants.%Op.specific_fn.694]
// CHECK:STDOUT:   %bound_method.loc10_24.3: <bound method> = bound_method %.loc10_24.2, %Op.specific_fn.36
// CHECK:STDOUT:   %addr.loc10_24.3: %ptr.2b1 = addr_of %.loc10_24.2
// CHECK:STDOUT:   %no_op.loc10_24.3: init %empty_tuple.type = call %bound_method.loc10_24.3(%addr.loc10_24.3)
// CHECK:STDOUT:   %Op.bound.loc9_35.4: <bound method> = bound_method %.loc9_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.37: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc9_35.5: <bound method> = bound_method %.loc9_35.1, %Op.specific_fn.37
// CHECK:STDOUT:   %addr.loc9_35.4: %ptr.f29 = addr_of %.loc9_35.1
// CHECK:STDOUT:   %no_op.loc9_35.4: init %empty_tuple.type = call %bound_method.loc9_35.5(%addr.loc9_35.4)
// CHECK:STDOUT:   %Op.bound.loc9_24.4: <bound method> = bound_method %.loc9_24.2, constants.%Op.404
// CHECK:STDOUT:   %Op.specific_fn.38: <specific function> = specific_function constants.%Op.404, @Op.1(constants.%C.681) [concrete = constants.%Op.specific_fn.822]
// CHECK:STDOUT:   %bound_method.loc9_24.4: <bound method> = bound_method %.loc9_24.2, %Op.specific_fn.38
// CHECK:STDOUT:   %addr.loc9_24.4: %ptr.3bd = addr_of %.loc9_24.2
// CHECK:STDOUT:   %no_op.loc9_24.4: init %empty_tuple.type = call %bound_method.loc9_24.4(%addr.loc9_24.4)
// CHECK:STDOUT:   %Op.bound.loc8_35.5: <bound method> = bound_method %.loc8_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.39: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc8_35.6: <bound method> = bound_method %.loc8_35.1, %Op.specific_fn.39
// CHECK:STDOUT:   %addr.loc8_35.5: %ptr.f29 = addr_of %.loc8_35.1
// CHECK:STDOUT:   %no_op.loc8_35.5: init %empty_tuple.type = call %bound_method.loc8_35.6(%addr.loc8_35.5)
// CHECK:STDOUT:   %Op.bound.loc8_24.5: <bound method> = bound_method %.loc8_24.2, constants.%Op.ad8
// CHECK:STDOUT:   %Op.specific_fn.40: <specific function> = specific_function constants.%Op.ad8, @Op.1(constants.%C.674) [concrete = constants.%Op.specific_fn.6b6]
// CHECK:STDOUT:   %bound_method.loc8_24.5: <bound method> = bound_method %.loc8_24.2, %Op.specific_fn.40
// CHECK:STDOUT:   %addr.loc8_24.5: %ptr.625 = addr_of %.loc8_24.2
// CHECK:STDOUT:   %no_op.loc8_24.5: init %empty_tuple.type = call %bound_method.loc8_24.5(%addr.loc8_24.5)
// CHECK:STDOUT:   %Op.bound.loc7_35.6: <bound method> = bound_method %.loc7_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.41: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc7_35.7: <bound method> = bound_method %.loc7_35.1, %Op.specific_fn.41
// CHECK:STDOUT:   %addr.loc7_35.6: %ptr.f29 = addr_of %.loc7_35.1
// CHECK:STDOUT:   %no_op.loc7_35.6: init %empty_tuple.type = call %bound_method.loc7_35.7(%addr.loc7_35.6)
// CHECK:STDOUT:   %Op.bound.loc7_24.6: <bound method> = bound_method %.loc7_24.2, constants.%Op.ae6
// CHECK:STDOUT:   %Op.specific_fn.42: <specific function> = specific_function constants.%Op.ae6, @Op.1(constants.%C.b00) [concrete = constants.%Op.specific_fn.e42]
// CHECK:STDOUT:   %bound_method.loc7_24.6: <bound method> = bound_method %.loc7_24.2, %Op.specific_fn.42
// CHECK:STDOUT:   %addr.loc7_24.6: %ptr.697 = addr_of %.loc7_24.2
// CHECK:STDOUT:   %no_op.loc7_24.6: init %empty_tuple.type = call %bound_method.loc7_24.6(%addr.loc7_24.6)
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
// CHECK:STDOUT:   %impl.elem0.loc13_34: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_34.1: <bound method> = bound_method %int_6, %impl.elem0.loc13_34 [concrete = constants.%Convert.bound.ce9]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_34, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_34.2: <bound method> = bound_method %int_6, %specific_fn.loc13 [concrete = constants.%bound_method.efa]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32 = call %bound_method.loc13_34.2(%int_6) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc13_34.1: %i32 = value_of_initializer %int.convert_checked.loc13 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc13_34.2: %i32 = converted %int_6, %.loc13_34.1 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %C.loc13: type = class_type @C, @C(constants.%int_6.e56) [concrete = constants.%C.c60]
// CHECK:STDOUT:   %.loc13_24.2: ref %C.c60 = temporary_storage
// CHECK:STDOUT:   %.loc13_24.3: init %C.c60 = class_init (), %.loc13_24.2 [concrete = constants.%C.val.a4b]
// CHECK:STDOUT:   %.loc13_24.4: ref %C.c60 = temporary %.loc13_24.2, %.loc13_24.3
// CHECK:STDOUT:   %.loc13_26.1: ref %C.c60 = converted %.loc13_24.1, %.loc13_24.4
// CHECK:STDOUT:   %impl.elem0.loc13_35: %.874 = impl_witness_access constants.%ImplicitAs.impl_witness.e4d, element0 [concrete = constants.%Convert.f59]
// CHECK:STDOUT:   %bound_method.loc13_35.1: <bound method> = bound_method %.loc13_26.1, %impl.elem0.loc13_35
// CHECK:STDOUT:   %.loc13_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc13_26.2: %C.c60 = bind_value %.loc13_26.1
// CHECK:STDOUT:   %Convert.call.loc13: init %D = call %bound_method.loc13_35.1(%.loc13_26.2) to %.loc13_35.1
// CHECK:STDOUT:   %.loc13_35.2: init %D = converted %.loc13_26.1, %Convert.call.loc13
// CHECK:STDOUT:   %Op.bound.loc13_35.1: <bound method> = bound_method %.loc13_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.43: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc13_35.2: <bound method> = bound_method %.loc13_35.1, %Op.specific_fn.43
// CHECK:STDOUT:   %addr.loc13_35.1: %ptr.f29 = addr_of %.loc13_35.1
// CHECK:STDOUT:   %no_op.loc13_35.1: init %empty_tuple.type = call %bound_method.loc13_35.2(%addr.loc13_35.1)
// CHECK:STDOUT:   %Op.bound.loc13_24.1: <bound method> = bound_method %.loc13_24.2, constants.%Op.ca5
// CHECK:STDOUT:   %Op.specific_fn.44: <specific function> = specific_function constants.%Op.ca5, @Op.1(constants.%C.c60) [concrete = constants.%Op.specific_fn.8d0]
// CHECK:STDOUT:   %bound_method.loc13_24.1: <bound method> = bound_method %.loc13_24.2, %Op.specific_fn.44
// CHECK:STDOUT:   %addr.loc13_24.1: %ptr.b5e = addr_of %.loc13_24.2
// CHECK:STDOUT:   %no_op.loc13_24.1: init %empty_tuple.type = call %bound_method.loc13_24.1(%addr.loc13_24.1)
// CHECK:STDOUT:   %Op.bound.loc12_35.2: <bound method> = bound_method %.loc12_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.45: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc12_35.3: <bound method> = bound_method %.loc12_35.1, %Op.specific_fn.45
// CHECK:STDOUT:   %addr.loc12_35.2: %ptr.f29 = addr_of %.loc12_35.1
// CHECK:STDOUT:   %no_op.loc12_35.2: init %empty_tuple.type = call %bound_method.loc12_35.3(%addr.loc12_35.2)
// CHECK:STDOUT:   %Op.bound.loc12_24.2: <bound method> = bound_method %.loc12_24.2, constants.%Op.985
// CHECK:STDOUT:   %Op.specific_fn.46: <specific function> = specific_function constants.%Op.985, @Op.1(constants.%C.f0a) [concrete = constants.%Op.specific_fn.a66]
// CHECK:STDOUT:   %bound_method.loc12_24.2: <bound method> = bound_method %.loc12_24.2, %Op.specific_fn.46
// CHECK:STDOUT:   %addr.loc12_24.2: %ptr.24c = addr_of %.loc12_24.2
// CHECK:STDOUT:   %no_op.loc12_24.2: init %empty_tuple.type = call %bound_method.loc12_24.2(%addr.loc12_24.2)
// CHECK:STDOUT:   %Op.bound.loc11_35.3: <bound method> = bound_method %.loc11_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.47: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc11_35.4: <bound method> = bound_method %.loc11_35.1, %Op.specific_fn.47
// CHECK:STDOUT:   %addr.loc11_35.3: %ptr.f29 = addr_of %.loc11_35.1
// CHECK:STDOUT:   %no_op.loc11_35.3: init %empty_tuple.type = call %bound_method.loc11_35.4(%addr.loc11_35.3)
// CHECK:STDOUT:   %Op.bound.loc11_24.3: <bound method> = bound_method %.loc11_24.2, constants.%Op.6b4
// CHECK:STDOUT:   %Op.specific_fn.48: <specific function> = specific_function constants.%Op.6b4, @Op.1(constants.%C.89d) [concrete = constants.%Op.specific_fn.357]
// CHECK:STDOUT:   %bound_method.loc11_24.3: <bound method> = bound_method %.loc11_24.2, %Op.specific_fn.48
// CHECK:STDOUT:   %addr.loc11_24.3: %ptr.c28b = addr_of %.loc11_24.2
// CHECK:STDOUT:   %no_op.loc11_24.3: init %empty_tuple.type = call %bound_method.loc11_24.3(%addr.loc11_24.3)
// CHECK:STDOUT:   %Op.bound.loc10_35.4: <bound method> = bound_method %.loc10_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.49: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc10_35.5: <bound method> = bound_method %.loc10_35.1, %Op.specific_fn.49
// CHECK:STDOUT:   %addr.loc10_35.4: %ptr.f29 = addr_of %.loc10_35.1
// CHECK:STDOUT:   %no_op.loc10_35.4: init %empty_tuple.type = call %bound_method.loc10_35.5(%addr.loc10_35.4)
// CHECK:STDOUT:   %Op.bound.loc10_24.4: <bound method> = bound_method %.loc10_24.2, constants.%Op.9cf
// CHECK:STDOUT:   %Op.specific_fn.50: <specific function> = specific_function constants.%Op.9cf, @Op.1(constants.%C.7ac) [concrete = constants.%Op.specific_fn.694]
// CHECK:STDOUT:   %bound_method.loc10_24.4: <bound method> = bound_method %.loc10_24.2, %Op.specific_fn.50
// CHECK:STDOUT:   %addr.loc10_24.4: %ptr.2b1 = addr_of %.loc10_24.2
// CHECK:STDOUT:   %no_op.loc10_24.4: init %empty_tuple.type = call %bound_method.loc10_24.4(%addr.loc10_24.4)
// CHECK:STDOUT:   %Op.bound.loc9_35.5: <bound method> = bound_method %.loc9_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.51: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc9_35.6: <bound method> = bound_method %.loc9_35.1, %Op.specific_fn.51
// CHECK:STDOUT:   %addr.loc9_35.5: %ptr.f29 = addr_of %.loc9_35.1
// CHECK:STDOUT:   %no_op.loc9_35.5: init %empty_tuple.type = call %bound_method.loc9_35.6(%addr.loc9_35.5)
// CHECK:STDOUT:   %Op.bound.loc9_24.5: <bound method> = bound_method %.loc9_24.2, constants.%Op.404
// CHECK:STDOUT:   %Op.specific_fn.52: <specific function> = specific_function constants.%Op.404, @Op.1(constants.%C.681) [concrete = constants.%Op.specific_fn.822]
// CHECK:STDOUT:   %bound_method.loc9_24.5: <bound method> = bound_method %.loc9_24.2, %Op.specific_fn.52
// CHECK:STDOUT:   %addr.loc9_24.5: %ptr.3bd = addr_of %.loc9_24.2
// CHECK:STDOUT:   %no_op.loc9_24.5: init %empty_tuple.type = call %bound_method.loc9_24.5(%addr.loc9_24.5)
// CHECK:STDOUT:   %Op.bound.loc8_35.6: <bound method> = bound_method %.loc8_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.53: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc8_35.7: <bound method> = bound_method %.loc8_35.1, %Op.specific_fn.53
// CHECK:STDOUT:   %addr.loc8_35.6: %ptr.f29 = addr_of %.loc8_35.1
// CHECK:STDOUT:   %no_op.loc8_35.6: init %empty_tuple.type = call %bound_method.loc8_35.7(%addr.loc8_35.6)
// CHECK:STDOUT:   %Op.bound.loc8_24.6: <bound method> = bound_method %.loc8_24.2, constants.%Op.ad8
// CHECK:STDOUT:   %Op.specific_fn.54: <specific function> = specific_function constants.%Op.ad8, @Op.1(constants.%C.674) [concrete = constants.%Op.specific_fn.6b6]
// CHECK:STDOUT:   %bound_method.loc8_24.6: <bound method> = bound_method %.loc8_24.2, %Op.specific_fn.54
// CHECK:STDOUT:   %addr.loc8_24.6: %ptr.625 = addr_of %.loc8_24.2
// CHECK:STDOUT:   %no_op.loc8_24.6: init %empty_tuple.type = call %bound_method.loc8_24.6(%addr.loc8_24.6)
// CHECK:STDOUT:   %Op.bound.loc7_35.7: <bound method> = bound_method %.loc7_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.55: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc7_35.8: <bound method> = bound_method %.loc7_35.1, %Op.specific_fn.55
// CHECK:STDOUT:   %addr.loc7_35.7: %ptr.f29 = addr_of %.loc7_35.1
// CHECK:STDOUT:   %no_op.loc7_35.7: init %empty_tuple.type = call %bound_method.loc7_35.8(%addr.loc7_35.7)
// CHECK:STDOUT:   %Op.bound.loc7_24.7: <bound method> = bound_method %.loc7_24.2, constants.%Op.ae6
// CHECK:STDOUT:   %Op.specific_fn.56: <specific function> = specific_function constants.%Op.ae6, @Op.1(constants.%C.b00) [concrete = constants.%Op.specific_fn.e42]
// CHECK:STDOUT:   %bound_method.loc7_24.7: <bound method> = bound_method %.loc7_24.2, %Op.specific_fn.56
// CHECK:STDOUT:   %addr.loc7_24.7: %ptr.697 = addr_of %.loc7_24.2
// CHECK:STDOUT:   %no_op.loc7_24.7: init %empty_tuple.type = call %bound_method.loc7_24.7(%addr.loc7_24.7)
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
// CHECK:STDOUT:   %impl.elem0.loc14_34: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_34.1: <bound method> = bound_method %int_7, %impl.elem0.loc14_34 [concrete = constants.%Convert.bound.208]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_34, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.2: <bound method> = bound_method %int_7, %specific_fn.loc14 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %int.convert_checked.loc14: init %i32 = call %bound_method.loc14_34.2(%int_7) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc14_34.1: %i32 = value_of_initializer %int.convert_checked.loc14 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc14_34.2: %i32 = converted %int_7, %.loc14_34.1 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %C.loc14: type = class_type @C, @C(constants.%int_7.0b1) [concrete = constants.%C.304]
// CHECK:STDOUT:   %.loc14_24.2: ref %C.304 = temporary_storage
// CHECK:STDOUT:   %.loc14_24.3: init %C.304 = class_init (), %.loc14_24.2 [concrete = constants.%C.val.f9f]
// CHECK:STDOUT:   %.loc14_24.4: ref %C.304 = temporary %.loc14_24.2, %.loc14_24.3
// CHECK:STDOUT:   %.loc14_26.1: ref %C.304 = converted %.loc14_24.1, %.loc14_24.4
// CHECK:STDOUT:   %impl.elem0.loc14_35: %.79b = impl_witness_access constants.%ImplicitAs.impl_witness.0ca, element0 [concrete = constants.%Convert.6c1]
// CHECK:STDOUT:   %bound_method.loc14_35.1: <bound method> = bound_method %.loc14_26.1, %impl.elem0.loc14_35
// CHECK:STDOUT:   %.loc14_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc14_26.2: %C.304 = bind_value %.loc14_26.1
// CHECK:STDOUT:   %Convert.call.loc14: init %D = call %bound_method.loc14_35.1(%.loc14_26.2) to %.loc14_35.1
// CHECK:STDOUT:   %.loc14_35.2: init %D = converted %.loc14_26.1, %Convert.call.loc14
// CHECK:STDOUT:   %Op.bound.loc14_35.1: <bound method> = bound_method %.loc14_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.57: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc14_35.2: <bound method> = bound_method %.loc14_35.1, %Op.specific_fn.57
// CHECK:STDOUT:   %addr.loc14_35.1: %ptr.f29 = addr_of %.loc14_35.1
// CHECK:STDOUT:   %no_op.loc14_35.1: init %empty_tuple.type = call %bound_method.loc14_35.2(%addr.loc14_35.1)
// CHECK:STDOUT:   %Op.bound.loc14_24.1: <bound method> = bound_method %.loc14_24.2, constants.%Op.6fc
// CHECK:STDOUT:   %Op.specific_fn.58: <specific function> = specific_function constants.%Op.6fc, @Op.1(constants.%C.304) [concrete = constants.%Op.specific_fn.a08]
// CHECK:STDOUT:   %bound_method.loc14_24.1: <bound method> = bound_method %.loc14_24.2, %Op.specific_fn.58
// CHECK:STDOUT:   %addr.loc14_24.1: %ptr.dc3 = addr_of %.loc14_24.2
// CHECK:STDOUT:   %no_op.loc14_24.1: init %empty_tuple.type = call %bound_method.loc14_24.1(%addr.loc14_24.1)
// CHECK:STDOUT:   %Op.bound.loc13_35.2: <bound method> = bound_method %.loc13_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.59: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc13_35.3: <bound method> = bound_method %.loc13_35.1, %Op.specific_fn.59
// CHECK:STDOUT:   %addr.loc13_35.2: %ptr.f29 = addr_of %.loc13_35.1
// CHECK:STDOUT:   %no_op.loc13_35.2: init %empty_tuple.type = call %bound_method.loc13_35.3(%addr.loc13_35.2)
// CHECK:STDOUT:   %Op.bound.loc13_24.2: <bound method> = bound_method %.loc13_24.2, constants.%Op.ca5
// CHECK:STDOUT:   %Op.specific_fn.60: <specific function> = specific_function constants.%Op.ca5, @Op.1(constants.%C.c60) [concrete = constants.%Op.specific_fn.8d0]
// CHECK:STDOUT:   %bound_method.loc13_24.2: <bound method> = bound_method %.loc13_24.2, %Op.specific_fn.60
// CHECK:STDOUT:   %addr.loc13_24.2: %ptr.b5e = addr_of %.loc13_24.2
// CHECK:STDOUT:   %no_op.loc13_24.2: init %empty_tuple.type = call %bound_method.loc13_24.2(%addr.loc13_24.2)
// CHECK:STDOUT:   %Op.bound.loc12_35.3: <bound method> = bound_method %.loc12_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.61: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc12_35.4: <bound method> = bound_method %.loc12_35.1, %Op.specific_fn.61
// CHECK:STDOUT:   %addr.loc12_35.3: %ptr.f29 = addr_of %.loc12_35.1
// CHECK:STDOUT:   %no_op.loc12_35.3: init %empty_tuple.type = call %bound_method.loc12_35.4(%addr.loc12_35.3)
// CHECK:STDOUT:   %Op.bound.loc12_24.3: <bound method> = bound_method %.loc12_24.2, constants.%Op.985
// CHECK:STDOUT:   %Op.specific_fn.62: <specific function> = specific_function constants.%Op.985, @Op.1(constants.%C.f0a) [concrete = constants.%Op.specific_fn.a66]
// CHECK:STDOUT:   %bound_method.loc12_24.3: <bound method> = bound_method %.loc12_24.2, %Op.specific_fn.62
// CHECK:STDOUT:   %addr.loc12_24.3: %ptr.24c = addr_of %.loc12_24.2
// CHECK:STDOUT:   %no_op.loc12_24.3: init %empty_tuple.type = call %bound_method.loc12_24.3(%addr.loc12_24.3)
// CHECK:STDOUT:   %Op.bound.loc11_35.4: <bound method> = bound_method %.loc11_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.63: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc11_35.5: <bound method> = bound_method %.loc11_35.1, %Op.specific_fn.63
// CHECK:STDOUT:   %addr.loc11_35.4: %ptr.f29 = addr_of %.loc11_35.1
// CHECK:STDOUT:   %no_op.loc11_35.4: init %empty_tuple.type = call %bound_method.loc11_35.5(%addr.loc11_35.4)
// CHECK:STDOUT:   %Op.bound.loc11_24.4: <bound method> = bound_method %.loc11_24.2, constants.%Op.6b4
// CHECK:STDOUT:   %Op.specific_fn.64: <specific function> = specific_function constants.%Op.6b4, @Op.1(constants.%C.89d) [concrete = constants.%Op.specific_fn.357]
// CHECK:STDOUT:   %bound_method.loc11_24.4: <bound method> = bound_method %.loc11_24.2, %Op.specific_fn.64
// CHECK:STDOUT:   %addr.loc11_24.4: %ptr.c28b = addr_of %.loc11_24.2
// CHECK:STDOUT:   %no_op.loc11_24.4: init %empty_tuple.type = call %bound_method.loc11_24.4(%addr.loc11_24.4)
// CHECK:STDOUT:   %Op.bound.loc10_35.5: <bound method> = bound_method %.loc10_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.65: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc10_35.6: <bound method> = bound_method %.loc10_35.1, %Op.specific_fn.65
// CHECK:STDOUT:   %addr.loc10_35.5: %ptr.f29 = addr_of %.loc10_35.1
// CHECK:STDOUT:   %no_op.loc10_35.5: init %empty_tuple.type = call %bound_method.loc10_35.6(%addr.loc10_35.5)
// CHECK:STDOUT:   %Op.bound.loc10_24.5: <bound method> = bound_method %.loc10_24.2, constants.%Op.9cf
// CHECK:STDOUT:   %Op.specific_fn.66: <specific function> = specific_function constants.%Op.9cf, @Op.1(constants.%C.7ac) [concrete = constants.%Op.specific_fn.694]
// CHECK:STDOUT:   %bound_method.loc10_24.5: <bound method> = bound_method %.loc10_24.2, %Op.specific_fn.66
// CHECK:STDOUT:   %addr.loc10_24.5: %ptr.2b1 = addr_of %.loc10_24.2
// CHECK:STDOUT:   %no_op.loc10_24.5: init %empty_tuple.type = call %bound_method.loc10_24.5(%addr.loc10_24.5)
// CHECK:STDOUT:   %Op.bound.loc9_35.6: <bound method> = bound_method %.loc9_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.67: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc9_35.7: <bound method> = bound_method %.loc9_35.1, %Op.specific_fn.67
// CHECK:STDOUT:   %addr.loc9_35.6: %ptr.f29 = addr_of %.loc9_35.1
// CHECK:STDOUT:   %no_op.loc9_35.6: init %empty_tuple.type = call %bound_method.loc9_35.7(%addr.loc9_35.6)
// CHECK:STDOUT:   %Op.bound.loc9_24.6: <bound method> = bound_method %.loc9_24.2, constants.%Op.404
// CHECK:STDOUT:   %Op.specific_fn.68: <specific function> = specific_function constants.%Op.404, @Op.1(constants.%C.681) [concrete = constants.%Op.specific_fn.822]
// CHECK:STDOUT:   %bound_method.loc9_24.6: <bound method> = bound_method %.loc9_24.2, %Op.specific_fn.68
// CHECK:STDOUT:   %addr.loc9_24.6: %ptr.3bd = addr_of %.loc9_24.2
// CHECK:STDOUT:   %no_op.loc9_24.6: init %empty_tuple.type = call %bound_method.loc9_24.6(%addr.loc9_24.6)
// CHECK:STDOUT:   %Op.bound.loc8_35.7: <bound method> = bound_method %.loc8_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.69: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc8_35.8: <bound method> = bound_method %.loc8_35.1, %Op.specific_fn.69
// CHECK:STDOUT:   %addr.loc8_35.7: %ptr.f29 = addr_of %.loc8_35.1
// CHECK:STDOUT:   %no_op.loc8_35.7: init %empty_tuple.type = call %bound_method.loc8_35.8(%addr.loc8_35.7)
// CHECK:STDOUT:   %Op.bound.loc8_24.7: <bound method> = bound_method %.loc8_24.2, constants.%Op.ad8
// CHECK:STDOUT:   %Op.specific_fn.70: <specific function> = specific_function constants.%Op.ad8, @Op.1(constants.%C.674) [concrete = constants.%Op.specific_fn.6b6]
// CHECK:STDOUT:   %bound_method.loc8_24.7: <bound method> = bound_method %.loc8_24.2, %Op.specific_fn.70
// CHECK:STDOUT:   %addr.loc8_24.7: %ptr.625 = addr_of %.loc8_24.2
// CHECK:STDOUT:   %no_op.loc8_24.7: init %empty_tuple.type = call %bound_method.loc8_24.7(%addr.loc8_24.7)
// CHECK:STDOUT:   %Op.bound.loc7_35.8: <bound method> = bound_method %.loc7_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.71: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc7_35.9: <bound method> = bound_method %.loc7_35.1, %Op.specific_fn.71
// CHECK:STDOUT:   %addr.loc7_35.8: %ptr.f29 = addr_of %.loc7_35.1
// CHECK:STDOUT:   %no_op.loc7_35.8: init %empty_tuple.type = call %bound_method.loc7_35.9(%addr.loc7_35.8)
// CHECK:STDOUT:   %Op.bound.loc7_24.8: <bound method> = bound_method %.loc7_24.2, constants.%Op.ae6
// CHECK:STDOUT:   %Op.specific_fn.72: <specific function> = specific_function constants.%Op.ae6, @Op.1(constants.%C.b00) [concrete = constants.%Op.specific_fn.e42]
// CHECK:STDOUT:   %bound_method.loc7_24.8: <bound method> = bound_method %.loc7_24.2, %Op.specific_fn.72
// CHECK:STDOUT:   %addr.loc7_24.8: %ptr.697 = addr_of %.loc7_24.2
// CHECK:STDOUT:   %no_op.loc7_24.8: init %empty_tuple.type = call %bound_method.loc7_24.8(%addr.loc7_24.8)
// CHECK:STDOUT:   return %.loc14_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc14:
// CHECK:STDOUT:   %P.ref.loc15: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%P.Make [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc6_15: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc6_15
// CHECK:STDOUT:   %Op.bound.loc6: <bound method> = bound_method %.loc6_15, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.73: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc6: <bound method> = bound_method %.loc6_15, %Op.specific_fn.73
// CHECK:STDOUT:   %addr.loc6: %ptr.f29 = addr_of %.loc6_15
// CHECK:STDOUT:   %no_op.loc6: init %empty_tuple.type = call %bound_method.loc6(%addr.loc6)
// CHECK:STDOUT:   %Op.bound.loc14_35.2: <bound method> = bound_method %.loc14_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.74: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc14_35.3: <bound method> = bound_method %.loc14_35.1, %Op.specific_fn.74
// CHECK:STDOUT:   %addr.loc14_35.2: %ptr.f29 = addr_of %.loc14_35.1
// CHECK:STDOUT:   %no_op.loc14_35.2: init %empty_tuple.type = call %bound_method.loc14_35.3(%addr.loc14_35.2)
// CHECK:STDOUT:   %Op.bound.loc14_24.2: <bound method> = bound_method %.loc14_24.2, constants.%Op.6fc
// CHECK:STDOUT:   %Op.specific_fn.75: <specific function> = specific_function constants.%Op.6fc, @Op.1(constants.%C.304) [concrete = constants.%Op.specific_fn.a08]
// CHECK:STDOUT:   %bound_method.loc14_24.2: <bound method> = bound_method %.loc14_24.2, %Op.specific_fn.75
// CHECK:STDOUT:   %addr.loc14_24.2: %ptr.dc3 = addr_of %.loc14_24.2
// CHECK:STDOUT:   %no_op.loc14_24.2: init %empty_tuple.type = call %bound_method.loc14_24.2(%addr.loc14_24.2)
// CHECK:STDOUT:   %Op.bound.loc13_35.3: <bound method> = bound_method %.loc13_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.76: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc13_35.4: <bound method> = bound_method %.loc13_35.1, %Op.specific_fn.76
// CHECK:STDOUT:   %addr.loc13_35.3: %ptr.f29 = addr_of %.loc13_35.1
// CHECK:STDOUT:   %no_op.loc13_35.3: init %empty_tuple.type = call %bound_method.loc13_35.4(%addr.loc13_35.3)
// CHECK:STDOUT:   %Op.bound.loc13_24.3: <bound method> = bound_method %.loc13_24.2, constants.%Op.ca5
// CHECK:STDOUT:   %Op.specific_fn.77: <specific function> = specific_function constants.%Op.ca5, @Op.1(constants.%C.c60) [concrete = constants.%Op.specific_fn.8d0]
// CHECK:STDOUT:   %bound_method.loc13_24.3: <bound method> = bound_method %.loc13_24.2, %Op.specific_fn.77
// CHECK:STDOUT:   %addr.loc13_24.3: %ptr.b5e = addr_of %.loc13_24.2
// CHECK:STDOUT:   %no_op.loc13_24.3: init %empty_tuple.type = call %bound_method.loc13_24.3(%addr.loc13_24.3)
// CHECK:STDOUT:   %Op.bound.loc12_35.4: <bound method> = bound_method %.loc12_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.78: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc12_35.5: <bound method> = bound_method %.loc12_35.1, %Op.specific_fn.78
// CHECK:STDOUT:   %addr.loc12_35.4: %ptr.f29 = addr_of %.loc12_35.1
// CHECK:STDOUT:   %no_op.loc12_35.4: init %empty_tuple.type = call %bound_method.loc12_35.5(%addr.loc12_35.4)
// CHECK:STDOUT:   %Op.bound.loc12_24.4: <bound method> = bound_method %.loc12_24.2, constants.%Op.985
// CHECK:STDOUT:   %Op.specific_fn.79: <specific function> = specific_function constants.%Op.985, @Op.1(constants.%C.f0a) [concrete = constants.%Op.specific_fn.a66]
// CHECK:STDOUT:   %bound_method.loc12_24.4: <bound method> = bound_method %.loc12_24.2, %Op.specific_fn.79
// CHECK:STDOUT:   %addr.loc12_24.4: %ptr.24c = addr_of %.loc12_24.2
// CHECK:STDOUT:   %no_op.loc12_24.4: init %empty_tuple.type = call %bound_method.loc12_24.4(%addr.loc12_24.4)
// CHECK:STDOUT:   %Op.bound.loc11_35.5: <bound method> = bound_method %.loc11_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.80: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc11_35.6: <bound method> = bound_method %.loc11_35.1, %Op.specific_fn.80
// CHECK:STDOUT:   %addr.loc11_35.5: %ptr.f29 = addr_of %.loc11_35.1
// CHECK:STDOUT:   %no_op.loc11_35.5: init %empty_tuple.type = call %bound_method.loc11_35.6(%addr.loc11_35.5)
// CHECK:STDOUT:   %Op.bound.loc11_24.5: <bound method> = bound_method %.loc11_24.2, constants.%Op.6b4
// CHECK:STDOUT:   %Op.specific_fn.81: <specific function> = specific_function constants.%Op.6b4, @Op.1(constants.%C.89d) [concrete = constants.%Op.specific_fn.357]
// CHECK:STDOUT:   %bound_method.loc11_24.5: <bound method> = bound_method %.loc11_24.2, %Op.specific_fn.81
// CHECK:STDOUT:   %addr.loc11_24.5: %ptr.c28b = addr_of %.loc11_24.2
// CHECK:STDOUT:   %no_op.loc11_24.5: init %empty_tuple.type = call %bound_method.loc11_24.5(%addr.loc11_24.5)
// CHECK:STDOUT:   %Op.bound.loc10_35.6: <bound method> = bound_method %.loc10_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.82: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc10_35.7: <bound method> = bound_method %.loc10_35.1, %Op.specific_fn.82
// CHECK:STDOUT:   %addr.loc10_35.6: %ptr.f29 = addr_of %.loc10_35.1
// CHECK:STDOUT:   %no_op.loc10_35.6: init %empty_tuple.type = call %bound_method.loc10_35.7(%addr.loc10_35.6)
// CHECK:STDOUT:   %Op.bound.loc10_24.6: <bound method> = bound_method %.loc10_24.2, constants.%Op.9cf
// CHECK:STDOUT:   %Op.specific_fn.83: <specific function> = specific_function constants.%Op.9cf, @Op.1(constants.%C.7ac) [concrete = constants.%Op.specific_fn.694]
// CHECK:STDOUT:   %bound_method.loc10_24.6: <bound method> = bound_method %.loc10_24.2, %Op.specific_fn.83
// CHECK:STDOUT:   %addr.loc10_24.6: %ptr.2b1 = addr_of %.loc10_24.2
// CHECK:STDOUT:   %no_op.loc10_24.6: init %empty_tuple.type = call %bound_method.loc10_24.6(%addr.loc10_24.6)
// CHECK:STDOUT:   %Op.bound.loc9_35.7: <bound method> = bound_method %.loc9_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.84: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc9_35.8: <bound method> = bound_method %.loc9_35.1, %Op.specific_fn.84
// CHECK:STDOUT:   %addr.loc9_35.7: %ptr.f29 = addr_of %.loc9_35.1
// CHECK:STDOUT:   %no_op.loc9_35.7: init %empty_tuple.type = call %bound_method.loc9_35.8(%addr.loc9_35.7)
// CHECK:STDOUT:   %Op.bound.loc9_24.7: <bound method> = bound_method %.loc9_24.2, constants.%Op.404
// CHECK:STDOUT:   %Op.specific_fn.85: <specific function> = specific_function constants.%Op.404, @Op.1(constants.%C.681) [concrete = constants.%Op.specific_fn.822]
// CHECK:STDOUT:   %bound_method.loc9_24.7: <bound method> = bound_method %.loc9_24.2, %Op.specific_fn.85
// CHECK:STDOUT:   %addr.loc9_24.7: %ptr.3bd = addr_of %.loc9_24.2
// CHECK:STDOUT:   %no_op.loc9_24.7: init %empty_tuple.type = call %bound_method.loc9_24.7(%addr.loc9_24.7)
// CHECK:STDOUT:   %Op.bound.loc8_35.8: <bound method> = bound_method %.loc8_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.86: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc8_35.9: <bound method> = bound_method %.loc8_35.1, %Op.specific_fn.86
// CHECK:STDOUT:   %addr.loc8_35.8: %ptr.f29 = addr_of %.loc8_35.1
// CHECK:STDOUT:   %no_op.loc8_35.8: init %empty_tuple.type = call %bound_method.loc8_35.9(%addr.loc8_35.8)
// CHECK:STDOUT:   %Op.bound.loc8_24.8: <bound method> = bound_method %.loc8_24.2, constants.%Op.ad8
// CHECK:STDOUT:   %Op.specific_fn.87: <specific function> = specific_function constants.%Op.ad8, @Op.1(constants.%C.674) [concrete = constants.%Op.specific_fn.6b6]
// CHECK:STDOUT:   %bound_method.loc8_24.8: <bound method> = bound_method %.loc8_24.2, %Op.specific_fn.87
// CHECK:STDOUT:   %addr.loc8_24.8: %ptr.625 = addr_of %.loc8_24.2
// CHECK:STDOUT:   %no_op.loc8_24.8: init %empty_tuple.type = call %bound_method.loc8_24.8(%addr.loc8_24.8)
// CHECK:STDOUT:   %Op.bound.loc7_35.9: <bound method> = bound_method %.loc7_35.1, constants.%Op.f5a
// CHECK:STDOUT:   %Op.specific_fn.88: <specific function> = specific_function constants.%Op.f5a, @Op.1(constants.%D) [concrete = constants.%Op.specific_fn.bd1]
// CHECK:STDOUT:   %bound_method.loc7_35.10: <bound method> = bound_method %.loc7_35.1, %Op.specific_fn.88
// CHECK:STDOUT:   %addr.loc7_35.9: %ptr.f29 = addr_of %.loc7_35.1
// CHECK:STDOUT:   %no_op.loc7_35.9: init %empty_tuple.type = call %bound_method.loc7_35.10(%addr.loc7_35.9)
// CHECK:STDOUT:   %Op.bound.loc7_24.9: <bound method> = bound_method %.loc7_24.2, constants.%Op.ae6
// CHECK:STDOUT:   %Op.specific_fn.89: <specific function> = specific_function constants.%Op.ae6, @Op.1(constants.%C.b00) [concrete = constants.%Op.specific_fn.e42]
// CHECK:STDOUT:   %bound_method.loc7_24.9: <bound method> = bound_method %.loc7_24.2, %Op.specific_fn.89
// CHECK:STDOUT:   %addr.loc7_24.9: %ptr.697 = addr_of %.loc7_24.2
// CHECK:STDOUT:   %no_op.loc7_24.9: init %empty_tuple.type = call %bound_method.loc7_24.9(%addr.loc7_24.9)
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.2 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.3 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.4 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.5 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.6 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.7 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.8 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.9 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%N.51e) {
// CHECK:STDOUT:   %N => constants.%N.51e
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
