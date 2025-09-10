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
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.506: type = class_type @C, @C(%N.51e) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.2a5: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%N.51e) [symbolic]
// CHECK:STDOUT:   %ptr.128: type = ptr_type %C.506 [symbolic]
// CHECK:STDOUT:   %pattern_type.d64: type = pattern_type %ptr.128 [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%N.51e) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.05d: <witness> = impl_witness @D.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.a94: type = pattern_type %ptr.19c [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type: type = fn_type @D.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op: %D.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.n.m.de8: type = struct_type {.n: %i32, .m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ea0: <witness> = complete_type_witness %struct_type.n.m.de8 [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %struct_type.n.m.819: type = struct_type {.n: Core.IntLiteral, .m: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f01: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.acc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6b, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.592: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e4d: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.acc) [concrete]
// CHECK:STDOUT:   %.7ea: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.e4d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7cb: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.592, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.698: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value (%int_0.6a9, %int_0.6a9) [concrete]
// CHECK:STDOUT:   %C.c8d: type = class_type @C, @C(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2a2: type = facet_type <@ImplicitAs, @ImplicitAs(%D)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.998: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc8 [concrete]
// CHECK:STDOUT:   %pattern_type.f33: type = pattern_type %C.c8d [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.12d: type = fn_type @C.as.ImplicitAs.impl.Convert.loc8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.7cf: %C.as.ImplicitAs.impl.Convert.type.12d = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a02: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.b59: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.c02: type = class_type @C, @C(%int_1.5d2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.75b: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc9 [concrete]
// CHECK:STDOUT:   %pattern_type.07c: type = pattern_type %C.c02 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.f7e: type = fn_type @C.as.ImplicitAs.impl.Convert.loc9 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.e3d: %C.as.ImplicitAs.impl.Convert.type.f7e = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2d9: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.f6f: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %C.70b: type = class_type @C, @C(%int_2.ef8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.67d: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc10 [concrete]
// CHECK:STDOUT:   %pattern_type.538: type = pattern_type %C.70b [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.296: type = fn_type @C.as.ImplicitAs.impl.Convert.loc10 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.983: %C.as.ImplicitAs.impl.Convert.type.296 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c36: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.f79: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.8cb: type = class_type @C, @C(%int_3.822) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.876: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc11 [concrete]
// CHECK:STDOUT:   %pattern_type.232: type = pattern_type %C.8cb [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.757: type = fn_type @C.as.ImplicitAs.impl.Convert.loc11 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.81e: %C.as.ImplicitAs.impl.Convert.type.757 = struct_value () [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.121: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.564: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %C.ba2: type = class_type @C, @C(%int_4.940) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.8ed: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc12 [concrete]
// CHECK:STDOUT:   %pattern_type.fe0: type = pattern_type %C.ba2 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.19e: type = fn_type @C.as.ImplicitAs.impl.Convert.loc12 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.dc6: %C.as.ImplicitAs.impl.Convert.type.19e = struct_value () [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.6c8: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.918: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %C.fcd: type = class_type @C, @C(%int_5.0f6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.aae: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc13 [concrete]
// CHECK:STDOUT:   %pattern_type.e96: type = pattern_type %C.fcd [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.d0c: type = fn_type @C.as.ImplicitAs.impl.Convert.loc13 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.a11: %C.as.ImplicitAs.impl.Convert.type.d0c = struct_value () [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7a5: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.351: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.e56: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %C.675: type = class_type @C, @C(%int_6.e56) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.574: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc14 [concrete]
// CHECK:STDOUT:   %pattern_type.bc0: type = pattern_type %C.675 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.aa0: type = fn_type @C.as.ImplicitAs.impl.Convert.loc14 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.7d6: %C.as.ImplicitAs.impl.Convert.type.aa0 = struct_value () [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.fa2: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.c8a: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.0b1: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %C.884: type = class_type @C, @C(%int_7.0b1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e74: <witness> = impl_witness file.%ImplicitAs.impl_witness_table.loc15 [concrete]
// CHECK:STDOUT:   %pattern_type.c15: type = pattern_type %C.884 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.07a: type = fn_type @C.as.ImplicitAs.impl.Convert.loc15 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.228: %C.as.ImplicitAs.impl.Convert.type.07a = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.428: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f01)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b6b = impl_witness_table (%Core.import_ref.428), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
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
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.f1d [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %impl.elem0: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:     %bound_method.loc8_9.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7cb]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc8_9.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.698]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_9.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc8_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc8_9.2: %i32 = converted %int_0, %.loc8_9.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_0.6a9) [concrete = constants.%C.c8d]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.2a2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc8 = impl_witness_table (@C.as.ImplicitAs.impl.f1d.%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.f1d [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc8: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc8 [concrete = constants.%ImplicitAs.impl_witness.998]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.401 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %impl.elem0: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:     %bound_method.loc9_9.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a02]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc9_9.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method.b59]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_9.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc9_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %.loc9_9.2: %i32 = converted %int_1, %.loc9_9.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_1.5d2) [concrete = constants.%C.c02]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.2a2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc9 = impl_witness_table (@C.as.ImplicitAs.impl.401.%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.401 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc9: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc9 [concrete = constants.%ImplicitAs.impl_witness.75b]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.131 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:     %bound_method.loc10_9.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2d9]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_9.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method.f6f]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc10_9.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc10_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc10_9.2: %i32 = converted %int_2, %.loc10_9.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_2.ef8) [concrete = constants.%C.70b]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.2a2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc10 = impl_witness_table (@C.as.ImplicitAs.impl.131.%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.131 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc10: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc10 [concrete = constants.%ImplicitAs.impl_witness.67d]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.3b7 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %impl.elem0: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:     %bound_method.loc11_9.1: <bound method> = bound_method %int_3, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c36]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc11_9.2: <bound method> = bound_method %int_3, %specific_fn [concrete = constants.%bound_method.f79]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_9.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc11_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %.loc11_9.2: %i32 = converted %int_3, %.loc11_9.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_3.822) [concrete = constants.%C.8cb]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.2a2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc11 = impl_witness_table (@C.as.ImplicitAs.impl.3b7.%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.3b7 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc11: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc11 [concrete = constants.%ImplicitAs.impl_witness.876]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.5ce [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:     %impl.elem0: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:     %bound_method.loc12_9.1: <bound method> = bound_method %int_4, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.121]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc12_9.2: <bound method> = bound_method %int_4, %specific_fn [concrete = constants.%bound_method.564]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_9.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %.loc12_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %.loc12_9.2: %i32 = converted %int_4, %.loc12_9.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_4.940) [concrete = constants.%C.ba2]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.2a2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc12 = impl_witness_table (@C.as.ImplicitAs.impl.5ce.%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.5ce [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc12: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc12 [concrete = constants.%ImplicitAs.impl_witness.8ed]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.193 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:     %impl.elem0: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:     %bound_method.loc13_9.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.6c8]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc13_9.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method.918]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_9.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %.loc13_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %.loc13_9.2: %i32 = converted %int_5, %.loc13_9.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_5.0f6) [concrete = constants.%C.fcd]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.2a2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc13 = impl_witness_table (@C.as.ImplicitAs.impl.193.%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.193 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc13: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc13 [concrete = constants.%ImplicitAs.impl_witness.aae]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.b1b [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:     %impl.elem0: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:     %bound_method.loc14_9.1: <bound method> = bound_method %int_6, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7a5]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc14_9.2: <bound method> = bound_method %int_6, %specific_fn [concrete = constants.%bound_method.351]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc14_9.2(%int_6) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:     %.loc14_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_6.e56]
// CHECK:STDOUT:     %.loc14_9.2: %i32 = converted %int_6, %.loc14_9.1 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_6.e56) [concrete = constants.%C.675]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.2a2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc14 = impl_witness_table (@C.as.ImplicitAs.impl.b1b.%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.b1b [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc14: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc14 [concrete = constants.%ImplicitAs.impl_witness.574]
// CHECK:STDOUT:   impl_decl @C.as.ImplicitAs.impl.c87 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:     %impl.elem0: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:     %bound_method.loc15_9.1: <bound method> = bound_method %int_7, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.fa2]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc15_9.2: <bound method> = bound_method %int_7, %specific_fn [concrete = constants.%bound_method.c8a]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_9.2(%int_7) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:     %.loc15_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:     %.loc15_9.2: %i32 = converted %int_7, %.loc15_9.1 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%int_7.0b1) [concrete = constants.%C.884]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%D)> [concrete = constants.%ImplicitAs.type.2a2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.loc15 = impl_witness_table (@C.as.ImplicitAs.impl.c87.%C.as.ImplicitAs.impl.Convert.decl), @C.as.ImplicitAs.impl.c87 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.loc15: <witness> = impl_witness %ImplicitAs.impl_witness_table.loc15 [concrete = constants.%ImplicitAs.impl_witness.e74]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(@C.%N.loc4_9.2: %i32) {
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.51e)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%N) [symbolic = %C (constants.%C.506)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%N) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.2a5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%N) [symbolic = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = struct_value () [symbolic = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %C.as.Destroy.impl.Op.decl: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type) = fn_decl @C.as.Destroy.impl.Op [symbolic = @C.as.Destroy.impl.%C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.d64) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.d64) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_18.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.128) = value_param call_param0
// CHECK:STDOUT:       %.loc4_18.2: type = splice_block %Self.ref [symbolic = %C (constants.%C.506)] {
// CHECK:STDOUT:         %.loc4_18.3: type = specific_constant constants.%C.506, @C(constants.%N.51e) [symbolic = %C (constants.%C.506)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_18.3 [symbolic = %C (constants.%C.506)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Destroy.impl.Op.%ptr (%ptr.128) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @C.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.Destroy.impl: @D.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.decl: %D.as.Destroy.impl.Op.type = fn_decl @D.as.Destroy.impl.Op [concrete = constants.%D.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a94 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a94 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc5: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.19c = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:     %self: %ptr.19c = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %D.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @D.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.f1d: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.12d = fn_decl @C.as.ImplicitAs.impl.Convert.loc8 [concrete = constants.%C.as.ImplicitAs.impl.Convert.7cf] {
// CHECK:STDOUT:     %self.patt: %pattern_type.f33 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f33 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.c8d = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.f1d.%C [concrete = constants.%C.c8d]
// CHECK:STDOUT:     %self: %C.c8d = bind_name self, %self.param
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
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.401: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.f7e = fn_decl @C.as.ImplicitAs.impl.Convert.loc9 [concrete = constants.%C.as.ImplicitAs.impl.Convert.e3d] {
// CHECK:STDOUT:     %self.patt: %pattern_type.07c = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.07c = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.c02 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.401.%C [concrete = constants.%C.c02]
// CHECK:STDOUT:     %self: %C.c02 = bind_name self, %self.param
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
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.131: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.296 = fn_decl @C.as.ImplicitAs.impl.Convert.loc10 [concrete = constants.%C.as.ImplicitAs.impl.Convert.983] {
// CHECK:STDOUT:     %self.patt: %pattern_type.538 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.538 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.70b = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.131.%C [concrete = constants.%C.70b]
// CHECK:STDOUT:     %self: %C.70b = bind_name self, %self.param
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
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.3b7: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.757 = fn_decl @C.as.ImplicitAs.impl.Convert.loc11 [concrete = constants.%C.as.ImplicitAs.impl.Convert.81e] {
// CHECK:STDOUT:     %self.patt: %pattern_type.232 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.232 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.8cb = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.3b7.%C [concrete = constants.%C.8cb]
// CHECK:STDOUT:     %self: %C.8cb = bind_name self, %self.param
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
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.5ce: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.19e = fn_decl @C.as.ImplicitAs.impl.Convert.loc12 [concrete = constants.%C.as.ImplicitAs.impl.Convert.dc6] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fe0 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fe0 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.ba2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.5ce.%C [concrete = constants.%C.ba2]
// CHECK:STDOUT:     %self: %C.ba2 = bind_name self, %self.param
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
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.193: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.d0c = fn_decl @C.as.ImplicitAs.impl.Convert.loc13 [concrete = constants.%C.as.ImplicitAs.impl.Convert.a11] {
// CHECK:STDOUT:     %self.patt: %pattern_type.e96 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e96 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.fcd = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.193.%C [concrete = constants.%C.fcd]
// CHECK:STDOUT:     %self: %C.fcd = bind_name self, %self.param
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
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.b1b: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.aa0 = fn_decl @C.as.ImplicitAs.impl.Convert.loc14 [concrete = constants.%C.as.ImplicitAs.impl.Convert.7d6] {
// CHECK:STDOUT:     %self.patt: %pattern_type.bc0 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.bc0 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.675 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.b1b.%C [concrete = constants.%C.675]
// CHECK:STDOUT:     %self: %C.675 = bind_name self, %self.param
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
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.c87: %C as %ImplicitAs.type {
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.decl: %C.as.ImplicitAs.impl.Convert.type.07a = fn_decl @C.as.ImplicitAs.impl.Convert.loc15 [concrete = constants.%C.as.ImplicitAs.impl.Convert.228] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c15 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c15 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self.param: %C.884 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @C.as.ImplicitAs.impl.c87.%C [concrete = constants.%C.884]
// CHECK:STDOUT:     %self: %C.884 = bind_name self, %self.param
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
// CHECK:STDOUT:   %N.loc4_9.1: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_9.1 (constants.%N.51e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C.506 [symbolic = @C.as.Destroy.impl.%C (constants.%C.506)]
// CHECK:STDOUT:     impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @C.as.Destroy.impl(constants.%N.51e) [symbolic = @C.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.2a5)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:   impl_decl @D.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@D.as.Destroy.impl.%D.as.Destroy.impl.Op.decl), @D.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.05d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n.m.de8 [concrete = constants.%complete_type.ea0]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .n = %.loc5_16
// CHECK:STDOUT:   .m = %.loc5_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(@C.%N.loc4_9.2: %i32) {
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.51e)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%N) [symbolic = %C (constants.%C.506)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [symbolic = %ptr (constants.%ptr.128)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.d64)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.128)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.Destroy.impl.Op(%self.param: %ptr.19c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc6_31: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_0.loc6_39: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc6_40.1: %struct_type.n.m.819 = struct_literal (%int_0.loc6_31, %int_0.loc6_39)
// CHECK:STDOUT:   %impl.elem0.loc6_40.1: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc6_40.1: <bound method> = bound_method %int_0.loc6_31, %impl.elem0.loc6_40.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7cb]
// CHECK:STDOUT:   %specific_fn.loc6_40.1: <specific function> = specific_function %impl.elem0.loc6_40.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.2: <bound method> = bound_method %int_0.loc6_31, %specific_fn.loc6_40.1 [concrete = constants.%bound_method.698]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.1: init %i32 = call %bound_method.loc6_40.2(%int_0.loc6_31) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.2: init %i32 = converted %int_0.loc6_31, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_40.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_40.3: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc6_40.4: init %i32 = initialize_from %.loc6_40.2 to %.loc6_40.3 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %impl.elem0.loc6_40.2: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc6_40.3: <bound method> = bound_method %int_0.loc6_39, %impl.elem0.loc6_40.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7cb]
// CHECK:STDOUT:   %specific_fn.loc6_40.2: <specific function> = specific_function %impl.elem0.loc6_40.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_40.4: <bound method> = bound_method %int_0.loc6_39, %specific_fn.loc6_40.2 [concrete = constants.%bound_method.698]
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
// CHECK:STDOUT: specific @C(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc4_9.1 => constants.%N.51e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%N.51e) {
// CHECK:STDOUT:   %N => constants.%N.51e
// CHECK:STDOUT:   %C => constants.%C.506
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.2a5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%N.51e) {
// CHECK:STDOUT:   %N => constants.%N.51e
// CHECK:STDOUT:   %C => constants.%C.506
// CHECK:STDOUT:   %ptr => constants.%ptr.128
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d64
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
// CHECK:STDOUT:   %C.17a: type = class_type @C, @C(%N.51e) [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f01: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.acc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6b, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.592: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e4d: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.acc) [concrete]
// CHECK:STDOUT:   %.7ea: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.e4d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7cb: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.592, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.698: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e35: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ee5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.6e1: %ImplicitAs.type.899 = facet_value %C.b00, (%ImplicitAs.impl_witness.e35) [concrete]
// CHECK:STDOUT:   %.93f: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.6e1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.994: type = fn_type @C.as.ImplicitAs.impl.Convert.1 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.78d: %C.as.ImplicitAs.impl.Convert.type.994 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.0d4: <witness> = impl_witness imports.%Destroy.impl_witness_table.f25, @C.as.Destroy.impl(%N.51e) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type.ed4: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%N.51e) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.dd8: %C.as.Destroy.impl.Op.type.ed4 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.9e5: type = ptr_type %C.17a [symbolic]
// CHECK:STDOUT:   %pattern_type.572: type = pattern_type %ptr.9e5 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.c93: <witness> = impl_witness imports.%Destroy.impl_witness_table.f25, @C.as.Destroy.impl(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type.615: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.dbb: %C.as.Destroy.impl.Op.type.615 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.697: type = ptr_type %C.b00 [concrete]
// CHECK:STDOUT:   %pattern_type.7d1: type = pattern_type %ptr.697 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.d03: <specific function> = specific_function %C.as.Destroy.impl.Op.dbb, @C.as.Destroy.impl.Op(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a02: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.b59: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.678: %C.674 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.480 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b04: %ImplicitAs.type.899 = facet_value %C.674, (%ImplicitAs.impl_witness.fb7) [concrete]
// CHECK:STDOUT:   %.fdb: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.b04 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.900: type = fn_type @C.as.ImplicitAs.impl.Convert.2 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.b44: %C.as.ImplicitAs.impl.Convert.type.900 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.b8a: <witness> = impl_witness imports.%Destroy.impl_witness_table.f25, @C.as.Destroy.impl(%int_1.5d2) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type.b82: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%int_1.5d2) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.3c7: %C.as.Destroy.impl.Op.type.b82 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.625: type = ptr_type %C.674 [concrete]
// CHECK:STDOUT:   %pattern_type.559: type = pattern_type %ptr.625 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.ac5: <specific function> = specific_function %C.as.Destroy.impl.Op.3c7, @C.as.Destroy.impl.Op(%int_1.5d2) [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2d9: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.f6f: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.fb5: %C.681 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3de: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.be4 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2fa: %ImplicitAs.type.899 = facet_value %C.681, (%ImplicitAs.impl_witness.3de) [concrete]
// CHECK:STDOUT:   %.f81: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.2fa [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.ae1: type = fn_type @C.as.ImplicitAs.impl.Convert.3 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.dd6: %C.as.ImplicitAs.impl.Convert.type.ae1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.979: <witness> = impl_witness imports.%Destroy.impl_witness_table.f25, @C.as.Destroy.impl(%int_2.ef8) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type.681: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%int_2.ef8) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.f2d: %C.as.Destroy.impl.Op.type.681 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.3bd: type = ptr_type %C.681 [concrete]
// CHECK:STDOUT:   %pattern_type.b50: type = pattern_type %ptr.3bd [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.774: <specific function> = specific_function %C.as.Destroy.impl.Op.f2d, @C.as.Destroy.impl.Op(%int_2.ef8) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c36: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.f79: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.fe7: %C.7ac = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f64: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.636 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.857: %ImplicitAs.type.899 = facet_value %C.7ac, (%ImplicitAs.impl_witness.f64) [concrete]
// CHECK:STDOUT:   %.427: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.857 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.cbe: type = fn_type @C.as.ImplicitAs.impl.Convert.4 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.d1c: %C.as.ImplicitAs.impl.Convert.type.cbe = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.dff: <witness> = impl_witness imports.%Destroy.impl_witness_table.f25, @C.as.Destroy.impl(%int_3.822) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type.f7b: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%int_3.822) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.ee6: %C.as.Destroy.impl.Op.type.f7b = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.2b1: type = ptr_type %C.7ac [concrete]
// CHECK:STDOUT:   %pattern_type.9f4: type = pattern_type %ptr.2b1 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.c71: <specific function> = specific_function %C.as.Destroy.impl.Op.ee6, @C.as.Destroy.impl.Op(%int_3.822) [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.121: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.564: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.1dd: %C.89d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c85: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.16f [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.221: %ImplicitAs.type.899 = facet_value %C.89d, (%ImplicitAs.impl_witness.c85) [concrete]
// CHECK:STDOUT:   %.67f: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.221 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.876: type = fn_type @C.as.ImplicitAs.impl.Convert.5 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.db5: %C.as.ImplicitAs.impl.Convert.type.876 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.1d2: <witness> = impl_witness imports.%Destroy.impl_witness_table.f25, @C.as.Destroy.impl(%int_4.940) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type.f38: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%int_4.940) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.d1c: %C.as.Destroy.impl.Op.type.f38 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28b: type = ptr_type %C.89d [concrete]
// CHECK:STDOUT:   %pattern_type.a33: type = pattern_type %ptr.c28b [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.12e: <specific function> = specific_function %C.as.Destroy.impl.Op.d1c, @C.as.Destroy.impl.Op(%int_4.940) [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.6c8: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.918: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.a4a: %C.f0a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.469: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.865 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d2e: %ImplicitAs.type.899 = facet_value %C.f0a, (%ImplicitAs.impl_witness.469) [concrete]
// CHECK:STDOUT:   %.745: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.d2e [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.043: type = fn_type @C.as.ImplicitAs.impl.Convert.6 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.a63: %C.as.ImplicitAs.impl.Convert.type.043 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.adc: <witness> = impl_witness imports.%Destroy.impl_witness_table.f25, @C.as.Destroy.impl(%int_5.0f6) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type.743: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%int_5.0f6) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.828: %C.as.Destroy.impl.Op.type.743 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.24c: type = ptr_type %C.f0a [concrete]
// CHECK:STDOUT:   %pattern_type.37e: type = pattern_type %ptr.24c [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.295: <specific function> = specific_function %C.as.Destroy.impl.Op.828, @C.as.Destroy.impl.Op(%int_5.0f6) [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7a5: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.351: <bound method> = bound_method %int_6.462, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.a4b: %C.c60 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.68e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.7ee [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9ea: %ImplicitAs.type.899 = facet_value %C.c60, (%ImplicitAs.impl_witness.68e) [concrete]
// CHECK:STDOUT:   %.678: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.9ea [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.7be: type = fn_type @C.as.ImplicitAs.impl.Convert.7 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.e2f: %C.as.ImplicitAs.impl.Convert.type.7be = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.10e: <witness> = impl_witness imports.%Destroy.impl_witness_table.f25, @C.as.Destroy.impl(%int_6.e56) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type.06a: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%int_6.e56) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.c43: %C.as.Destroy.impl.Op.type.06a = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b5e: type = ptr_type %C.c60 [concrete]
// CHECK:STDOUT:   %pattern_type.667: type = pattern_type %ptr.b5e [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.210: <specific function> = specific_function %C.as.Destroy.impl.Op.c43, @C.as.Destroy.impl.Op(%int_6.e56) [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.fa2: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.c8a: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %C.val.f9f: %C.304 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.7f9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9cd [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.466: %ImplicitAs.type.899 = facet_value %C.304, (%ImplicitAs.impl_witness.7f9) [concrete]
// CHECK:STDOUT:   %.bfa: type = fn_type_with_self_type %ImplicitAs.Convert.type.334, %ImplicitAs.facet.466 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type.c30: type = fn_type @C.as.ImplicitAs.impl.Convert.8 [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.5ea: %C.as.ImplicitAs.impl.Convert.type.c30 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.65d: <witness> = impl_witness imports.%Destroy.impl_witness_table.f25, @C.as.Destroy.impl(%int_7.0b1) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type.77c: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%int_7.0b1) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.518: %C.as.Destroy.impl.Op.type.77c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.dc3: type = ptr_type %C.304 [concrete]
// CHECK:STDOUT:   %pattern_type.12a: type = pattern_type %ptr.dc3 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.1ed: <specific function> = specific_function %C.as.Destroy.impl.Op.518, @C.as.Destroy.impl.Op(%int_7.0b1) [concrete]
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
// CHECK:STDOUT:   %P.import_ref.cab = import_ref P//library, inst111 [no loc], unloaded
// CHECK:STDOUT:   %P.import_ref.a52 = import_ref P//library, loc5_16, unloaded
// CHECK:STDOUT:   %P.import_ref.b4a = import_ref P//library, loc5_28, unloaded
// CHECK:STDOUT:   %P.C: %C.type = import_ref P//library, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %P.import_ref.1b7f13.1: %i32 = import_ref P//library, loc4_9, loaded [symbolic = @C.%N (constants.%N.51e)]
// CHECK:STDOUT:   %P.import_ref.8f2: <witness> = import_ref P//library, loc4_19, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %P.import_ref.f65 = import_ref P//library, inst54 [no loc], unloaded
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.428: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f01)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b6b = impl_witness_table (%Core.import_ref.428), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %P.import_ref.dda: <witness> = import_ref P//library, loc8_33, loaded [concrete = constants.%ImplicitAs.impl_witness.e35]
// CHECK:STDOUT:   %P.import_ref.d2c: type = import_ref P//library, loc8_9, loaded [concrete = constants.%C.b00]
// CHECK:STDOUT:   %P.import_ref.a4653a.1: type = import_ref P//library, loc8_31, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.447: <witness> = import_ref P//library, loc9_33, loaded [concrete = constants.%ImplicitAs.impl_witness.fb7]
// CHECK:STDOUT:   %P.import_ref.cf3: type = import_ref P//library, loc9_9, loaded [concrete = constants.%C.674]
// CHECK:STDOUT:   %P.import_ref.a4653a.2: type = import_ref P//library, loc9_31, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.f61: <witness> = import_ref P//library, loc10_33, loaded [concrete = constants.%ImplicitAs.impl_witness.3de]
// CHECK:STDOUT:   %P.import_ref.095: type = import_ref P//library, loc10_9, loaded [concrete = constants.%C.681]
// CHECK:STDOUT:   %P.import_ref.a4653a.3: type = import_ref P//library, loc10_31, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.3f5: <witness> = import_ref P//library, loc11_33, loaded [concrete = constants.%ImplicitAs.impl_witness.f64]
// CHECK:STDOUT:   %P.import_ref.82b: type = import_ref P//library, loc11_9, loaded [concrete = constants.%C.7ac]
// CHECK:STDOUT:   %P.import_ref.a4653a.4: type = import_ref P//library, loc11_31, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.a0f: <witness> = import_ref P//library, loc12_33, loaded [concrete = constants.%ImplicitAs.impl_witness.c85]
// CHECK:STDOUT:   %P.import_ref.f7d: type = import_ref P//library, loc12_9, loaded [concrete = constants.%C.89d]
// CHECK:STDOUT:   %P.import_ref.a4653a.5: type = import_ref P//library, loc12_31, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.e36: <witness> = import_ref P//library, loc13_33, loaded [concrete = constants.%ImplicitAs.impl_witness.469]
// CHECK:STDOUT:   %P.import_ref.a43: type = import_ref P//library, loc13_9, loaded [concrete = constants.%C.f0a]
// CHECK:STDOUT:   %P.import_ref.a4653a.6: type = import_ref P//library, loc13_31, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.14d: <witness> = import_ref P//library, loc14_33, loaded [concrete = constants.%ImplicitAs.impl_witness.68e]
// CHECK:STDOUT:   %P.import_ref.654: type = import_ref P//library, loc14_9, loaded [concrete = constants.%C.c60]
// CHECK:STDOUT:   %P.import_ref.a4653a.7: type = import_ref P//library, loc14_31, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.2b3: <witness> = import_ref P//library, loc15_33, loaded [concrete = constants.%ImplicitAs.impl_witness.7f9]
// CHECK:STDOUT:   %P.import_ref.dae: type = import_ref P//library, loc15_9, loaded [concrete = constants.%C.304]
// CHECK:STDOUT:   %P.import_ref.a4653a.8: type = import_ref P//library, loc15_31, loaded [concrete = constants.%ImplicitAs.type.899]
// CHECK:STDOUT:   %P.import_ref.f2b: %C.as.ImplicitAs.impl.Convert.type.994 = import_ref P//library, loc8_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.78d]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ee5 = impl_witness_table (%P.import_ref.f2b), @C.as.ImplicitAs.impl.87c [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %P.import_ref.82f: <witness> = import_ref P//library, loc4_18, loaded [symbolic = @C.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.0d4)]
// CHECK:STDOUT:   %P.import_ref.1b7f13.2: %i32 = import_ref P//library, loc4_9, loaded [symbolic = @C.%N (constants.%N.51e)]
// CHECK:STDOUT:   %P.import_ref.8b3: type = import_ref P//library, loc4_18, loaded [symbolic = @C.as.Destroy.impl.%C (constants.%C.17a)]
// CHECK:STDOUT:   %P.import_ref.063034.1: type = import_ref P//library, inst57 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %P.import_ref.557: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type.ed4) = import_ref P//library, loc4_18, loaded [symbolic = @C.as.Destroy.impl.%C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op.dd8)]
// CHECK:STDOUT:   %Destroy.impl_witness_table.f25 = impl_witness_table (%P.import_ref.557), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %P.import_ref.1b7f13.3: %i32 = import_ref P//library, loc4_9, loaded [symbolic = @C.%N (constants.%N.51e)]
// CHECK:STDOUT:   %P.import_ref.cd9 = import_ref P//library, loc5_9, unloaded
// CHECK:STDOUT:   %P.import_ref.130: type = import_ref P//library, loc5_9, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %P.import_ref.063034.2: type = import_ref P//library, inst57 [no loc], loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %P.import_ref.057: %C.as.ImplicitAs.impl.Convert.type.900 = import_ref P//library, loc9_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.b44]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.480 = impl_witness_table (%P.import_ref.057), @C.as.ImplicitAs.impl.a01 [concrete]
// CHECK:STDOUT:   %P.import_ref.594: %C.as.ImplicitAs.impl.Convert.type.ae1 = import_ref P//library, loc10_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.dd6]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.be4 = impl_witness_table (%P.import_ref.594), @C.as.ImplicitAs.impl.390 [concrete]
// CHECK:STDOUT:   %P.import_ref.a12: %C.as.ImplicitAs.impl.Convert.type.cbe = import_ref P//library, loc11_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.d1c]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.636 = impl_witness_table (%P.import_ref.a12), @C.as.ImplicitAs.impl.071 [concrete]
// CHECK:STDOUT:   %P.import_ref.df9: %C.as.ImplicitAs.impl.Convert.type.876 = import_ref P//library, loc12_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.db5]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.16f = impl_witness_table (%P.import_ref.df9), @C.as.ImplicitAs.impl.891 [concrete]
// CHECK:STDOUT:   %P.import_ref.c67: %C.as.ImplicitAs.impl.Convert.type.043 = import_ref P//library, loc13_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.a63]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.865 = impl_witness_table (%P.import_ref.c67), @C.as.ImplicitAs.impl.882 [concrete]
// CHECK:STDOUT:   %P.import_ref.ece: %C.as.ImplicitAs.impl.Convert.type.7be = import_ref P//library, loc14_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.e2f]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.7ee = impl_witness_table (%P.import_ref.ece), @C.as.ImplicitAs.impl.ad9 [concrete]
// CHECK:STDOUT:   %P.import_ref.58e: %C.as.ImplicitAs.impl.Convert.type.c30 = import_ref P//library, loc15_65, loaded [concrete = constants.%C.as.ImplicitAs.impl.Convert.5ea]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9cd = impl_witness_table (%P.import_ref.58e), @C.as.ImplicitAs.impl.5d4 [concrete]
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
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.87c: imports.%P.import_ref.d2c as imports.%P.import_ref.a4653a.1 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.dda
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.a01: imports.%P.import_ref.cf3 as imports.%P.import_ref.a4653a.2 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.447
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.390: imports.%P.import_ref.095 as imports.%P.import_ref.a4653a.3 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.f61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.071: imports.%P.import_ref.82b as imports.%P.import_ref.a4653a.4 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.3f5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.891: imports.%P.import_ref.f7d as imports.%P.import_ref.a4653a.5 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.a0f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.882: imports.%P.import_ref.a43 as imports.%P.import_ref.a4653a.6 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.e36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.ad9: imports.%P.import_ref.654 as imports.%P.import_ref.a4653a.7 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.14d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.ImplicitAs.impl.5d4: imports.%P.import_ref.dae as imports.%P.import_ref.a4653a.8 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.2b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(imports.%P.import_ref.1b7f13.2: %i32) [from "library.carbon"] {
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.51e)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%N) [symbolic = %C (constants.%C.17a)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table.f25, @C.as.Destroy.impl(%N) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.0d4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%N) [symbolic = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type.ed4)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type.ed4) = struct_value () [symbolic = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op.dd8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: imports.%P.import_ref.8b3 as imports.%P.import_ref.063034.1 {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     witness = imports.%P.import_ref.82f
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.Destroy.impl: imports.%P.import_ref.130 as imports.%P.import_ref.063034.2 [from "library.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%P.import_ref.cd9
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
// CHECK:STDOUT: generic class @C(imports.%P.import_ref.1b7f13.1: %i32) [from "library.carbon"] {
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
// CHECK:STDOUT:   %impl.elem0.loc7_34: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc7_34.1: <bound method> = bound_method %int_0, %impl.elem0.loc7_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7cb]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_34.2: <bound method> = bound_method %int_0, %specific_fn.loc7 [concrete = constants.%bound_method.698]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_34.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc7_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc7_34.2: %i32 = converted %int_0, %.loc7_34.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %C.loc7: type = class_type @C, @C(constants.%int_0.6a9) [concrete = constants.%C.b00]
// CHECK:STDOUT:   %.loc7_24.2: ref %C.b00 = temporary_storage
// CHECK:STDOUT:   %.loc7_24.3: init %C.b00 = class_init (), %.loc7_24.2 [concrete = constants.%C.val.452]
// CHECK:STDOUT:   %.loc7_24.4: ref %C.b00 = temporary %.loc7_24.2, %.loc7_24.3
// CHECK:STDOUT:   %.loc7_26.1: ref %C.b00 = converted %.loc7_24.1, %.loc7_24.4
// CHECK:STDOUT:   %impl.elem0.loc7_35: %.93f = impl_witness_access constants.%ImplicitAs.impl_witness.e35, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.78d]
// CHECK:STDOUT:   %bound_method.loc7_35: <bound method> = bound_method %.loc7_26.1, %impl.elem0.loc7_35
// CHECK:STDOUT:   %.loc7_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc7_26.2: %C.b00 = bind_value %.loc7_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc7: init %D = call %bound_method.loc7_35(%.loc7_26.2) to %.loc7_35.1
// CHECK:STDOUT:   %.loc7_35.2: init %D = converted %.loc7_26.1, %C.as.ImplicitAs.impl.Convert.call.loc7
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc7_24.1: <bound method> = bound_method %.loc7_24.4, constants.%C.as.Destroy.impl.Op.dbb
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.dbb, @C.as.Destroy.impl.Op(constants.%int_0.6a9) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.d03]
// CHECK:STDOUT:   %bound_method.loc7_24.1: <bound method> = bound_method %.loc7_24.4, %C.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc7_24.1: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc7_24.1: init %empty_tuple.type = call %bound_method.loc7_24.1(%addr.loc7_24.1)
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
// CHECK:STDOUT:   %impl.elem0.loc8_34: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a02]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_1, %specific_fn.loc8 [concrete = constants.%bound_method.b59]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_34.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_1, %.loc8_34.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %C.loc8: type = class_type @C, @C(constants.%int_1.5d2) [concrete = constants.%C.674]
// CHECK:STDOUT:   %.loc8_24.2: ref %C.674 = temporary_storage
// CHECK:STDOUT:   %.loc8_24.3: init %C.674 = class_init (), %.loc8_24.2 [concrete = constants.%C.val.678]
// CHECK:STDOUT:   %.loc8_24.4: ref %C.674 = temporary %.loc8_24.2, %.loc8_24.3
// CHECK:STDOUT:   %.loc8_26.1: ref %C.674 = converted %.loc8_24.1, %.loc8_24.4
// CHECK:STDOUT:   %impl.elem0.loc8_35: %.fdb = impl_witness_access constants.%ImplicitAs.impl_witness.fb7, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.b44]
// CHECK:STDOUT:   %bound_method.loc8_35: <bound method> = bound_method %.loc8_26.1, %impl.elem0.loc8_35
// CHECK:STDOUT:   %.loc8_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc8_26.2: %C.674 = bind_value %.loc8_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc8: init %D = call %bound_method.loc8_35(%.loc8_26.2) to %.loc8_35.1
// CHECK:STDOUT:   %.loc8_35.2: init %D = converted %.loc8_26.1, %C.as.ImplicitAs.impl.Convert.call.loc8
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc8_24.1: <bound method> = bound_method %.loc8_24.4, constants.%C.as.Destroy.impl.Op.3c7
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.3c7, @C.as.Destroy.impl.Op(constants.%int_1.5d2) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.ac5]
// CHECK:STDOUT:   %bound_method.loc8_24.1: <bound method> = bound_method %.loc8_24.4, %C.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc8_24.1: %ptr.625 = addr_of %.loc8_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc8_24.1: init %empty_tuple.type = call %bound_method.loc8_24.1(%addr.loc8_24.1)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc7_24.2: <bound method> = bound_method %.loc7_24.4, constants.%C.as.Destroy.impl.Op.dbb
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.3: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.dbb, @C.as.Destroy.impl.Op(constants.%int_0.6a9) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.d03]
// CHECK:STDOUT:   %bound_method.loc7_24.2: <bound method> = bound_method %.loc7_24.4, %C.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc7_24.2: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc7_24.2: init %empty_tuple.type = call %bound_method.loc7_24.2(%addr.loc7_24.2)
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
// CHECK:STDOUT:   %impl.elem0.loc9_34: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc9_34.1: <bound method> = bound_method %int_2, %impl.elem0.loc9_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2d9]
// CHECK:STDOUT:   %specific_fn.loc9: <specific function> = specific_function %impl.elem0.loc9_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_34.2: <bound method> = bound_method %int_2, %specific_fn.loc9 [concrete = constants.%bound_method.f6f]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9: init %i32 = call %bound_method.loc9_34.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_34.2: %i32 = converted %int_2, %.loc9_34.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %C.loc9: type = class_type @C, @C(constants.%int_2.ef8) [concrete = constants.%C.681]
// CHECK:STDOUT:   %.loc9_24.2: ref %C.681 = temporary_storage
// CHECK:STDOUT:   %.loc9_24.3: init %C.681 = class_init (), %.loc9_24.2 [concrete = constants.%C.val.fb5]
// CHECK:STDOUT:   %.loc9_24.4: ref %C.681 = temporary %.loc9_24.2, %.loc9_24.3
// CHECK:STDOUT:   %.loc9_26.1: ref %C.681 = converted %.loc9_24.1, %.loc9_24.4
// CHECK:STDOUT:   %impl.elem0.loc9_35: %.f81 = impl_witness_access constants.%ImplicitAs.impl_witness.3de, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.dd6]
// CHECK:STDOUT:   %bound_method.loc9_35: <bound method> = bound_method %.loc9_26.1, %impl.elem0.loc9_35
// CHECK:STDOUT:   %.loc9_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc9_26.2: %C.681 = bind_value %.loc9_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc9: init %D = call %bound_method.loc9_35(%.loc9_26.2) to %.loc9_35.1
// CHECK:STDOUT:   %.loc9_35.2: init %D = converted %.loc9_26.1, %C.as.ImplicitAs.impl.Convert.call.loc9
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc9_24.1: <bound method> = bound_method %.loc9_24.4, constants.%C.as.Destroy.impl.Op.f2d
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.4: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.f2d, @C.as.Destroy.impl.Op(constants.%int_2.ef8) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.774]
// CHECK:STDOUT:   %bound_method.loc9_24.1: <bound method> = bound_method %.loc9_24.4, %C.as.Destroy.impl.Op.specific_fn.4
// CHECK:STDOUT:   %addr.loc9_24.1: %ptr.3bd = addr_of %.loc9_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc9_24.1: init %empty_tuple.type = call %bound_method.loc9_24.1(%addr.loc9_24.1)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc8_24.2: <bound method> = bound_method %.loc8_24.4, constants.%C.as.Destroy.impl.Op.3c7
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.5: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.3c7, @C.as.Destroy.impl.Op(constants.%int_1.5d2) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.ac5]
// CHECK:STDOUT:   %bound_method.loc8_24.2: <bound method> = bound_method %.loc8_24.4, %C.as.Destroy.impl.Op.specific_fn.5
// CHECK:STDOUT:   %addr.loc8_24.2: %ptr.625 = addr_of %.loc8_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc8_24.2: init %empty_tuple.type = call %bound_method.loc8_24.2(%addr.loc8_24.2)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc7_24.3: <bound method> = bound_method %.loc7_24.4, constants.%C.as.Destroy.impl.Op.dbb
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.6: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.dbb, @C.as.Destroy.impl.Op(constants.%int_0.6a9) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.d03]
// CHECK:STDOUT:   %bound_method.loc7_24.3: <bound method> = bound_method %.loc7_24.4, %C.as.Destroy.impl.Op.specific_fn.6
// CHECK:STDOUT:   %addr.loc7_24.3: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc7_24.3: init %empty_tuple.type = call %bound_method.loc7_24.3(%addr.loc7_24.3)
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
// CHECK:STDOUT:   %impl.elem0.loc10_34: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc10_34.1: <bound method> = bound_method %int_3, %impl.elem0.loc10_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.c36]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_34.2: <bound method> = bound_method %int_3, %specific_fn.loc10 [concrete = constants.%bound_method.f79]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %i32 = call %bound_method.loc10_34.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc10_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc10_34.2: %i32 = converted %int_3, %.loc10_34.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %C.loc10: type = class_type @C, @C(constants.%int_3.822) [concrete = constants.%C.7ac]
// CHECK:STDOUT:   %.loc10_24.2: ref %C.7ac = temporary_storage
// CHECK:STDOUT:   %.loc10_24.3: init %C.7ac = class_init (), %.loc10_24.2 [concrete = constants.%C.val.fe7]
// CHECK:STDOUT:   %.loc10_24.4: ref %C.7ac = temporary %.loc10_24.2, %.loc10_24.3
// CHECK:STDOUT:   %.loc10_26.1: ref %C.7ac = converted %.loc10_24.1, %.loc10_24.4
// CHECK:STDOUT:   %impl.elem0.loc10_35: %.427 = impl_witness_access constants.%ImplicitAs.impl_witness.f64, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.d1c]
// CHECK:STDOUT:   %bound_method.loc10_35: <bound method> = bound_method %.loc10_26.1, %impl.elem0.loc10_35
// CHECK:STDOUT:   %.loc10_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc10_26.2: %C.7ac = bind_value %.loc10_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc10: init %D = call %bound_method.loc10_35(%.loc10_26.2) to %.loc10_35.1
// CHECK:STDOUT:   %.loc10_35.2: init %D = converted %.loc10_26.1, %C.as.ImplicitAs.impl.Convert.call.loc10
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc10_24.1: <bound method> = bound_method %.loc10_24.4, constants.%C.as.Destroy.impl.Op.ee6
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.7: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.ee6, @C.as.Destroy.impl.Op(constants.%int_3.822) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.c71]
// CHECK:STDOUT:   %bound_method.loc10_24.1: <bound method> = bound_method %.loc10_24.4, %C.as.Destroy.impl.Op.specific_fn.7
// CHECK:STDOUT:   %addr.loc10_24.1: %ptr.2b1 = addr_of %.loc10_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc10_24.1: init %empty_tuple.type = call %bound_method.loc10_24.1(%addr.loc10_24.1)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc9_24.2: <bound method> = bound_method %.loc9_24.4, constants.%C.as.Destroy.impl.Op.f2d
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.8: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.f2d, @C.as.Destroy.impl.Op(constants.%int_2.ef8) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.774]
// CHECK:STDOUT:   %bound_method.loc9_24.2: <bound method> = bound_method %.loc9_24.4, %C.as.Destroy.impl.Op.specific_fn.8
// CHECK:STDOUT:   %addr.loc9_24.2: %ptr.3bd = addr_of %.loc9_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc9_24.2: init %empty_tuple.type = call %bound_method.loc9_24.2(%addr.loc9_24.2)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc8_24.3: <bound method> = bound_method %.loc8_24.4, constants.%C.as.Destroy.impl.Op.3c7
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.9: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.3c7, @C.as.Destroy.impl.Op(constants.%int_1.5d2) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.ac5]
// CHECK:STDOUT:   %bound_method.loc8_24.3: <bound method> = bound_method %.loc8_24.4, %C.as.Destroy.impl.Op.specific_fn.9
// CHECK:STDOUT:   %addr.loc8_24.3: %ptr.625 = addr_of %.loc8_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc8_24.3: init %empty_tuple.type = call %bound_method.loc8_24.3(%addr.loc8_24.3)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc7_24.4: <bound method> = bound_method %.loc7_24.4, constants.%C.as.Destroy.impl.Op.dbb
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.10: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.dbb, @C.as.Destroy.impl.Op(constants.%int_0.6a9) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.d03]
// CHECK:STDOUT:   %bound_method.loc7_24.4: <bound method> = bound_method %.loc7_24.4, %C.as.Destroy.impl.Op.specific_fn.10
// CHECK:STDOUT:   %addr.loc7_24.4: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc7_24.4: init %empty_tuple.type = call %bound_method.loc7_24.4(%addr.loc7_24.4)
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
// CHECK:STDOUT:   %impl.elem0.loc11_34: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc11_34.1: <bound method> = bound_method %int_4, %impl.elem0.loc11_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.121]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_34.2: <bound method> = bound_method %int_4, %specific_fn.loc11 [concrete = constants.%bound_method.564]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %i32 = call %bound_method.loc11_34.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc11_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc11_34.2: %i32 = converted %int_4, %.loc11_34.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %C.loc11: type = class_type @C, @C(constants.%int_4.940) [concrete = constants.%C.89d]
// CHECK:STDOUT:   %.loc11_24.2: ref %C.89d = temporary_storage
// CHECK:STDOUT:   %.loc11_24.3: init %C.89d = class_init (), %.loc11_24.2 [concrete = constants.%C.val.1dd]
// CHECK:STDOUT:   %.loc11_24.4: ref %C.89d = temporary %.loc11_24.2, %.loc11_24.3
// CHECK:STDOUT:   %.loc11_26.1: ref %C.89d = converted %.loc11_24.1, %.loc11_24.4
// CHECK:STDOUT:   %impl.elem0.loc11_35: %.67f = impl_witness_access constants.%ImplicitAs.impl_witness.c85, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.db5]
// CHECK:STDOUT:   %bound_method.loc11_35: <bound method> = bound_method %.loc11_26.1, %impl.elem0.loc11_35
// CHECK:STDOUT:   %.loc11_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc11_26.2: %C.89d = bind_value %.loc11_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc11: init %D = call %bound_method.loc11_35(%.loc11_26.2) to %.loc11_35.1
// CHECK:STDOUT:   %.loc11_35.2: init %D = converted %.loc11_26.1, %C.as.ImplicitAs.impl.Convert.call.loc11
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc11_24.1: <bound method> = bound_method %.loc11_24.4, constants.%C.as.Destroy.impl.Op.d1c
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.11: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.d1c, @C.as.Destroy.impl.Op(constants.%int_4.940) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.12e]
// CHECK:STDOUT:   %bound_method.loc11_24.1: <bound method> = bound_method %.loc11_24.4, %C.as.Destroy.impl.Op.specific_fn.11
// CHECK:STDOUT:   %addr.loc11_24.1: %ptr.c28b = addr_of %.loc11_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc11_24.1: init %empty_tuple.type = call %bound_method.loc11_24.1(%addr.loc11_24.1)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc10_24.2: <bound method> = bound_method %.loc10_24.4, constants.%C.as.Destroy.impl.Op.ee6
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.12: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.ee6, @C.as.Destroy.impl.Op(constants.%int_3.822) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.c71]
// CHECK:STDOUT:   %bound_method.loc10_24.2: <bound method> = bound_method %.loc10_24.4, %C.as.Destroy.impl.Op.specific_fn.12
// CHECK:STDOUT:   %addr.loc10_24.2: %ptr.2b1 = addr_of %.loc10_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc10_24.2: init %empty_tuple.type = call %bound_method.loc10_24.2(%addr.loc10_24.2)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc9_24.3: <bound method> = bound_method %.loc9_24.4, constants.%C.as.Destroy.impl.Op.f2d
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.13: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.f2d, @C.as.Destroy.impl.Op(constants.%int_2.ef8) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.774]
// CHECK:STDOUT:   %bound_method.loc9_24.3: <bound method> = bound_method %.loc9_24.4, %C.as.Destroy.impl.Op.specific_fn.13
// CHECK:STDOUT:   %addr.loc9_24.3: %ptr.3bd = addr_of %.loc9_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc9_24.3: init %empty_tuple.type = call %bound_method.loc9_24.3(%addr.loc9_24.3)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc8_24.4: <bound method> = bound_method %.loc8_24.4, constants.%C.as.Destroy.impl.Op.3c7
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.14: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.3c7, @C.as.Destroy.impl.Op(constants.%int_1.5d2) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.ac5]
// CHECK:STDOUT:   %bound_method.loc8_24.4: <bound method> = bound_method %.loc8_24.4, %C.as.Destroy.impl.Op.specific_fn.14
// CHECK:STDOUT:   %addr.loc8_24.4: %ptr.625 = addr_of %.loc8_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc8_24.4: init %empty_tuple.type = call %bound_method.loc8_24.4(%addr.loc8_24.4)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc7_24.5: <bound method> = bound_method %.loc7_24.4, constants.%C.as.Destroy.impl.Op.dbb
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.15: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.dbb, @C.as.Destroy.impl.Op(constants.%int_0.6a9) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.d03]
// CHECK:STDOUT:   %bound_method.loc7_24.5: <bound method> = bound_method %.loc7_24.4, %C.as.Destroy.impl.Op.specific_fn.15
// CHECK:STDOUT:   %addr.loc7_24.5: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc7_24.5: init %empty_tuple.type = call %bound_method.loc7_24.5(%addr.loc7_24.5)
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
// CHECK:STDOUT:   %impl.elem0.loc12_34: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc12_34.1: <bound method> = bound_method %int_5, %impl.elem0.loc12_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.6c8]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_34.2: <bound method> = bound_method %int_5, %specific_fn.loc12 [concrete = constants.%bound_method.918]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %i32 = call %bound_method.loc12_34.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc12_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc12_34.2: %i32 = converted %int_5, %.loc12_34.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %C.loc12: type = class_type @C, @C(constants.%int_5.0f6) [concrete = constants.%C.f0a]
// CHECK:STDOUT:   %.loc12_24.2: ref %C.f0a = temporary_storage
// CHECK:STDOUT:   %.loc12_24.3: init %C.f0a = class_init (), %.loc12_24.2 [concrete = constants.%C.val.a4a]
// CHECK:STDOUT:   %.loc12_24.4: ref %C.f0a = temporary %.loc12_24.2, %.loc12_24.3
// CHECK:STDOUT:   %.loc12_26.1: ref %C.f0a = converted %.loc12_24.1, %.loc12_24.4
// CHECK:STDOUT:   %impl.elem0.loc12_35: %.745 = impl_witness_access constants.%ImplicitAs.impl_witness.469, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.a63]
// CHECK:STDOUT:   %bound_method.loc12_35: <bound method> = bound_method %.loc12_26.1, %impl.elem0.loc12_35
// CHECK:STDOUT:   %.loc12_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc12_26.2: %C.f0a = bind_value %.loc12_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc12: init %D = call %bound_method.loc12_35(%.loc12_26.2) to %.loc12_35.1
// CHECK:STDOUT:   %.loc12_35.2: init %D = converted %.loc12_26.1, %C.as.ImplicitAs.impl.Convert.call.loc12
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc12_24.1: <bound method> = bound_method %.loc12_24.4, constants.%C.as.Destroy.impl.Op.828
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.16: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.828, @C.as.Destroy.impl.Op(constants.%int_5.0f6) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.295]
// CHECK:STDOUT:   %bound_method.loc12_24.1: <bound method> = bound_method %.loc12_24.4, %C.as.Destroy.impl.Op.specific_fn.16
// CHECK:STDOUT:   %addr.loc12_24.1: %ptr.24c = addr_of %.loc12_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc12_24.1: init %empty_tuple.type = call %bound_method.loc12_24.1(%addr.loc12_24.1)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc11_24.2: <bound method> = bound_method %.loc11_24.4, constants.%C.as.Destroy.impl.Op.d1c
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.17: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.d1c, @C.as.Destroy.impl.Op(constants.%int_4.940) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.12e]
// CHECK:STDOUT:   %bound_method.loc11_24.2: <bound method> = bound_method %.loc11_24.4, %C.as.Destroy.impl.Op.specific_fn.17
// CHECK:STDOUT:   %addr.loc11_24.2: %ptr.c28b = addr_of %.loc11_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc11_24.2: init %empty_tuple.type = call %bound_method.loc11_24.2(%addr.loc11_24.2)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc10_24.3: <bound method> = bound_method %.loc10_24.4, constants.%C.as.Destroy.impl.Op.ee6
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.18: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.ee6, @C.as.Destroy.impl.Op(constants.%int_3.822) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.c71]
// CHECK:STDOUT:   %bound_method.loc10_24.3: <bound method> = bound_method %.loc10_24.4, %C.as.Destroy.impl.Op.specific_fn.18
// CHECK:STDOUT:   %addr.loc10_24.3: %ptr.2b1 = addr_of %.loc10_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc10_24.3: init %empty_tuple.type = call %bound_method.loc10_24.3(%addr.loc10_24.3)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc9_24.4: <bound method> = bound_method %.loc9_24.4, constants.%C.as.Destroy.impl.Op.f2d
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.19: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.f2d, @C.as.Destroy.impl.Op(constants.%int_2.ef8) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.774]
// CHECK:STDOUT:   %bound_method.loc9_24.4: <bound method> = bound_method %.loc9_24.4, %C.as.Destroy.impl.Op.specific_fn.19
// CHECK:STDOUT:   %addr.loc9_24.4: %ptr.3bd = addr_of %.loc9_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc9_24.4: init %empty_tuple.type = call %bound_method.loc9_24.4(%addr.loc9_24.4)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc8_24.5: <bound method> = bound_method %.loc8_24.4, constants.%C.as.Destroy.impl.Op.3c7
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.20: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.3c7, @C.as.Destroy.impl.Op(constants.%int_1.5d2) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.ac5]
// CHECK:STDOUT:   %bound_method.loc8_24.5: <bound method> = bound_method %.loc8_24.4, %C.as.Destroy.impl.Op.specific_fn.20
// CHECK:STDOUT:   %addr.loc8_24.5: %ptr.625 = addr_of %.loc8_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc8_24.5: init %empty_tuple.type = call %bound_method.loc8_24.5(%addr.loc8_24.5)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc7_24.6: <bound method> = bound_method %.loc7_24.4, constants.%C.as.Destroy.impl.Op.dbb
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.21: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.dbb, @C.as.Destroy.impl.Op(constants.%int_0.6a9) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.d03]
// CHECK:STDOUT:   %bound_method.loc7_24.6: <bound method> = bound_method %.loc7_24.4, %C.as.Destroy.impl.Op.specific_fn.21
// CHECK:STDOUT:   %addr.loc7_24.6: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc7_24.6: init %empty_tuple.type = call %bound_method.loc7_24.6(%addr.loc7_24.6)
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
// CHECK:STDOUT:   %impl.elem0.loc13_34: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc13_34.1: <bound method> = bound_method %int_6, %impl.elem0.loc13_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7a5]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_34.2: <bound method> = bound_method %int_6, %specific_fn.loc13 [concrete = constants.%bound_method.351]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %i32 = call %bound_method.loc13_34.2(%int_6) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc13_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc13_34.2: %i32 = converted %int_6, %.loc13_34.1 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %C.loc13: type = class_type @C, @C(constants.%int_6.e56) [concrete = constants.%C.c60]
// CHECK:STDOUT:   %.loc13_24.2: ref %C.c60 = temporary_storage
// CHECK:STDOUT:   %.loc13_24.3: init %C.c60 = class_init (), %.loc13_24.2 [concrete = constants.%C.val.a4b]
// CHECK:STDOUT:   %.loc13_24.4: ref %C.c60 = temporary %.loc13_24.2, %.loc13_24.3
// CHECK:STDOUT:   %.loc13_26.1: ref %C.c60 = converted %.loc13_24.1, %.loc13_24.4
// CHECK:STDOUT:   %impl.elem0.loc13_35: %.678 = impl_witness_access constants.%ImplicitAs.impl_witness.68e, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.e2f]
// CHECK:STDOUT:   %bound_method.loc13_35: <bound method> = bound_method %.loc13_26.1, %impl.elem0.loc13_35
// CHECK:STDOUT:   %.loc13_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc13_26.2: %C.c60 = bind_value %.loc13_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc13: init %D = call %bound_method.loc13_35(%.loc13_26.2) to %.loc13_35.1
// CHECK:STDOUT:   %.loc13_35.2: init %D = converted %.loc13_26.1, %C.as.ImplicitAs.impl.Convert.call.loc13
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc13_24.1: <bound method> = bound_method %.loc13_24.4, constants.%C.as.Destroy.impl.Op.c43
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.22: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.c43, @C.as.Destroy.impl.Op(constants.%int_6.e56) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.210]
// CHECK:STDOUT:   %bound_method.loc13_24.1: <bound method> = bound_method %.loc13_24.4, %C.as.Destroy.impl.Op.specific_fn.22
// CHECK:STDOUT:   %addr.loc13_24.1: %ptr.b5e = addr_of %.loc13_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc13_24.1: init %empty_tuple.type = call %bound_method.loc13_24.1(%addr.loc13_24.1)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc12_24.2: <bound method> = bound_method %.loc12_24.4, constants.%C.as.Destroy.impl.Op.828
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.23: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.828, @C.as.Destroy.impl.Op(constants.%int_5.0f6) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.295]
// CHECK:STDOUT:   %bound_method.loc12_24.2: <bound method> = bound_method %.loc12_24.4, %C.as.Destroy.impl.Op.specific_fn.23
// CHECK:STDOUT:   %addr.loc12_24.2: %ptr.24c = addr_of %.loc12_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc12_24.2: init %empty_tuple.type = call %bound_method.loc12_24.2(%addr.loc12_24.2)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc11_24.3: <bound method> = bound_method %.loc11_24.4, constants.%C.as.Destroy.impl.Op.d1c
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.24: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.d1c, @C.as.Destroy.impl.Op(constants.%int_4.940) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.12e]
// CHECK:STDOUT:   %bound_method.loc11_24.3: <bound method> = bound_method %.loc11_24.4, %C.as.Destroy.impl.Op.specific_fn.24
// CHECK:STDOUT:   %addr.loc11_24.3: %ptr.c28b = addr_of %.loc11_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc11_24.3: init %empty_tuple.type = call %bound_method.loc11_24.3(%addr.loc11_24.3)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc10_24.4: <bound method> = bound_method %.loc10_24.4, constants.%C.as.Destroy.impl.Op.ee6
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.25: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.ee6, @C.as.Destroy.impl.Op(constants.%int_3.822) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.c71]
// CHECK:STDOUT:   %bound_method.loc10_24.4: <bound method> = bound_method %.loc10_24.4, %C.as.Destroy.impl.Op.specific_fn.25
// CHECK:STDOUT:   %addr.loc10_24.4: %ptr.2b1 = addr_of %.loc10_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc10_24.4: init %empty_tuple.type = call %bound_method.loc10_24.4(%addr.loc10_24.4)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc9_24.5: <bound method> = bound_method %.loc9_24.4, constants.%C.as.Destroy.impl.Op.f2d
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.26: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.f2d, @C.as.Destroy.impl.Op(constants.%int_2.ef8) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.774]
// CHECK:STDOUT:   %bound_method.loc9_24.5: <bound method> = bound_method %.loc9_24.4, %C.as.Destroy.impl.Op.specific_fn.26
// CHECK:STDOUT:   %addr.loc9_24.5: %ptr.3bd = addr_of %.loc9_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc9_24.5: init %empty_tuple.type = call %bound_method.loc9_24.5(%addr.loc9_24.5)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc8_24.6: <bound method> = bound_method %.loc8_24.4, constants.%C.as.Destroy.impl.Op.3c7
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.27: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.3c7, @C.as.Destroy.impl.Op(constants.%int_1.5d2) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.ac5]
// CHECK:STDOUT:   %bound_method.loc8_24.6: <bound method> = bound_method %.loc8_24.4, %C.as.Destroy.impl.Op.specific_fn.27
// CHECK:STDOUT:   %addr.loc8_24.6: %ptr.625 = addr_of %.loc8_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc8_24.6: init %empty_tuple.type = call %bound_method.loc8_24.6(%addr.loc8_24.6)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc7_24.7: <bound method> = bound_method %.loc7_24.4, constants.%C.as.Destroy.impl.Op.dbb
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.28: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.dbb, @C.as.Destroy.impl.Op(constants.%int_0.6a9) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.d03]
// CHECK:STDOUT:   %bound_method.loc7_24.7: <bound method> = bound_method %.loc7_24.4, %C.as.Destroy.impl.Op.specific_fn.28
// CHECK:STDOUT:   %addr.loc7_24.7: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc7_24.7: init %empty_tuple.type = call %bound_method.loc7_24.7(%addr.loc7_24.7)
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
// CHECK:STDOUT:   %impl.elem0.loc14_34: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc14_34.1: <bound method> = bound_method %int_7, %impl.elem0.loc14_34 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.fa2]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14_34, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.2: <bound method> = bound_method %int_7, %specific_fn.loc14 [concrete = constants.%bound_method.c8a]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14: init %i32 = call %bound_method.loc14_34.2(%int_7) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc14_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc14_34.2: %i32 = converted %int_7, %.loc14_34.1 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %C.loc14: type = class_type @C, @C(constants.%int_7.0b1) [concrete = constants.%C.304]
// CHECK:STDOUT:   %.loc14_24.2: ref %C.304 = temporary_storage
// CHECK:STDOUT:   %.loc14_24.3: init %C.304 = class_init (), %.loc14_24.2 [concrete = constants.%C.val.f9f]
// CHECK:STDOUT:   %.loc14_24.4: ref %C.304 = temporary %.loc14_24.2, %.loc14_24.3
// CHECK:STDOUT:   %.loc14_26.1: ref %C.304 = converted %.loc14_24.1, %.loc14_24.4
// CHECK:STDOUT:   %impl.elem0.loc14_35: %.bfa = impl_witness_access constants.%ImplicitAs.impl_witness.7f9, element0 [concrete = constants.%C.as.ImplicitAs.impl.Convert.5ea]
// CHECK:STDOUT:   %bound_method.loc14_35: <bound method> = bound_method %.loc14_26.1, %impl.elem0.loc14_35
// CHECK:STDOUT:   %.loc14_35.1: ref %D = temporary_storage
// CHECK:STDOUT:   %.loc14_26.2: %C.304 = bind_value %.loc14_26.1
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.call.loc14: init %D = call %bound_method.loc14_35(%.loc14_26.2) to %.loc14_35.1
// CHECK:STDOUT:   %.loc14_35.2: init %D = converted %.loc14_26.1, %C.as.ImplicitAs.impl.Convert.call.loc14
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc14_24.1: <bound method> = bound_method %.loc14_24.4, constants.%C.as.Destroy.impl.Op.518
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.29: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.518, @C.as.Destroy.impl.Op(constants.%int_7.0b1) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.1ed]
// CHECK:STDOUT:   %bound_method.loc14_24.1: <bound method> = bound_method %.loc14_24.4, %C.as.Destroy.impl.Op.specific_fn.29
// CHECK:STDOUT:   %addr.loc14_24.1: %ptr.dc3 = addr_of %.loc14_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc14_24.1: init %empty_tuple.type = call %bound_method.loc14_24.1(%addr.loc14_24.1)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc13_24.2: <bound method> = bound_method %.loc13_24.4, constants.%C.as.Destroy.impl.Op.c43
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.30: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.c43, @C.as.Destroy.impl.Op(constants.%int_6.e56) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.210]
// CHECK:STDOUT:   %bound_method.loc13_24.2: <bound method> = bound_method %.loc13_24.4, %C.as.Destroy.impl.Op.specific_fn.30
// CHECK:STDOUT:   %addr.loc13_24.2: %ptr.b5e = addr_of %.loc13_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc13_24.2: init %empty_tuple.type = call %bound_method.loc13_24.2(%addr.loc13_24.2)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc12_24.3: <bound method> = bound_method %.loc12_24.4, constants.%C.as.Destroy.impl.Op.828
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.31: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.828, @C.as.Destroy.impl.Op(constants.%int_5.0f6) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.295]
// CHECK:STDOUT:   %bound_method.loc12_24.3: <bound method> = bound_method %.loc12_24.4, %C.as.Destroy.impl.Op.specific_fn.31
// CHECK:STDOUT:   %addr.loc12_24.3: %ptr.24c = addr_of %.loc12_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc12_24.3: init %empty_tuple.type = call %bound_method.loc12_24.3(%addr.loc12_24.3)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc11_24.4: <bound method> = bound_method %.loc11_24.4, constants.%C.as.Destroy.impl.Op.d1c
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.32: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.d1c, @C.as.Destroy.impl.Op(constants.%int_4.940) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.12e]
// CHECK:STDOUT:   %bound_method.loc11_24.4: <bound method> = bound_method %.loc11_24.4, %C.as.Destroy.impl.Op.specific_fn.32
// CHECK:STDOUT:   %addr.loc11_24.4: %ptr.c28b = addr_of %.loc11_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc11_24.4: init %empty_tuple.type = call %bound_method.loc11_24.4(%addr.loc11_24.4)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc10_24.5: <bound method> = bound_method %.loc10_24.4, constants.%C.as.Destroy.impl.Op.ee6
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.33: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.ee6, @C.as.Destroy.impl.Op(constants.%int_3.822) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.c71]
// CHECK:STDOUT:   %bound_method.loc10_24.5: <bound method> = bound_method %.loc10_24.4, %C.as.Destroy.impl.Op.specific_fn.33
// CHECK:STDOUT:   %addr.loc10_24.5: %ptr.2b1 = addr_of %.loc10_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc10_24.5: init %empty_tuple.type = call %bound_method.loc10_24.5(%addr.loc10_24.5)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc9_24.6: <bound method> = bound_method %.loc9_24.4, constants.%C.as.Destroy.impl.Op.f2d
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.34: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.f2d, @C.as.Destroy.impl.Op(constants.%int_2.ef8) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.774]
// CHECK:STDOUT:   %bound_method.loc9_24.6: <bound method> = bound_method %.loc9_24.4, %C.as.Destroy.impl.Op.specific_fn.34
// CHECK:STDOUT:   %addr.loc9_24.6: %ptr.3bd = addr_of %.loc9_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc9_24.6: init %empty_tuple.type = call %bound_method.loc9_24.6(%addr.loc9_24.6)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc8_24.7: <bound method> = bound_method %.loc8_24.4, constants.%C.as.Destroy.impl.Op.3c7
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.35: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.3c7, @C.as.Destroy.impl.Op(constants.%int_1.5d2) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.ac5]
// CHECK:STDOUT:   %bound_method.loc8_24.7: <bound method> = bound_method %.loc8_24.4, %C.as.Destroy.impl.Op.specific_fn.35
// CHECK:STDOUT:   %addr.loc8_24.7: %ptr.625 = addr_of %.loc8_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc8_24.7: init %empty_tuple.type = call %bound_method.loc8_24.7(%addr.loc8_24.7)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc7_24.8: <bound method> = bound_method %.loc7_24.4, constants.%C.as.Destroy.impl.Op.dbb
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.36: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.dbb, @C.as.Destroy.impl.Op(constants.%int_0.6a9) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.d03]
// CHECK:STDOUT:   %bound_method.loc7_24.8: <bound method> = bound_method %.loc7_24.4, %C.as.Destroy.impl.Op.specific_fn.36
// CHECK:STDOUT:   %addr.loc7_24.8: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc7_24.8: init %empty_tuple.type = call %bound_method.loc7_24.8(%addr.loc7_24.8)
// CHECK:STDOUT:   return %.loc14_35.2 to %return
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc14:
// CHECK:STDOUT:   %P.ref.loc15: <namespace> = name_ref P, imports.%P [concrete = imports.%P]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, imports.%P.Make [concrete = constants.%Make]
// CHECK:STDOUT:   %.loc6_15: ref %D = splice_block %return {}
// CHECK:STDOUT:   %Make.call: init %D = call %Make.ref() to %.loc6_15
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc14_24.2: <bound method> = bound_method %.loc14_24.4, constants.%C.as.Destroy.impl.Op.518
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.37: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.518, @C.as.Destroy.impl.Op(constants.%int_7.0b1) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.1ed]
// CHECK:STDOUT:   %bound_method.loc14_24.2: <bound method> = bound_method %.loc14_24.4, %C.as.Destroy.impl.Op.specific_fn.37
// CHECK:STDOUT:   %addr.loc14_24.2: %ptr.dc3 = addr_of %.loc14_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc14_24.2: init %empty_tuple.type = call %bound_method.loc14_24.2(%addr.loc14_24.2)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc13_24.3: <bound method> = bound_method %.loc13_24.4, constants.%C.as.Destroy.impl.Op.c43
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.38: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.c43, @C.as.Destroy.impl.Op(constants.%int_6.e56) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.210]
// CHECK:STDOUT:   %bound_method.loc13_24.3: <bound method> = bound_method %.loc13_24.4, %C.as.Destroy.impl.Op.specific_fn.38
// CHECK:STDOUT:   %addr.loc13_24.3: %ptr.b5e = addr_of %.loc13_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc13_24.3: init %empty_tuple.type = call %bound_method.loc13_24.3(%addr.loc13_24.3)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc12_24.4: <bound method> = bound_method %.loc12_24.4, constants.%C.as.Destroy.impl.Op.828
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.39: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.828, @C.as.Destroy.impl.Op(constants.%int_5.0f6) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.295]
// CHECK:STDOUT:   %bound_method.loc12_24.4: <bound method> = bound_method %.loc12_24.4, %C.as.Destroy.impl.Op.specific_fn.39
// CHECK:STDOUT:   %addr.loc12_24.4: %ptr.24c = addr_of %.loc12_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc12_24.4: init %empty_tuple.type = call %bound_method.loc12_24.4(%addr.loc12_24.4)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc11_24.5: <bound method> = bound_method %.loc11_24.4, constants.%C.as.Destroy.impl.Op.d1c
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.40: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.d1c, @C.as.Destroy.impl.Op(constants.%int_4.940) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.12e]
// CHECK:STDOUT:   %bound_method.loc11_24.5: <bound method> = bound_method %.loc11_24.4, %C.as.Destroy.impl.Op.specific_fn.40
// CHECK:STDOUT:   %addr.loc11_24.5: %ptr.c28b = addr_of %.loc11_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc11_24.5: init %empty_tuple.type = call %bound_method.loc11_24.5(%addr.loc11_24.5)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc10_24.6: <bound method> = bound_method %.loc10_24.4, constants.%C.as.Destroy.impl.Op.ee6
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.41: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.ee6, @C.as.Destroy.impl.Op(constants.%int_3.822) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.c71]
// CHECK:STDOUT:   %bound_method.loc10_24.6: <bound method> = bound_method %.loc10_24.4, %C.as.Destroy.impl.Op.specific_fn.41
// CHECK:STDOUT:   %addr.loc10_24.6: %ptr.2b1 = addr_of %.loc10_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc10_24.6: init %empty_tuple.type = call %bound_method.loc10_24.6(%addr.loc10_24.6)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc9_24.7: <bound method> = bound_method %.loc9_24.4, constants.%C.as.Destroy.impl.Op.f2d
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.42: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.f2d, @C.as.Destroy.impl.Op(constants.%int_2.ef8) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.774]
// CHECK:STDOUT:   %bound_method.loc9_24.7: <bound method> = bound_method %.loc9_24.4, %C.as.Destroy.impl.Op.specific_fn.42
// CHECK:STDOUT:   %addr.loc9_24.7: %ptr.3bd = addr_of %.loc9_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc9_24.7: init %empty_tuple.type = call %bound_method.loc9_24.7(%addr.loc9_24.7)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc8_24.8: <bound method> = bound_method %.loc8_24.4, constants.%C.as.Destroy.impl.Op.3c7
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.43: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.3c7, @C.as.Destroy.impl.Op(constants.%int_1.5d2) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.ac5]
// CHECK:STDOUT:   %bound_method.loc8_24.8: <bound method> = bound_method %.loc8_24.4, %C.as.Destroy.impl.Op.specific_fn.43
// CHECK:STDOUT:   %addr.loc8_24.8: %ptr.625 = addr_of %.loc8_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc8_24.8: init %empty_tuple.type = call %bound_method.loc8_24.8(%addr.loc8_24.8)
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound.loc7_24.9: <bound method> = bound_method %.loc7_24.4, constants.%C.as.Destroy.impl.Op.dbb
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn.44: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.dbb, @C.as.Destroy.impl.Op(constants.%int_0.6a9) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn.d03]
// CHECK:STDOUT:   %bound_method.loc7_24.9: <bound method> = bound_method %.loc7_24.4, %C.as.Destroy.impl.Op.specific_fn.44
// CHECK:STDOUT:   %addr.loc7_24.9: %ptr.697 = addr_of %.loc7_24.4
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call.loc7_24.9: init %empty_tuple.type = call %bound_method.loc7_24.9(%addr.loc7_24.9)
// CHECK:STDOUT:   return %Make.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.ImplicitAs.impl.Convert.1 [from "library.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(imports.%P.import_ref.1b7f13.3: %i32) [from "library.carbon"] {
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 0 [symbolic = %N (constants.%N.51e)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%N) [symbolic = %C (constants.%C.17a)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [symbolic = %ptr (constants.%ptr.9e5)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.572)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn = "no_op";
// CHECK:STDOUT: }
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
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%N.51e) {
// CHECK:STDOUT:   %N => constants.%N.51e
// CHECK:STDOUT:   %C => constants.%C.17a
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.0d4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%N.51e) {
// CHECK:STDOUT:   %N => constants.%N.51e
// CHECK:STDOUT:   %C => constants.%C.17a
// CHECK:STDOUT:   %ptr => constants.%ptr.9e5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.572
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%int_0.6a9) {
// CHECK:STDOUT:   %N => constants.%int_0.6a9
// CHECK:STDOUT:   %C => constants.%C.b00
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.c93
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type => constants.%C.as.Destroy.impl.Op.type.615
// CHECK:STDOUT:   %C.as.Destroy.impl.Op => constants.%C.as.Destroy.impl.Op.dbb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%int_0.6a9) {
// CHECK:STDOUT:   %N => constants.%int_0.6a9
// CHECK:STDOUT:   %C => constants.%C.b00
// CHECK:STDOUT:   %ptr => constants.%ptr.697
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7d1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%int_1.5d2) {
// CHECK:STDOUT:   %N => constants.%int_1.5d2
// CHECK:STDOUT:   %C => constants.%C.674
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.b8a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type => constants.%C.as.Destroy.impl.Op.type.b82
// CHECK:STDOUT:   %C.as.Destroy.impl.Op => constants.%C.as.Destroy.impl.Op.3c7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%int_1.5d2) {
// CHECK:STDOUT:   %N => constants.%int_1.5d2
// CHECK:STDOUT:   %C => constants.%C.674
// CHECK:STDOUT:   %ptr => constants.%ptr.625
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.559
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%int_2.ef8) {
// CHECK:STDOUT:   %N => constants.%int_2.ef8
// CHECK:STDOUT:   %C => constants.%C.681
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.979
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type => constants.%C.as.Destroy.impl.Op.type.681
// CHECK:STDOUT:   %C.as.Destroy.impl.Op => constants.%C.as.Destroy.impl.Op.f2d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%int_2.ef8) {
// CHECK:STDOUT:   %N => constants.%int_2.ef8
// CHECK:STDOUT:   %C => constants.%C.681
// CHECK:STDOUT:   %ptr => constants.%ptr.3bd
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%int_3.822) {
// CHECK:STDOUT:   %N => constants.%int_3.822
// CHECK:STDOUT:   %C => constants.%C.7ac
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.dff
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type => constants.%C.as.Destroy.impl.Op.type.f7b
// CHECK:STDOUT:   %C.as.Destroy.impl.Op => constants.%C.as.Destroy.impl.Op.ee6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%int_3.822) {
// CHECK:STDOUT:   %N => constants.%int_3.822
// CHECK:STDOUT:   %C => constants.%C.7ac
// CHECK:STDOUT:   %ptr => constants.%ptr.2b1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%int_4.940) {
// CHECK:STDOUT:   %N => constants.%int_4.940
// CHECK:STDOUT:   %C => constants.%C.89d
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.1d2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type => constants.%C.as.Destroy.impl.Op.type.f38
// CHECK:STDOUT:   %C.as.Destroy.impl.Op => constants.%C.as.Destroy.impl.Op.d1c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%int_4.940) {
// CHECK:STDOUT:   %N => constants.%int_4.940
// CHECK:STDOUT:   %C => constants.%C.89d
// CHECK:STDOUT:   %ptr => constants.%ptr.c28b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%int_5.0f6) {
// CHECK:STDOUT:   %N => constants.%int_5.0f6
// CHECK:STDOUT:   %C => constants.%C.f0a
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.adc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type => constants.%C.as.Destroy.impl.Op.type.743
// CHECK:STDOUT:   %C.as.Destroy.impl.Op => constants.%C.as.Destroy.impl.Op.828
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%int_5.0f6) {
// CHECK:STDOUT:   %N => constants.%int_5.0f6
// CHECK:STDOUT:   %C => constants.%C.f0a
// CHECK:STDOUT:   %ptr => constants.%ptr.24c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.37e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%int_6.e56) {
// CHECK:STDOUT:   %N => constants.%int_6.e56
// CHECK:STDOUT:   %C => constants.%C.c60
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.10e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type => constants.%C.as.Destroy.impl.Op.type.06a
// CHECK:STDOUT:   %C.as.Destroy.impl.Op => constants.%C.as.Destroy.impl.Op.c43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%int_6.e56) {
// CHECK:STDOUT:   %N => constants.%int_6.e56
// CHECK:STDOUT:   %C => constants.%C.c60
// CHECK:STDOUT:   %ptr => constants.%ptr.b5e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.667
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%int_7.0b1) {
// CHECK:STDOUT:   %N => constants.%int_7.0b1
// CHECK:STDOUT:   %C => constants.%C.304
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.65d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type => constants.%C.as.Destroy.impl.Op.type.77c
// CHECK:STDOUT:   %C.as.Destroy.impl.Op => constants.%C.as.Destroy.impl.Op.518
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%int_7.0b1) {
// CHECK:STDOUT:   %N => constants.%int_7.0b1
// CHECK:STDOUT:   %C => constants.%C.304
// CHECK:STDOUT:   %ptr => constants.%ptr.dc3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.12a
// CHECK:STDOUT: }
// CHECK:STDOUT:
