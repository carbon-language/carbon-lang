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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/complete_in_conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/complete_in_conversion.carbon

// --- fail_derived_to_base.carbon

fn Int(N: Core.IntLiteral()) -> type = "int.make_type_signed";

base class B {}

class A(N:! i32) {
  extend base: B;

  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+3]]:10: error: integer type width of 0 is not positive [IntWidthNotPositive]
  // CHECK:STDERR:   var n: Int(N);
  // CHECK:STDERR:          ^~~~~~
  var n: Int(N);
}

fn F(a: A(0)*) {
  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+4]]:15: note: in `A(0)` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   let b: B* = a;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  let b: B* = a;
}

// CHECK:STDOUT: --- fail_derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %Int.type.913: type = fn_type @Int.loc2 [concrete]
// CHECK:STDOUT:   %Int.779: %Int.type.913 = struct_value () [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type.878: type = generic_class_type @Int.1 [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type.878 = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int.1, @Int.1(%int_32) [concrete]
// CHECK:STDOUT:   %N.fe8: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.496: type = class_type @A, @A(%N.fe8) [symbolic]
// CHECK:STDOUT:   %A.elem.57d: type = unbound_element_type %A.496, %B [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7a9: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2c2: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.cdf: %Int.as.ImplicitAs.impl.Convert.type.2c2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.972: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.783, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.592: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.047: %Int.as.ImplicitAs.impl.Convert.type.592 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.64a: %ImplicitAs.type.7a9 = facet_value %i32, (%ImplicitAs.impl_witness.972) [concrete]
// CHECK:STDOUT:   %.63b: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.64a [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.0e0: <bound method> = bound_method %N.fe8, %Int.as.ImplicitAs.impl.Convert.047 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.047, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.490: <bound method> = bound_method %N.fe8, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.490(%N.fe8) [symbolic]
// CHECK:STDOUT:   %iN.builtin.3f2: type = int_type signed, %Int.as.ImplicitAs.impl.Convert.call [symbolic]
// CHECK:STDOUT:   %require_complete.3d7: <witness> = require_complete_type %iN.builtin.3f2 [symbolic]
// CHECK:STDOUT:   %A.elem.1ed: type = unbound_element_type %A.496, %iN.builtin.3f2 [symbolic]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: %iN.builtin.3f2} [symbolic]
// CHECK:STDOUT:   %complete_type.8ef: <witness> = complete_type_witness %struct_type.base.n [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.3ad: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet.3ad [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9be: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %A.6fc: type = class_type @A, @A(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %ptr.b65: type = ptr_type %A.6fc [concrete]
// CHECK:STDOUT:   %pattern_type.213: type = pattern_type %ptr.b65 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %A.elem.d81: type = unbound_element_type %A.6fc, %B [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.c51: <bound method> = bound_method %int_0.6a9, %Int.as.ImplicitAs.impl.Convert.047 [concrete]
// CHECK:STDOUT:   %bound_method.27e: <bound method> = bound_method %int_0.6a9, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type.878 = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.a86: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2c2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.cdf)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.783 = impl_witness_table (%Core.import_ref.a86), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Int.decl: %Int.type.913 = fn_decl @Int.loc2 [concrete = constants.%Int.779] {
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = value_binding_pattern N [concrete]
// CHECK:STDOUT:     %N.param_patt: %pattern_type.dc0 = value_param_pattern %N.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc2_27.1: type = splice_block %.loc2_27.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc2_27.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc2_27.3: type = converted %IntLiteral.call, %.loc2_27.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = value_binding N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %N.patt: %pattern_type.7ce = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int.1, @Int.1(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_9.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_9.1 (constants.%N.fe8)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %pattern_type.213 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.213 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.b65 = value_param call_param0
// CHECK:STDOUT:     %.loc15_13: type = splice_block %ptr.loc15 [concrete = constants.%ptr.b65] {
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:       %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:       %bound_method.loc15_12.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc15_12.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.9be]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_12.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:       %.loc15_12.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:       %.loc15_12.2: %i32 = converted %int_0, %.loc15_12.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:       %A: type = class_type @A, @A(constants.%int_0.6a9) [concrete = constants.%A.6fc]
// CHECK:STDOUT:       %ptr.loc15: type = ptr_type %A [concrete = constants.%ptr.b65]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.b65 = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .Int = <poisoned>
// CHECK:STDOUT:   .N = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%N.loc6_9.2: %i32) {
// CHECK:STDOUT:   %N.loc6_9.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_9.1 (constants.%N.fe8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A: type = class_type @A, @A(%N.loc6_9.1) [symbolic = %A (constants.%A.496)]
// CHECK:STDOUT:   %A.elem.loc7: type = unbound_element_type %A, constants.%B [symbolic = %A.elem.loc7 (constants.%A.elem.57d)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc6_9.1, constants.%Int.as.ImplicitAs.impl.Convert.047 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound.0e0)]
// CHECK:STDOUT:   %bound_method.loc12_14.3: <bound method> = bound_method %N.loc6_9.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc12_14.3 (constants.%bound_method.490)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc12_14.2: init Core.IntLiteral = call %bound_method.loc12_14.3(%N.loc6_9.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc12_14.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %Int.as.ImplicitAs.impl.Convert.call.loc12_14.2 [symbolic = %iN.builtin (constants.%iN.builtin.3f2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %iN.builtin [symbolic = %require_complete (constants.%require_complete.3d7)]
// CHECK:STDOUT:   %A.elem.loc12: type = unbound_element_type %A, %iN.builtin [symbolic = %A.elem.loc12 (constants.%A.elem.1ed)]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: @A.%iN.builtin (%iN.builtin.3f2)} [symbolic = %struct_type.base.n (constants.%struct_type.base.n)]
// CHECK:STDOUT:   %complete_type.loc13_1.2: <witness> = complete_type_witness %struct_type.base.n [symbolic = %complete_type.loc13_1.2 (constants.%complete_type.8ef)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc7: @A.%A.elem.loc7 (%A.elem.57d) = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:     %Int.ref: %Int.type.913 = name_ref Int, file.%Int.decl [concrete = constants.%Int.779]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %N.loc6_9.2 [symbolic = %N.loc6_9.1 (constants.%N.fe8)]
// CHECK:STDOUT:     %impl.elem0: %.63b = impl_witness_access constants.%ImplicitAs.impl_witness.972, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.047]
// CHECK:STDOUT:     %bound_method.loc12_14.1: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound.0e0)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc12_14.2: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc12_14.3 (constants.%bound_method.490)]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call.loc12_14.1: init Core.IntLiteral = call %bound_method.loc12_14.2(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc12_14.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc12_14.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc12_14.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc12_14.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc12_14.2: Core.IntLiteral = converted %N.ref, %.loc12_14.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc12_14.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %Int.call: init type = call %Int.ref(%.loc12_14.2) [symbolic = %iN.builtin (constants.%iN.builtin.3f2)]
// CHECK:STDOUT:     %.loc12_15.1: type = value_of_initializer %Int.call [symbolic = %iN.builtin (constants.%iN.builtin.3f2)]
// CHECK:STDOUT:     %.loc12_15.2: type = converted %Int.call, %.loc12_15.1 [symbolic = %iN.builtin (constants.%iN.builtin.3f2)]
// CHECK:STDOUT:     %.loc12_8: @A.%A.elem.loc12 (%A.elem.1ed) = field_decl n, element1 [concrete]
// CHECK:STDOUT:     %complete_type.loc13_1.1: <witness> = complete_type_witness constants.%struct_type.base.n [symbolic = %complete_type.loc13_1.2 (constants.%complete_type.8ef)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc13_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.496
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     .base = %.loc7
// CHECK:STDOUT:     .Int = <poisoned>
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .n = %.loc12_8
// CHECK:STDOUT:     extend %B.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int.loc2(%N.param: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %ptr.b65) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.960 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref: %ptr.b65 = name_ref a, %a
// CHECK:STDOUT:   %.loc20_11: type = splice_block %ptr.loc20 [concrete = constants.%ptr.e79] {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc20: type = ptr_type %B.ref [concrete = constants.%ptr.e79]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20_15.1: ref %A.6fc = deref %a.ref
// CHECK:STDOUT:   %.loc20_15.2: ref %B = class_element_access %.loc20_15.1, element0
// CHECK:STDOUT:   %addr: %ptr.e79 = addr_of %.loc20_15.2
// CHECK:STDOUT:   %.loc20_15.3: %ptr.e79 = converted %a.ref, %addr
// CHECK:STDOUT:   %b: %ptr.e79 = value_binding b, %.loc20_15.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%N.fe8) {
// CHECK:STDOUT:   %N.loc6_9.1 => constants.%N.fe8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%int_0.6a9) {
// CHECK:STDOUT:   %N.loc6_9.1 => constants.%int_0.6a9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.6fc
// CHECK:STDOUT:   %A.elem.loc7 => constants.%A.elem.d81
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound.c51
// CHECK:STDOUT:   %bound_method.loc12_14.3 => constants.%bound_method.27e
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc12_14.2 => constants.%int_0.5c6
// CHECK:STDOUT:   %iN.builtin => <error>
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %A.elem.loc12 => <error>
// CHECK:STDOUT:   %struct_type.base.n => <error>
// CHECK:STDOUT:   %complete_type.loc13_1.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
