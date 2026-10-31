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

fn Int(N: Core.IntLiteral) -> type = "int.make_type_signed";

base class B {}

class A(N: i32) {
  extend base: B;

  var n: Int(N);
}

fn F(a: A(0)*) {
  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+3]]:22: error: unable to monomorphize specific `A(0)` [ResolvingSpecificHere]
  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE-5]]:10-15: note: integer type width of 0 is not positive [IntWidthNotPositive]
  // CHECK:STDERR:
  let unused b: B* = a;
}

// CHECK:STDOUT: --- fail_derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N.param_patt: %pattern_type.dc0 = value_param_pattern [concrete]
// CHECK:STDOUT:   %N.patt.4a4: %pattern_type.dc0 = wrapper_binding_pattern N, %N.param_patt [concrete]
// CHECK:STDOUT:   %.576: Core.Form = init_form type [concrete]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %return.param_patt.90b: %pattern_type.9a5 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.718: %pattern_type.9a5 = return_slot_pattern %return.param_patt.90b, type [concrete]
// CHECK:STDOUT:   %Int.type.9c0: type = fn_type @Int.loc2 [concrete]
// CHECK:STDOUT:   %Int.c4a: %Int.type.9c0 = struct_value () [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type.cb4: type = parameterized_class_name_type @Int.1 [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type.cb4 = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int.1, @Int.1(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.d64: %pattern_type.d94 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.c34: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = parameterized_class_name_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.2ee: type = class_type @A, @A(%N.c34) [symbolic]
// CHECK:STDOUT:   %A.elem.1fa: type = unbound_element_type %A.2ee, %B [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.dfd: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.914: %Int.as.ImplicitAs.impl.Convert.type.dfd = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.323: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b7c, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.0aa: %ImplicitAs.type.5be = facet_value %i32, (%ImplicitAs.impl_witness.323) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.4c7: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.0aa) [concrete]
// CHECK:STDOUT:   %.ce7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.4c7, %ImplicitAs.facet.0aa [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.f39: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.779: %Int.as.ImplicitAs.impl.Convert.type.f39 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.de4: <bound method> = bound_method %N.c34, %Int.as.ImplicitAs.impl.Convert.779 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.779, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.2e8: <bound method> = bound_method %N.c34, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.2e8(%N.c34) [symbolic]
// CHECK:STDOUT:   %iN.builtin.2cb: type = int_type signed, %Int.as.ImplicitAs.impl.Convert.call [symbolic]
// CHECK:STDOUT:   %require_complete.23a: <witness> = require_complete_type %iN.builtin.2cb [symbolic]
// CHECK:STDOUT:   %A.elem.b93: type = unbound_element_type %A.2ee, %iN.builtin.2cb [symbolic]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: %iN.builtin.2cb} [symbolic]
// CHECK:STDOUT:   %complete_type.642: <witness> = complete_type_witness %struct_type.base.n [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fc6: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.fc6) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet.fc6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.036: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %A.10a: type = class_type @A, @A(%int_0.753) [concrete]
// CHECK:STDOUT:   %ptr.0b6: type = ptr_type %A.10a [concrete]
// CHECK:STDOUT:   %pattern_type.ada: type = pattern_type %ptr.0b6 [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.ada = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.ada = wrapper_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.432: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.3c8: type = pattern_type %ptr.432 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.3c8 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %A.elem.093: type = unbound_element_type %A.10a, %B [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.741: <bound method> = bound_method %int_0.753, %Int.as.ImplicitAs.impl.Convert.779 [concrete]
// CHECK:STDOUT:   %bound_method.ae8: <bound method> = bound_method %int_0.753, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral: type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type.cb4 = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.ed0: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.dfd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.914)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b7c = impl_witness_table (%Core.import_ref.ed0), @Int.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %Int.decl: %Int.type.9c0 = fn_decl @Int.loc2 [concrete = constants.%Int.c4a] {
// CHECK:STDOUT:     %N.param_patt: %pattern_type.dc0 = value_param_pattern [concrete = constants.%N.param_patt]
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = wrapper_binding_pattern N, %N.param_patt [concrete = constants.%N.patt.4a4]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9a5 = out_param_pattern [concrete = constants.%return.param_patt.90b]
// CHECK:STDOUT:     %return.patt: %pattern_type.9a5 = return_slot_pattern %return.param_patt, %.loc2_31.1 [concrete = constants.%return.patt.718]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc2_31.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc2_31.2: Core.Form = init_form %.loc2_31.1 [concrete = constants.%.576]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc2_15: type = splice_block %IntLiteral.ref [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = wrapper_binding N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %N.patt.loc6_10.1: %pattern_type.d94 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_10.2 (constants.%N.patt.d64)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_10.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_10.1 (constants.%N.c34)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.ada = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.ada = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.0b6 = value_param call_param0
// CHECK:STDOUT:     %.loc12_13: type = splice_block %ptr.loc12 [concrete = constants.%ptr.0b6] {
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:       %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:       %bound_method.loc12_12.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc12_12.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.036]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_12.2(%int_0) [concrete = constants.%int_0.753]
// CHECK:STDOUT:       %.loc12_12.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.753]
// CHECK:STDOUT:       %.loc12_12.2: %i32 = converted %int_0, %.loc12_12.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:       %A: type = class_type @A, @A(constants.%int_0.753) [concrete = constants.%A.10a]
// CHECK:STDOUT:       %ptr.loc12: type = ptr_type %A [concrete = constants.%ptr.0b6]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.0b6 = wrapper_binding a, %a.param
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
// CHECK:STDOUT: generic class @A(%N.loc6_10.2: %i32) {
// CHECK:STDOUT:   %N.patt.loc6_10.2: %pattern_type.d94 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_10.2 (constants.%N.patt.d64)]
// CHECK:STDOUT:   %N.loc6_10.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_10.1 (constants.%N.c34)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A: type = class_type @A, @A(%N.loc6_10.1) [symbolic = %A (constants.%A.2ee)]
// CHECK:STDOUT:   %A.elem.loc7: type = unbound_element_type %A, constants.%B [symbolic = %A.elem.loc7 (constants.%A.elem.1fa)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc6_10.1, constants.%Int.as.ImplicitAs.impl.Convert.779 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound.de4)]
// CHECK:STDOUT:   %bound_method.loc9_14.3: <bound method> = bound_method %N.loc6_10.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc9_14.3 (constants.%bound_method.2e8)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc9_14.2: init Core.IntLiteral = call %bound_method.loc9_14.3(%N.loc6_10.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc9_14.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %Int.as.ImplicitAs.impl.Convert.call.loc9_14.2 [symbolic = %iN.builtin (constants.%iN.builtin.2cb)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %iN.builtin [symbolic = %require_complete (constants.%require_complete.23a)]
// CHECK:STDOUT:   %A.elem.loc9: type = unbound_element_type %A, %iN.builtin [symbolic = %A.elem.loc9 (constants.%A.elem.b93)]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: @A.%iN.builtin (%iN.builtin.2cb)} [symbolic = %struct_type.base.n (constants.%struct_type.base.n)]
// CHECK:STDOUT:   %complete_type.loc10_1.2: <witness> = complete_type_witness %struct_type.base.n [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.642)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc7: @A.%A.elem.loc7 (%A.elem.1fa) = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:     %field_decl: @A.%A.elem.loc9 (%A.elem.b93) = field_decl n, element1, %.loc9_15.2 in [concrete] {
// CHECK:STDOUT:       %Int.ref: %Int.type.9c0 = name_ref Int, file.%Int.decl [concrete = constants.%Int.c4a]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc6_10.2 [symbolic = %N.loc6_10.1 (constants.%N.c34)]
// CHECK:STDOUT:       %impl.elem0: %.ce7 = impl_witness_access constants.%ImplicitAs.impl_witness.323, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.779]
// CHECK:STDOUT:       %bound_method.loc9_14.1: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound.de4)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc9_14.2: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc9_14.3 (constants.%bound_method.2e8)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc9_14.1: init Core.IntLiteral = call %bound_method.loc9_14.2(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc9_14.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc9_14.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc9_14.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc9_14.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc9_14.2: Core.IntLiteral = converted %N.ref, %.loc9_14.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc9_14.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %Int.call: init type = call %Int.ref(%.loc9_14.2) [symbolic = %iN.builtin (constants.%iN.builtin.2cb)]
// CHECK:STDOUT:       %.loc9_15.1: type = value_of_initializer %Int.call [symbolic = %iN.builtin (constants.%iN.builtin.2cb)]
// CHECK:STDOUT:       %.loc9_15.2: type = converted %Int.call, %.loc9_15.1 [symbolic = %iN.builtin (constants.%iN.builtin.2cb)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type.loc10_1.1: <witness> = complete_type_witness constants.%struct_type.base.n [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.642)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc10_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.2ee
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     .base = %.loc7
// CHECK:STDOUT:     .Int = <poisoned>
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .n = %field_decl
// CHECK:STDOUT:     extend %B.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int.loc2(%N.param: Core.IntLiteral) -> out %return.param: type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %ptr.0b6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.0b6 = name_ref a, %a
// CHECK:STDOUT:   %.loc16_22.1: ref %A.10a = deref %a.ref
// CHECK:STDOUT:   %.loc16_22.2: ref %B = class_element_access %.loc16_22.1, element0
// CHECK:STDOUT:   %addr: %ptr.432 = addr_of %.loc16_22.2
// CHECK:STDOUT:   %.loc16_22.3: %ptr.432 = converted %a.ref, %addr
// CHECK:STDOUT:   %.loc16_18: type = splice_block %ptr.loc16 [concrete = constants.%ptr.432] {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc16: type = ptr_type %B.ref [concrete = constants.%ptr.432]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.432 = wrapper_binding b, %.loc16_22.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.3c8 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%N.c34) {
// CHECK:STDOUT:   %N.patt.loc6_10.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc6_10.1 => constants.%N.c34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%int_0.753) {
// CHECK:STDOUT:   %N.patt.loc6_10.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc6_10.1 => constants.%int_0.753
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.10a
// CHECK:STDOUT:   %A.elem.loc7 => constants.%A.elem.093
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound.741
// CHECK:STDOUT:   %bound_method.loc9_14.3 => constants.%bound_method.ae8
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc9_14.2 => constants.%int_0.5c6
// CHECK:STDOUT:   %iN.builtin => <error>
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %A.elem.loc9 => <error>
// CHECK:STDOUT:   %struct_type.base.n => <error>
// CHECK:STDOUT:   %complete_type.loc10_1.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
