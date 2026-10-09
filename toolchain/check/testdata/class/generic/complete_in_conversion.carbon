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
  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+7]]:22: error: unable to monomorphize specific `A(0)` [ResolvingSpecificHere]
  // CHECK:STDERR:   let unused b: B* = a;
  // CHECK:STDERR:                      ^
  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE-7]]:10: note: integer type width of 0 is not positive [IntWidthNotPositive]
  // CHECK:STDERR:   var n: Int(N);
  // CHECK:STDERR:          ^~~~~~
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
// CHECK:STDOUT:   %Int.type.6c1: type = generic_class_type @Int.1 [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type.6c1 = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int.1, @Int.1(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.b2d: %pattern_type.9d9 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.5d3: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.acd: type = class_type @A, @A(%N.5d3) [symbolic]
// CHECK:STDOUT:   %A.elem.a05: type = unbound_element_type %A.acd, %B [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.500: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bfc: %Int.as.ImplicitAs.impl.Convert.type.500 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.803: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.6ce, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.de1: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.76f: %Int.as.ImplicitAs.impl.Convert.type.de1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eb0: %ImplicitAs.type.5be = facet_value %i32, (%ImplicitAs.impl_witness.803) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.797: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.eb0) [concrete]
// CHECK:STDOUT:   %.917: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.797, %ImplicitAs.facet.eb0 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.cc4: <bound method> = bound_method %N.5d3, %Int.as.ImplicitAs.impl.Convert.76f [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.76f, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.210: <bound method> = bound_method %N.5d3, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.210(%N.5d3) [symbolic]
// CHECK:STDOUT:   %iN.builtin.57d: type = int_type signed, %Int.as.ImplicitAs.impl.Convert.call [symbolic]
// CHECK:STDOUT:   %require_complete.2a8: <witness> = require_complete_type %iN.builtin.57d [symbolic]
// CHECK:STDOUT:   %A.elem.205: type = unbound_element_type %A.acd, %iN.builtin.57d [symbolic]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: %iN.builtin.57d} [symbolic]
// CHECK:STDOUT:   %complete_type.657: <witness> = complete_type_witness %struct_type.base.n [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.09b: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ec2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eeb: %ImplicitAs.type.09b = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ec2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.eeb) [concrete]
// CHECK:STDOUT:   %.f0f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e04, %ImplicitAs.facet.eeb [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.614: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.2b2: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %A.d5b: type = class_type @A, @A(%int_0.2b2) [concrete]
// CHECK:STDOUT:   %ptr.0aa: type = ptr_type %A.d5b [concrete]
// CHECK:STDOUT:   %pattern_type.2ba: type = pattern_type %ptr.0aa [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.2ba = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.2ba = wrapper_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.432: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.3c8: type = pattern_type %ptr.432 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.3c8 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %A.elem.7aa: type = unbound_element_type %A.d5b, %B [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.0bd: <bound method> = bound_method %int_0.2b2, %Int.as.ImplicitAs.impl.Convert.76f [concrete]
// CHECK:STDOUT:   %bound_method.38f: <bound method> = bound_method %int_0.2b2, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type.6c1 = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.5b7: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.500) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.bfc)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.6ce = impl_witness_table (%Core.import_ref.5b7), @Int.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %N.patt.loc6_10.1: %pattern_type.9d9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_10.2 (constants.%N.patt.b2d)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_10.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_10.1 (constants.%N.5d3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.2ba = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.2ba = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.0aa = value_param call_param0
// CHECK:STDOUT:     %.loc12_13: type = splice_block %ptr.loc12 [concrete = constants.%ptr.0aa] {
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:       %impl.elem0: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:       %bound_method.loc12_12.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc12_12.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.614]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_12.2(%int_0) [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:       %.loc12_12.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:       %.loc12_12.2: %i32 = converted %int_0, %.loc12_12.1 [concrete = constants.%int_0.2b2]
// CHECK:STDOUT:       %A: type = class_type @A, @A(constants.%int_0.2b2) [concrete = constants.%A.d5b]
// CHECK:STDOUT:       %ptr.loc12: type = ptr_type %A [concrete = constants.%ptr.0aa]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.0aa = wrapper_binding a, %a.param
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
// CHECK:STDOUT:   %N.patt.loc6_10.2: %pattern_type.9d9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_10.2 (constants.%N.patt.b2d)]
// CHECK:STDOUT:   %N.loc6_10.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_10.1 (constants.%N.5d3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A: type = class_type @A, @A(%N.loc6_10.1) [symbolic = %A (constants.%A.acd)]
// CHECK:STDOUT:   %A.elem.loc7: type = unbound_element_type %A, constants.%B [symbolic = %A.elem.loc7 (constants.%A.elem.a05)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc6_10.1, constants.%Int.as.ImplicitAs.impl.Convert.76f [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound.cc4)]
// CHECK:STDOUT:   %bound_method.loc9_14.3: <bound method> = bound_method %N.loc6_10.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc9_14.3 (constants.%bound_method.210)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc9_14.2: init Core.IntLiteral = call %bound_method.loc9_14.3(%N.loc6_10.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc9_14.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %Int.as.ImplicitAs.impl.Convert.call.loc9_14.2 [symbolic = %iN.builtin (constants.%iN.builtin.57d)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %iN.builtin [symbolic = %require_complete (constants.%require_complete.2a8)]
// CHECK:STDOUT:   %A.elem.loc9: type = unbound_element_type %A, %iN.builtin [symbolic = %A.elem.loc9 (constants.%A.elem.205)]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: @A.%iN.builtin (%iN.builtin.57d)} [symbolic = %struct_type.base.n (constants.%struct_type.base.n)]
// CHECK:STDOUT:   %complete_type.loc10_1.2: <witness> = complete_type_witness %struct_type.base.n [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.657)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc7: @A.%A.elem.loc7 (%A.elem.a05) = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:     %field_decl: @A.%A.elem.loc9 (%A.elem.205) = field_decl n, element1, %.loc9_15.2 in [concrete] {
// CHECK:STDOUT:       %Int.ref: %Int.type.9c0 = name_ref Int, file.%Int.decl [concrete = constants.%Int.c4a]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc6_10.2 [symbolic = %N.loc6_10.1 (constants.%N.5d3)]
// CHECK:STDOUT:       %impl.elem0: %.917 = impl_witness_access constants.%ImplicitAs.impl_witness.803, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.76f]
// CHECK:STDOUT:       %bound_method.loc9_14.1: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound.cc4)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc9_14.2: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc9_14.3 (constants.%bound_method.210)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc9_14.1: init Core.IntLiteral = call %bound_method.loc9_14.2(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc9_14.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc9_14.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc9_14.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc9_14.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc9_14.2: Core.IntLiteral = converted %N.ref, %.loc9_14.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc9_14.2 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %Int.call: init type = call %Int.ref(%.loc9_14.2) [symbolic = %iN.builtin (constants.%iN.builtin.57d)]
// CHECK:STDOUT:       %.loc9_15.1: type = value_of_initializer %Int.call [symbolic = %iN.builtin (constants.%iN.builtin.57d)]
// CHECK:STDOUT:       %.loc9_15.2: type = converted %Int.call, %.loc9_15.1 [symbolic = %iN.builtin (constants.%iN.builtin.57d)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type.loc10_1.1: <witness> = complete_type_witness constants.%struct_type.base.n [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.657)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc10_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.acd
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
// CHECK:STDOUT: fn @F(%a.param: %ptr.0aa) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.0aa = name_ref a, %a
// CHECK:STDOUT:   %.loc20_22.1: ref %A.d5b = deref %a.ref
// CHECK:STDOUT:   %.loc20_22.2: ref %B = class_element_access %.loc20_22.1, element0
// CHECK:STDOUT:   %addr: %ptr.432 = addr_of %.loc20_22.2
// CHECK:STDOUT:   %.loc20_22.3: %ptr.432 = converted %a.ref, %addr
// CHECK:STDOUT:   %.loc20_18: type = splice_block %ptr.loc20 [concrete = constants.%ptr.432] {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc20: type = ptr_type %B.ref [concrete = constants.%ptr.432]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %ptr.432 = wrapper_binding b, %.loc20_22.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.3c8 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%N.5d3) {
// CHECK:STDOUT:   %N.patt.loc6_10.2 => constants.%N.patt.b2d
// CHECK:STDOUT:   %N.loc6_10.1 => constants.%N.5d3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%int_0.2b2) {
// CHECK:STDOUT:   %N.patt.loc6_10.2 => constants.%N.patt.b2d
// CHECK:STDOUT:   %N.loc6_10.1 => constants.%int_0.2b2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.d5b
// CHECK:STDOUT:   %A.elem.loc7 => constants.%A.elem.7aa
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound.0bd
// CHECK:STDOUT:   %bound_method.loc9_14.3 => constants.%bound_method.38f
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc9_14.2 => constants.%int_0.5c6
// CHECK:STDOUT:   %iN.builtin => <error>
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %A.elem.loc9 => <error>
// CHECK:STDOUT:   %struct_type.base.n => <error>
// CHECK:STDOUT:   %complete_type.loc10_1.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
