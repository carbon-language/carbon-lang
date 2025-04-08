// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %Int.type.913: type = fn_type @Int.1 [concrete]
// CHECK:STDOUT:   %Int.779: %Int.type.913 = struct_value () [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int.2, @Int.2(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.8e2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.dd3: type = class_type @A, @A(%N.51e) [symbolic]
// CHECK:STDOUT:   %A.elem.500: type = unbound_element_type %A.dd3, %B [symbolic]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %Convert.bound.588: <bound method> = bound_method %N.51e, %Convert.960 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.8a8: <specific function> = specific_function %Convert.960, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.f9c: <bound method> = bound_method %N.51e, %Convert.specific_fn.8a8 [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %bound_method.f9c(%N.51e) [symbolic]
// CHECK:STDOUT:   %iN.builtin.8fe: type = int_type signed, %int.convert_checked [symbolic]
// CHECK:STDOUT:   %require_complete.e34: <witness> = require_complete_type %iN.builtin.8fe [symbolic]
// CHECK:STDOUT:   %A.elem.07f: type = unbound_element_type %A.dd3, %iN.builtin.8fe [symbolic]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: %iN.builtin.8fe} [symbolic]
// CHECK:STDOUT:   %complete_type.beb: <witness> = complete_type_witness %struct_type.base.n [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.2b8: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.2b8 [concrete]
// CHECK:STDOUT:   %Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.b6f: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn.b6f [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %A.6fc: type = class_type @A, @A(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %ptr.b65: type = ptr_type %A.6fc [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %A.elem.d81: type = unbound_element_type %A.6fc, %B [concrete]
// CHECK:STDOUT:   %Convert.bound.0fd: <bound method> = bound_method %int_0.6a9, %Convert.960 [concrete]
// CHECK:STDOUT:   %bound_method.4b5: <bound method> = bound_method %int_0.6a9, %Convert.specific_fn.8a8 [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
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
// CHECK:STDOUT:   %Int.decl: %Int.type.913 = fn_decl @Int.1 [concrete = constants.%Int.779] {
// CHECK:STDOUT:     %N.patt: Core.IntLiteral = binding_pattern N
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt, call_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc2_27.1: type = splice_block %.loc2_27.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc2_27.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc2_27.3: type = converted %int_literal.make_type, %.loc2_27.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %N.patt.loc6_9.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int.2, @Int.2(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_9.1: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.2 (constants.%N.51e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %ptr.b65 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %ptr.b65 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.b65 = value_param call_param0
// CHECK:STDOUT:     %.loc15_13: type = splice_block %ptr.loc15 [concrete = constants.%ptr.b65] {
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:       %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:       %bound_method.loc15_12.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.b6f]
// CHECK:STDOUT:       %bound_method.loc15_12.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:       %int.convert_checked: init %i32 = call %bound_method.loc15_12.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:       %.loc15_12.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:       %.loc15_12.2: %i32 = converted %int_0, %.loc15_12.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:       %A: type = class_type @A, @A(constants.%int_0.6a9) [concrete = constants.%A.6fc]
// CHECK:STDOUT:       %ptr.loc15: type = ptr_type %A [concrete = constants.%ptr.b65]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.b65 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .Int = <poisoned>
// CHECK:STDOUT:   .N = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%N.loc6_9.1: %i32) {
// CHECK:STDOUT:   %N.loc6_9.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.2 (constants.%N.51e)]
// CHECK:STDOUT:   %N.patt.loc6_9.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A: type = class_type @A, @A(%N.loc6_9.2) [symbolic = %A (constants.%A.dd3)]
// CHECK:STDOUT:   %A.elem.loc7: type = unbound_element_type %A, constants.%B [symbolic = %A.elem.loc7 (constants.%A.elem.500)]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.loc6_9.2, constants.%Convert.960 [symbolic = %Convert.bound (constants.%Convert.bound.588)]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.loc6_9.2, constants.%Convert.specific_fn.8a8 [symbolic = %bound_method (constants.%bound_method.f9c)]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %bound_method(%N.loc6_9.2) [symbolic = %int.convert_checked (constants.%int.convert_checked)]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %int.convert_checked [symbolic = %iN.builtin (constants.%iN.builtin.8fe)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @A.%iN.builtin (%iN.builtin.8fe) [symbolic = %require_complete (constants.%require_complete.e34)]
// CHECK:STDOUT:   %A.elem.loc12: type = unbound_element_type %A, %iN.builtin [symbolic = %A.elem.loc12 (constants.%A.elem.07f)]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: @A.%iN.builtin (%iN.builtin.8fe)} [symbolic = %struct_type.base.n (constants.%struct_type.base.n)]
// CHECK:STDOUT:   %complete_type.loc13_1.2: <witness> = complete_type_witness @A.%struct_type.base.n (%struct_type.base.n) [symbolic = %complete_type.loc13_1.2 (constants.%complete_type.beb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc7: @A.%A.elem.loc7 (%A.elem.500) = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:     %.loc12_8: @A.%A.elem.loc12 (%A.elem.07f) = field_decl n, element1 [concrete]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %.loc12_3: @A.%A.elem.loc12 (%A.elem.07f) = var_pattern %.loc12_8
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.var: ref @A.%A.elem.loc12 (%A.elem.07f) = var <none>
// CHECK:STDOUT:     %complete_type.loc13_1.1: <witness> = complete_type_witness %struct_type.base.n [symbolic = %complete_type.loc13_1.2 (constants.%complete_type.beb)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc13_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.dd3
// CHECK:STDOUT:     .B = <poisoned>
// CHECK:STDOUT:     .base = %.loc7
// CHECK:STDOUT:     .Int = <poisoned>
// CHECK:STDOUT:     .N = <poisoned>
// CHECK:STDOUT:     .n = %.loc12_8
// CHECK:STDOUT:     extend %B.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int.1(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %ptr.b65) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %ptr.e79 = binding_pattern b
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
// CHECK:STDOUT:   %b: %ptr.e79 = bind_name b, %.loc20_15.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%N.51e
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%N.patt.8e2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(%N.loc6_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%int_0.6a9) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_0.6a9
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%N.patt.8e2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.6fc
// CHECK:STDOUT:   %A.elem.loc7 => constants.%A.elem.d81
// CHECK:STDOUT:   %Convert.bound => constants.%Convert.bound.0fd
// CHECK:STDOUT:   %bound_method => constants.%bound_method.4b5
// CHECK:STDOUT:   %int.convert_checked => constants.%int_0.5c6
// CHECK:STDOUT:   %iN.builtin => <error>
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %A.elem.loc12 => <error>
// CHECK:STDOUT:   %struct_type.base.n => <error>
// CHECK:STDOUT:   %complete_type.loc13_1.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
