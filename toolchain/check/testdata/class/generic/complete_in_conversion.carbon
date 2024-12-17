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
  // CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+3]]:3: note: in `A(0)` used here [ResolvingSpecificHere]
  // CHECK:STDERR:   let b: B* = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  let b: B* = a;
}

// CHECK:STDOUT: --- fail_derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Int.type.1: type = fn_type @Int.1 [template]
// CHECK:STDOUT:   %Int.1: %Int.type.1 = struct_value () [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int.2, @Int.2(%int_32) [template]
// CHECK:STDOUT:   %N.2: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [template]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [template]
// CHECK:STDOUT:   %A.1: type = class_type @A, @A(%N.2) [symbolic]
// CHECK:STDOUT:   %A.elem.1: type = unbound_element_type %A.1, %B [symbolic]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.3, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %N.2, %Convert.9 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.3(%int_32) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn.1(%N.2) [symbolic]
// CHECK:STDOUT:   %iN.builtin.2: type = int_type signed, %int.convert_checked [symbolic]
// CHECK:STDOUT:   %require_complete.5: <witness> = require_complete_type %iN.builtin.2 [symbolic]
// CHECK:STDOUT:   %A.elem.2: type = unbound_element_type %A.1, %iN.builtin.2 [symbolic]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: %iN.builtin.2} [symbolic]
// CHECK:STDOUT:   %complete_type.4: <witness> = complete_type_witness %struct_type.base.n [symbolic]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [template]
// CHECK:STDOUT:   %interface.6: <witness> = interface_witness (%Convert.11) [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_0.1, %Convert.11 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %A.2: type = class_type @A, @A(%int_0.2) [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %A.2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %ptr.3: type = ptr_type %B [template]
// CHECK:STDOUT:   %A.elem.3: type = unbound_element_type %A.2, %B [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_0.2, %Convert.9 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.3(%int_32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Int.decl: %Int.type.1 = fn_decl @Int.1 [template = constants.%Int.1] {
// CHECK:STDOUT:     %N.patt: Core.IntLiteral = binding_pattern N
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc2_27.3: type = splice_block %.loc2_27.2 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc2_27.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc2_27.2: type = converted %int_literal.make_type, %.loc2_27.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [template = constants.%A.generic] {
// CHECK:STDOUT:     %N.patt.loc6_9.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc6_9.1, runtime_param<invalid> [symbolic = %N.patt.loc6_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int.2, @Int.2(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_9.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_9.2 (constants.%N.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt: %ptr.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %ptr.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %.loc15_13: type = splice_block %ptr [template = constants.%ptr.2] {
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A.generic]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:       %impl.elem0: %Convert.type.10 = interface_witness_access constants.%interface.6, element0 [template = constants.%Convert.11]
// CHECK:STDOUT:       %Convert.bound: <bound method> = bound_method %int_0, %impl.elem0 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:       %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:       %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_0) [template = constants.%int_0.2]
// CHECK:STDOUT:       %.loc15_12.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_0.2]
// CHECK:STDOUT:       %.loc15_12.2: %i32 = converted %int_0, %.loc15_12.1 [template = constants.%int_0.2]
// CHECK:STDOUT:       %A: type = class_type @A, @A(constants.%int_0.2) [template = constants.%A.2]
// CHECK:STDOUT:       %ptr: type = ptr_type %A.2 [template = constants.%ptr.2]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.2 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%N.loc6_9.1: %i32) {
// CHECK:STDOUT:   %N.loc6_9.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc6_9.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A: type = class_type @A, @A(%N.loc6_9.2) [symbolic = %A (constants.%A.1)]
// CHECK:STDOUT:   %A.elem.loc7: type = unbound_element_type @A.%A (%A.1), %B [symbolic = %A.elem.loc7 (constants.%A.elem.1)]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.loc6_9.2, constants.%Convert.9 [symbolic = %Convert.bound (constants.%Convert.bound.1)]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.3(constants.%int_32) [symbolic = %Convert.specific_fn (constants.%Convert.specific_fn.1)]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn(%N.loc6_9.2) [symbolic = %int.convert_checked (constants.%int.convert_checked)]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %int.convert_checked [symbolic = %iN.builtin (constants.%iN.builtin.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @A.%iN.builtin (%iN.builtin.2) [symbolic = %require_complete (constants.%require_complete.5)]
// CHECK:STDOUT:   %A.elem.loc12: type = unbound_element_type @A.%A (%A.1), @A.%iN.builtin (%iN.builtin.2) [symbolic = %A.elem.loc12 (constants.%A.elem.2)]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: @A.%iN.builtin (%iN.builtin.2)} [symbolic = %struct_type.base.n (constants.%struct_type.base.n)]
// CHECK:STDOUT:   %complete_type.loc13_1.2: <witness> = complete_type_witness @A.%struct_type.base.n (%struct_type.base.n) [symbolic = %complete_type.loc13_1.2 (constants.%complete_type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc7: @A.%A.elem.loc7 (%A.elem.1) = base_decl %B.ref, element0 [template]
// CHECK:STDOUT:     %.loc12: @A.%A.elem.loc12 (%A.elem.2) = field_decl n, element1 [template]
// CHECK:STDOUT:     %complete_type.loc13_1.1: <witness> = complete_type_witness %struct_type.base.n [symbolic = %complete_type.loc13_1.2 (constants.%complete_type.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.1
// CHECK:STDOUT:     .base = %.loc7
// CHECK:STDOUT:     .n = %.loc12
// CHECK:STDOUT:     extend %B.ref
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc13_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int.1(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %ptr.2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.2 = name_ref a, %a
// CHECK:STDOUT:   %.loc19_16.1: ref %A.2 = deref %a.ref
// CHECK:STDOUT:   %.loc19_16.2: ref %B = class_element_access %.loc19_16.1, element0
// CHECK:STDOUT:   %addr: %ptr.3 = addr_of %.loc19_16.2
// CHECK:STDOUT:   %.loc19_16.3: %ptr.3 = converted %a.ref, %addr
// CHECK:STDOUT:   %b: %ptr.3 = bind_name b, %.loc19_16.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%N.2) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%N.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(%N.loc6_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%int_0.2) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_0.2
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_0.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.2
// CHECK:STDOUT:   %A.elem.loc7 => constants.%A.elem.3
// CHECK:STDOUT:   %Convert.bound => constants.%Convert.bound.3
// CHECK:STDOUT:   %Convert.specific_fn => constants.%Convert.specific_fn.3
// CHECK:STDOUT:   %int.convert_checked => constants.%int_0.1
// CHECK:STDOUT:   %iN.builtin => <error>
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %A.elem.loc12 => <error>
// CHECK:STDOUT:   %struct_type.base.n => <error>
// CHECK:STDOUT:   %complete_type.loc13_1.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
