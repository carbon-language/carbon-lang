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
// CHECK:STDOUT:   %Int.type.913b: type = fn_type @Int.1 [template]
// CHECK:STDOUT:   %Int.7799: %Int.type.913b = struct_value () [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3571: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int.2, @Int.2(%int_32) [template]
// CHECK:STDOUT:   %N.3e4d: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.52ab: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [template]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [template]
// CHECK:STDOUT:   %A.71db: type = class_type @A, @A(%N.3e4d) [symbolic]
// CHECK:STDOUT:   %A.elem.7922: type = unbound_element_type %A.71db, %B [symbolic]
// CHECK:STDOUT:   %Convert.type.d18e: type = fn_type @Convert.3, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.079a: %Convert.type.d18e = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.d9b8: <bound method> = bound_method %N.3e4d, %Convert.079a [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.ad3a: <specific function> = specific_function %Convert.bound.d9b8, @Convert.3(%int_32) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn.ad3a(%N.3e4d) [symbolic]
// CHECK:STDOUT:   %iN.builtin.3cef: type = int_type signed, %int.convert_checked [symbolic]
// CHECK:STDOUT:   %require_complete.47c1: <witness> = require_complete_type %iN.builtin.3cef [symbolic]
// CHECK:STDOUT:   %A.elem.674f: type = unbound_element_type %A.71db, %iN.builtin.3cef [symbolic]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: %iN.builtin.3cef} [symbolic]
// CHECK:STDOUT:   %complete_type.8e38: <witness> = complete_type_witness %struct_type.base.n [symbolic]
// CHECK:STDOUT:   %int_0.5c61: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.cd1a: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b05: <witness> = impl_witness (imports.%import_ref.7238), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.4666: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.9251: %Convert.type.4666 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.b412: <bound method> = bound_method %int_0.5c61, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.c350: <specific function> = specific_function %Convert.bound.b412, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.f61a: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %A.b937: type = class_type @A, @A(%int_0.f61a) [template]
// CHECK:STDOUT:   %ptr.ee11: type = ptr_type %A.b937 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %ptr.e792: type = ptr_type %B [template]
// CHECK:STDOUT:   %A.elem.fcc9: type = unbound_element_type %A.b937, %B [template]
// CHECK:STDOUT:   %Convert.bound.64ec: <bound method> = bound_method %int_0.f61a, %Convert.079a [template]
// CHECK:STDOUT:   %Convert.specific_fn.f03e: <specific function> = specific_function %Convert.bound.64ec, @Convert.3(%int_32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.8ee9
// CHECK:STDOUT:     .Int = %import_ref.187e
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a691
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8ee9: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
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
// CHECK:STDOUT:   %Int.decl: %Int.type.913b = fn_decl @Int.1 [template = constants.%Int.7799] {
// CHECK:STDOUT:     %N.patt: Core.IntLiteral = binding_pattern N
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc2_27.1: type = splice_block %.loc2_27.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.8ee9 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc2_27.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc2_27.3: type = converted %int_literal.make_type, %.loc2_27.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [template = constants.%A.generic] {
// CHECK:STDOUT:     %N.patt.loc6_9.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt.52ab)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc6_9.1, runtime_param<invalid> [symbolic = %N.patt.loc6_9.2 (constants.%N.patt.52ab)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int.2, @Int.2(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_9.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_9.2 (constants.%N.3e4d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt: %ptr.ee11 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %ptr.ee11 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.ee11 = value_param runtime_param0
// CHECK:STDOUT:     %.loc15_13: type = splice_block %ptr [template = constants.%ptr.ee11] {
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A.generic]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c61]
// CHECK:STDOUT:       %impl.elem0: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:       %Convert.bound: <bound method> = bound_method %int_0, %impl.elem0 [template = constants.%Convert.bound.b412]
// CHECK:STDOUT:       %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.c350]
// CHECK:STDOUT:       %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_0) [template = constants.%int_0.f61a]
// CHECK:STDOUT:       %.loc15_12.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_0.f61a]
// CHECK:STDOUT:       %.loc15_12.2: %i32 = converted %int_0, %.loc15_12.1 [template = constants.%int_0.f61a]
// CHECK:STDOUT:       %A: type = class_type @A, @A(constants.%int_0.f61a) [template = constants.%A.b937]
// CHECK:STDOUT:       %ptr: type = ptr_type %A.b937 [template = constants.%ptr.ee11]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.ee11 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.3571]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%N.loc6_9.1: %i32) {
// CHECK:STDOUT:   %N.loc6_9.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_9.2 (constants.%N.3e4d)]
// CHECK:STDOUT:   %N.patt.loc6_9.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_9.2 (constants.%N.patt.52ab)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A: type = class_type @A, @A(%N.loc6_9.2) [symbolic = %A (constants.%A.71db)]
// CHECK:STDOUT:   %A.elem.loc7: type = unbound_element_type @A.%A (%A.71db), %B [symbolic = %A.elem.loc7 (constants.%A.elem.7922)]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.loc6_9.2, constants.%Convert.079a [symbolic = %Convert.bound (constants.%Convert.bound.d9b8)]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.3(constants.%int_32) [symbolic = %Convert.specific_fn (constants.%Convert.specific_fn.ad3a)]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn(%N.loc6_9.2) [symbolic = %int.convert_checked (constants.%int.convert_checked)]
// CHECK:STDOUT:   %iN.builtin: type = int_type signed, %int.convert_checked [symbolic = %iN.builtin (constants.%iN.builtin.3cef)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @A.%iN.builtin (%iN.builtin.3cef) [symbolic = %require_complete (constants.%require_complete.47c1)]
// CHECK:STDOUT:   %A.elem.loc12: type = unbound_element_type @A.%A (%A.71db), @A.%iN.builtin (%iN.builtin.3cef) [symbolic = %A.elem.loc12 (constants.%A.elem.674f)]
// CHECK:STDOUT:   %struct_type.base.n: type = struct_type {.base: %B, .n: @A.%iN.builtin (%iN.builtin.3cef)} [symbolic = %struct_type.base.n (constants.%struct_type.base.n)]
// CHECK:STDOUT:   %complete_type.loc13_1.2: <witness> = complete_type_witness @A.%struct_type.base.n (%struct_type.base.n) [symbolic = %complete_type.loc13_1.2 (constants.%complete_type.8e38)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc7: @A.%A.elem.loc7 (%A.elem.7922) = base_decl %B.ref, element0 [template]
// CHECK:STDOUT:     %.loc12: @A.%A.elem.loc12 (%A.elem.674f) = field_decl n, element1 [template]
// CHECK:STDOUT:     %complete_type.loc13_1.1: <witness> = complete_type_witness %struct_type.base.n [symbolic = %complete_type.loc13_1.2 (constants.%complete_type.8e38)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.71db
// CHECK:STDOUT:     .base = %.loc7
// CHECK:STDOUT:     .n = %.loc12
// CHECK:STDOUT:     extend %B.ref
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc13_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int.1(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %ptr.ee11) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.ee11 = name_ref a, %a
// CHECK:STDOUT:   %.loc19_16.1: ref %A.b937 = deref %a.ref
// CHECK:STDOUT:   %.loc19_16.2: ref %B = class_element_access %.loc19_16.1, element0
// CHECK:STDOUT:   %addr: %ptr.e792 = addr_of %.loc19_16.2
// CHECK:STDOUT:   %.loc19_16.3: %ptr.e792 = converted %a.ref, %addr
// CHECK:STDOUT:   %b: %ptr.e792 = bind_name b, %.loc19_16.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%N.3e4d) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%N.3e4d
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%N.3e4d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(%N.loc6_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%int_0.f61a) {
// CHECK:STDOUT:   %N.loc6_9.2 => constants.%int_0.f61a
// CHECK:STDOUT:   %N.patt.loc6_9.2 => constants.%int_0.f61a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A => constants.%A.b937
// CHECK:STDOUT:   %A.elem.loc7 => constants.%A.elem.fcc9
// CHECK:STDOUT:   %Convert.bound => constants.%Convert.bound.64ec
// CHECK:STDOUT:   %Convert.specific_fn => constants.%Convert.specific_fn.f03e
// CHECK:STDOUT:   %int.convert_checked => constants.%int_0.5c61
// CHECK:STDOUT:   %iN.builtin => <error>
// CHECK:STDOUT:   %require_complete => <error>
// CHECK:STDOUT:   %A.elem.loc12 => <error>
// CHECK:STDOUT:   %struct_type.base.n => <error>
// CHECK:STDOUT:   %complete_type.loc13_1.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
