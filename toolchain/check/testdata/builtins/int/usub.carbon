// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/usub.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/usub.carbon

// --- int_sub.carbon

fn Sub(a: i32, b: i32) -> i32 = "int.usub";

// CHECK:STDERR: int_sub.carbon:[[@LINE+3]]:16: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var arr: [i32; Sub(3, 2)];
// CHECK:STDERR:                ^~~~~~~~~
var arr: [i32; Sub(3, 2)];
let arr_p: [i32; 1]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Sub(a, b);
}

// --- overflow.carbon

package Overflow;

fn Sub(a: i32, b: i32) -> i32 = "int.usub";

let a: i32 = Sub(0, 0x7FFFFFFF);
let b: i32 = Sub(Sub(0, 0x7FFFFFFF), 1);
let c: i32 = Sub(Sub(0, 0x7FFFFFFF), 2);

// CHECK:STDOUT: --- int_sub.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Sub.type: type = fn_type @Sub [template]
// CHECK:STDOUT:   %Sub: %Sub.type = struct_value () [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.9: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_3, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_3) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %iN = call %Convert.specific_fn.2(%int_2) [symbolic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, %i32 [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Sub.decl: %Sub.type = fn_decl @Sub [template = constants.%Sub] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc2_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_11: init type = call constants.%Int(%int_32.loc2_11) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_11.1: type = value_of_initializer %int.make_type_signed.loc2_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_11.2: type = converted %int.make_type_signed.loc2_11, %.loc2_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc2_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_19: init type = call constants.%Int(%int_32.loc2_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.1: type = value_of_initializer %int.make_type_signed.loc2_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.2: type = converted %int.make_type_signed.loc2_19, %.loc2_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc2_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_27: init type = call constants.%Int(%int_32.loc2_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.1: type = value_of_initializer %int.make_type_signed.loc2_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.2: type = converted %int.make_type_signed.loc2_27, %.loc2_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%int_32.loc7) [template = constants.%i32]
// CHECK:STDOUT:   %Sub.ref: %Sub.type = name_ref Sub, %Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %impl.elem0.loc7_20: %Convert.type.2 = impl_witness_access constants.%impl_witness.9, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc7_20: <bound method> = bound_method %int_3, %impl.elem0.loc7_20 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_20: <specific function> = specific_function %Convert.bound.loc7_20, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc7_20: init %iN = call %Convert.specific_fn.loc7_20(%int_3) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc7_20.1: %i32 = value_of_initializer %int.convert_checked.loc7_20 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc7_20.2: %i32 = converted %int_3, %.loc7_20.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc7_23: %Convert.type.2 = impl_witness_access constants.%impl_witness.9, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc7_23: <bound method> = bound_method %int_2, %impl.elem0.loc7_23 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_23: <specific function> = specific_function %Convert.bound.loc7_23, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc7_23: init %iN = call %Convert.specific_fn.loc7_23(%int_2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc7_23.1: %i32 = value_of_initializer %int.convert_checked.loc7_23 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc7_23.2: %i32 = converted %int_2, %.loc7_23.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.usub: init %i32 = call %Sub.ref(%.loc7_20.2, %.loc7_23.2)
// CHECK:STDOUT:   %.loc7_11.1: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_11.2: type = converted %int.make_type_signed.loc7, %.loc7_11.1 [template = constants.%i32]
// CHECK:STDOUT:   %arr.var: ref <error> = var arr
// CHECK:STDOUT:   %arr: ref <error> = bind_name arr, %arr.var
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%int_32.loc8) [template = constants.%i32]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc8_13.1: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_13.2: type = converted %int.make_type_signed.loc8, %.loc8_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template = constants.%ptr]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc10_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc10_19: init type = call constants.%Int(%int_32.loc10_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_19.1: type = value_of_initializer %int.make_type_signed.loc10_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_19.2: type = converted %int.make_type_signed.loc10_19, %.loc10_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc10_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc10_27: init type = call constants.%Int(%int_32.loc10_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_27.1: type = value_of_initializer %int.make_type_signed.loc10_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_27.2: type = converted %int.make_type_signed.loc10_27, %.loc10_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc10_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc10_35: init type = call constants.%Int(%int_32.loc10_35) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_35.1: type = value_of_initializer %int.make_type_signed.loc10_35 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_35.2: type = converted %int.make_type_signed.loc10_35, %.loc10_35.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.usub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sub.ref: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.usub: init %i32 = call %Sub.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc11_19.1: %i32 = value_of_initializer %int.usub
// CHECK:STDOUT:   %.loc11_19.2: %i32 = converted %int.usub, %.loc11_19.1
// CHECK:STDOUT:   return %.loc11_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref <error> = name_ref arr, file.%arr
// CHECK:STDOUT:   %addr: <error> = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %ptr = bind_name arr_p, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Sub.type: type = fn_type @Sub [template]
// CHECK:STDOUT:   %Sub: %Sub.type = struct_value () [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.9: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_0, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_0) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2147483647, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %iN = call %Convert.specific_fn.2(%int_2147483647) [symbolic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %iN = call %Convert.specific_fn.3(%int_1) [symbolic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_2, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.4: init %iN = call %Convert.specific_fn.4(%int_2) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Sub.decl: %Sub.type = fn_decl @Sub [template = constants.%Sub] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_11: init type = call constants.%Int(%int_32.loc4_11) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.1: type = value_of_initializer %int.make_type_signed.loc4_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.2: type = converted %int.make_type_signed.loc4_11, %.loc4_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_19: init type = call constants.%Int(%int_32.loc4_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.1: type = value_of_initializer %int.make_type_signed.loc4_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.2: type = converted %int.make_type_signed.loc4_19, %.loc4_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_27: init type = call constants.%Int(%int_32.loc4_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_signed.loc4_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_signed.loc4_27, %.loc4_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%int_32.loc6) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_8.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_8.2: type = converted %int.make_type_signed.loc6, %.loc6_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%int_32.loc7) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.1: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.2: type = converted %int.make_type_signed.loc7, %.loc7_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%int_32.loc8) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8.1: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8.2: type = converted %int.make_type_signed.loc8, %.loc8_8.1 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.usub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sub.ref.loc6: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %int_0.loc6: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_2147483647.loc6: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc6_18: %Convert.type.2 = impl_witness_access constants.%impl_witness.9, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc6_18: <bound method> = bound_method %int_0.loc6, %impl.elem0.loc6_18 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_18: <specific function> = specific_function %Convert.bound.loc6_18, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc6_18: init %iN = call %Convert.specific_fn.loc6_18(%int_0.loc6) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc6_18.1: %i32 = value_of_initializer %int.convert_checked.loc6_18 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc6_18.2: %i32 = converted %int_0.loc6, %.loc6_18.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc6_21: %Convert.type.2 = impl_witness_access constants.%impl_witness.9, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc6_21: <bound method> = bound_method %int_2147483647.loc6, %impl.elem0.loc6_21 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_21: <specific function> = specific_function %Convert.bound.loc6_21, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc6_21: init %iN = call %Convert.specific_fn.loc6_21(%int_2147483647.loc6) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc6_21.1: %i32 = value_of_initializer %int.convert_checked.loc6_21 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc6_21.2: %i32 = converted %int_2147483647.loc6, %.loc6_21.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.usub.loc6: init %i32 = call %Sub.ref.loc6(%.loc6_18.2, %.loc6_21.2)
// CHECK:STDOUT:   %.loc6_32.1: %i32 = value_of_initializer %int.usub.loc6
// CHECK:STDOUT:   %.loc6_32.2: %i32 = converted %int.usub.loc6, %.loc6_32.1
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc6_32.2
// CHECK:STDOUT:   %Sub.ref.loc7_14: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Sub.ref.loc7_18: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %int_0.loc7: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_2147483647.loc7: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc7_22: %Convert.type.2 = impl_witness_access constants.%impl_witness.9, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc7_22: <bound method> = bound_method %int_0.loc7, %impl.elem0.loc7_22 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_22: <specific function> = specific_function %Convert.bound.loc7_22, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc7_22: init %iN = call %Convert.specific_fn.loc7_22(%int_0.loc7) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc7_22.1: %i32 = value_of_initializer %int.convert_checked.loc7_22 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc7_22.2: %i32 = converted %int_0.loc7, %.loc7_22.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc7_25: %Convert.type.2 = impl_witness_access constants.%impl_witness.9, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc7_25: <bound method> = bound_method %int_2147483647.loc7, %impl.elem0.loc7_25 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_25: <specific function> = specific_function %Convert.bound.loc7_25, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc7_25: init %iN = call %Convert.specific_fn.loc7_25(%int_2147483647.loc7) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc7_25.1: %i32 = value_of_initializer %int.convert_checked.loc7_25 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc7_25.2: %i32 = converted %int_2147483647.loc7, %.loc7_25.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.usub.loc7_35: init %i32 = call %Sub.ref.loc7_18(%.loc7_22.2, %.loc7_25.2)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc7_35.1: %i32 = value_of_initializer %int.usub.loc7_35
// CHECK:STDOUT:   %.loc7_35.2: %i32 = converted %int.usub.loc7_35, %.loc7_35.1
// CHECK:STDOUT:   %impl.elem0.loc7_38: %Convert.type.2 = impl_witness_access constants.%impl_witness.9, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc7_38: <bound method> = bound_method %int_1, %impl.elem0.loc7_38 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_38: <specific function> = specific_function %Convert.bound.loc7_38, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc7_38: init %iN = call %Convert.specific_fn.loc7_38(%int_1) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc7_38.1: %i32 = value_of_initializer %int.convert_checked.loc7_38 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc7_38.2: %i32 = converted %int_1, %.loc7_38.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.usub.loc7_39: init %i32 = call %Sub.ref.loc7_14(%.loc7_35.2, %.loc7_38.2)
// CHECK:STDOUT:   %.loc7_40.1: %i32 = value_of_initializer %int.usub.loc7_39
// CHECK:STDOUT:   %.loc7_40.2: %i32 = converted %int.usub.loc7_39, %.loc7_40.1
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc7_40.2
// CHECK:STDOUT:   %Sub.ref.loc8_14: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Sub.ref.loc8_18: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %int_0.loc8: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_2147483647.loc8: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc8_22: %Convert.type.2 = impl_witness_access constants.%impl_witness.9, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_22: <bound method> = bound_method %int_0.loc8, %impl.elem0.loc8_22 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_22: <specific function> = specific_function %Convert.bound.loc8_22, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc8_22: init %iN = call %Convert.specific_fn.loc8_22(%int_0.loc8) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc8_22.1: %i32 = value_of_initializer %int.convert_checked.loc8_22 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc8_22.2: %i32 = converted %int_0.loc8, %.loc8_22.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc8_25: %Convert.type.2 = impl_witness_access constants.%impl_witness.9, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_25: <bound method> = bound_method %int_2147483647.loc8, %impl.elem0.loc8_25 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_25: <specific function> = specific_function %Convert.bound.loc8_25, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc8_25: init %iN = call %Convert.specific_fn.loc8_25(%int_2147483647.loc8) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc8_25.1: %i32 = value_of_initializer %int.convert_checked.loc8_25 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc8_25.2: %i32 = converted %int_2147483647.loc8, %.loc8_25.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.usub.loc8_35: init %i32 = call %Sub.ref.loc8_18(%.loc8_22.2, %.loc8_25.2)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc8_35.1: %i32 = value_of_initializer %int.usub.loc8_35
// CHECK:STDOUT:   %.loc8_35.2: %i32 = converted %int.usub.loc8_35, %.loc8_35.1
// CHECK:STDOUT:   %impl.elem0.loc8_38: %Convert.type.2 = impl_witness_access constants.%impl_witness.9, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_38: <bound method> = bound_method %int_2, %impl.elem0.loc8_38 [symbolic = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_38: <specific function> = specific_function %Convert.bound.loc8_38, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc8_38: init %iN = call %Convert.specific_fn.loc8_38(%int_2) [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc8_38.1: %i32 = value_of_initializer %int.convert_checked.loc8_38 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc8_38.2: %i32 = converted %int_2, %.loc8_38.1 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %int.usub.loc8_39: init %i32 = call %Sub.ref.loc8_14(%.loc8_35.2, %.loc8_38.2)
// CHECK:STDOUT:   %.loc8_40.1: %i32 = value_of_initializer %int.usub.loc8_39
// CHECK:STDOUT:   %.loc8_40.2: %i32 = converted %int.usub.loc8_39, %.loc8_40.1
// CHECK:STDOUT:   %c: %i32 = bind_name c, %.loc8_40.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
