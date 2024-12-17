// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/right_shift.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/right_shift.carbon

// --- int_right_shift.carbon

fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";

// CHECK:STDERR: int_right_shift.carbon:[[@LINE+4]]:16: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var arr: [i32; RightShift(22, 2)];
// CHECK:STDERR:                ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var arr: [i32; RightShift(22, 2)];
let arr_p: [i32; 5]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return RightShift(a, b);
}

// TODO: Test mixed types for LHS and RHS.

// --- arith_shift.carbon

// TODO: Also test unsigned / logical right shift.

package ArithShift;

fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
fn Negate(a: i32) -> i32 = "int.snegate";

// -1 >> 1 is -1.
// CHECK:STDERR: arith_shift.carbon:[[@LINE+4]]:17: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var arr1: [i32; Negate(RightShift(Negate(1), 1))];
// CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var arr1: [i32; Negate(RightShift(Negate(1), 1))];
let arr1_p: [i32; 1]* = &arr1;

// -10 >> 2 is -3.
// CHECK:STDERR: arith_shift.carbon:[[@LINE+3]]:17: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var arr2: [i32; Negate(RightShift(Negate(10), 2))];
// CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
var arr2: [i32; Negate(RightShift(Negate(10), 2))];
let arr2_p: [i32; 3]* = &arr2;

// --- fail_bad_shift.carbon

package BadShift;

fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
fn Negate(a: i32) -> i32 = "int.snegate";

// Shift greater than size is disallowed.
let size_1: i32 = RightShift(1, 31);
let size_2: i32 = RightShift(1, 32);
let size_3: i32 = RightShift(1, 33);

// Negative shifts aren't allowed either.
let negative: i32 = RightShift(1, Negate(1));

// CHECK:STDOUT: --- int_right_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %RightShift.type.1: type = fn_type @RightShift.1 [template]
// CHECK:STDOUT:   %RightShift: %RightShift.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_22, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %Int = call %Convert.specific_fn.1(%int_22) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %Int = call %Convert.specific_fn.2(%int_2) [symbolic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_5, %i32 [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .RightShift = %RightShift.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %RightShift.decl: %RightShift.type.1 = fn_decl @RightShift.1 [template = constants.%RightShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc2_18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc2_18: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc2_26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc2_26: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc2_34: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc2_34: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %RightShift.ref: %RightShift.type.1 = name_ref RightShift, %RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [template = constants.%int_22]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %impl.elem0.loc8_27: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_27: <bound method> = bound_method %int_22, %impl.elem0.loc8_27 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_27: <specific function> = specific_function %Convert.bound.loc8_27, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc8_27: init %Int = call %Convert.specific_fn.loc8_27(%int_22) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc8_27.1: %i32 = value_of_initializer %int.convert_checked.loc8_27 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc8_27.2: %i32 = converted %int_22, %.loc8_27.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc8_31: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_31: <bound method> = bound_method %int_2, %impl.elem0.loc8_31 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_31: <specific function> = specific_function %Convert.bound.loc8_31, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc8_31: init %Int = call %Convert.specific_fn.loc8_31(%int_2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc8_31.1: %i32 = value_of_initializer %int.convert_checked.loc8_31 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc8_31.2: %i32 = converted %int_2, %.loc8_31.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.right_shift: init %i32 = call %RightShift.ref(%.loc8_27.2, %.loc8_31.2)
// CHECK:STDOUT:   %arr.var: ref <error> = var arr
// CHECK:STDOUT:   %arr: ref <error> = bind_name arr, %arr.var
// CHECK:STDOUT:   %int_32.loc9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %array_type: type = array_type %int_5, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template = constants.%ptr]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_35: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShift.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %RightShift.ref: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.right_shift: init %i32 = call %RightShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc12_26.1: %i32 = value_of_initializer %int.right_shift
// CHECK:STDOUT:   %.loc12_26.2: %i32 = converted %int.right_shift, %.loc12_26.1
// CHECK:STDOUT:   return %.loc12_26.2
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
// CHECK:STDOUT: --- arith_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %RightShift.type.1: type = fn_type @RightShift.1 [template]
// CHECK:STDOUT:   %RightShift: %RightShift.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Negate.type.1: type = fn_type @Negate.1 [template]
// CHECK:STDOUT:   %Negate: %Negate.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %Int = call %Convert.specific_fn.1(%int_1) [symbolic]
// CHECK:STDOUT:   %array_type.1: type = array_type %int_1, %i32 [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %array_type.1 [template]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_10, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %Int = call %Convert.specific_fn.2(%int_10) [symbolic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_2, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %Int = call %Convert.specific_fn.3(%int_2) [symbolic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type.2: type = array_type %int_3, %i32 [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %array_type.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .RightShift = %RightShift.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .arr1 = %arr1
// CHECK:STDOUT:     .arr1_p = @__global_init.%arr1_p
// CHECK:STDOUT:     .arr2 = %arr2
// CHECK:STDOUT:     .arr2_p = @__global_init.%arr2_p
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %RightShift.decl: %RightShift.type.1 = fn_decl @RightShift.1 [template = constants.%RightShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_18: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc6_26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_26: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc6_34: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_34: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: %Negate.type.1 = fn_decl @Negate.1 [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc7_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7_14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc7_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %Negate.ref.loc14_17: %Negate.type.1 = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %RightShift.ref.loc14: %RightShift.type.1 = name_ref RightShift, %RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %Negate.ref.loc14_35: %Negate.type.1 = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc14_42: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc14_42: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc14_42: <bound method> = bound_method %int_1.loc14_42, %impl.elem0.loc14_42 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_42: <specific function> = specific_function %Convert.bound.loc14_42, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc14_42: init %Int = call %Convert.specific_fn.loc14_42(%int_1.loc14_42) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc14_42.1: %i32 = value_of_initializer %int.convert_checked.loc14_42 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc14_42.2: %i32 = converted %int_1.loc14_42, %.loc14_42.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.snegate.loc14_43: init %i32 = call %Negate.ref.loc14_35(%.loc14_42.2)
// CHECK:STDOUT:   %int_1.loc14_46: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc14_43.1: %i32 = value_of_initializer %int.snegate.loc14_43
// CHECK:STDOUT:   %.loc14_43.2: %i32 = converted %int.snegate.loc14_43, %.loc14_43.1
// CHECK:STDOUT:   %impl.elem0.loc14_46: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc14_46: <bound method> = bound_method %int_1.loc14_46, %impl.elem0.loc14_46 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_46: <specific function> = specific_function %Convert.bound.loc14_46, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc14_46: init %Int = call %Convert.specific_fn.loc14_46(%int_1.loc14_46) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc14_46.1: %i32 = value_of_initializer %int.convert_checked.loc14_46 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc14_46.2: %i32 = converted %int_1.loc14_46, %.loc14_46.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.right_shift.loc14: init %i32 = call %RightShift.ref.loc14(%.loc14_43.2, %.loc14_46.2)
// CHECK:STDOUT:   %.loc14_47.1: %i32 = value_of_initializer %int.right_shift.loc14
// CHECK:STDOUT:   %.loc14_47.2: %i32 = converted %int.right_shift.loc14, %.loc14_47.1
// CHECK:STDOUT:   %int.snegate.loc14_48: init %i32 = call %Negate.ref.loc14_17(%.loc14_47.2)
// CHECK:STDOUT:   %arr1.var: ref <error> = var arr1
// CHECK:STDOUT:   %arr1: ref <error> = bind_name arr1, %arr1.var
// CHECK:STDOUT:   %int_32.loc15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %array_type.loc15: type = array_type %int_1.loc15, %i32 [template = constants.%array_type.1]
// CHECK:STDOUT:   %ptr.loc15: type = ptr_type %array_type.1 [template = constants.%ptr.1]
// CHECK:STDOUT:   %int_32.loc21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %Negate.ref.loc21_17: %Negate.type.1 = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %RightShift.ref.loc21: %RightShift.type.1 = name_ref RightShift, %RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %Negate.ref.loc21_35: %Negate.type.1 = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template = constants.%int_10]
// CHECK:STDOUT:   %impl.elem0.loc21_42: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc21_42: <bound method> = bound_method %int_10, %impl.elem0.loc21_42 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_42: <specific function> = specific_function %Convert.bound.loc21_42, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc21_42: init %Int = call %Convert.specific_fn.loc21_42(%int_10) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc21_42.1: %i32 = value_of_initializer %int.convert_checked.loc21_42 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc21_42.2: %i32 = converted %int_10, %.loc21_42.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.snegate.loc21_44: init %i32 = call %Negate.ref.loc21_35(%.loc21_42.2)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc21_44.1: %i32 = value_of_initializer %int.snegate.loc21_44
// CHECK:STDOUT:   %.loc21_44.2: %i32 = converted %int.snegate.loc21_44, %.loc21_44.1
// CHECK:STDOUT:   %impl.elem0.loc21_47: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc21_47: <bound method> = bound_method %int_2, %impl.elem0.loc21_47 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_47: <specific function> = specific_function %Convert.bound.loc21_47, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc21_47: init %Int = call %Convert.specific_fn.loc21_47(%int_2) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc21_47.1: %i32 = value_of_initializer %int.convert_checked.loc21_47 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc21_47.2: %i32 = converted %int_2, %.loc21_47.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.right_shift.loc21: init %i32 = call %RightShift.ref.loc21(%.loc21_44.2, %.loc21_47.2)
// CHECK:STDOUT:   %.loc21_48.1: %i32 = value_of_initializer %int.right_shift.loc21
// CHECK:STDOUT:   %.loc21_48.2: %i32 = converted %int.right_shift.loc21, %.loc21_48.1
// CHECK:STDOUT:   %int.snegate.loc21_49: init %i32 = call %Negate.ref.loc21_17(%.loc21_48.2)
// CHECK:STDOUT:   %arr2.var: ref <error> = var arr2
// CHECK:STDOUT:   %arr2: ref <error> = bind_name arr2, %arr2.var
// CHECK:STDOUT:   %int_32.loc22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %array_type.loc22: type = array_type %int_3, %i32 [template = constants.%array_type.2]
// CHECK:STDOUT:   %ptr.loc22: type = ptr_type %array_type.2 [template = constants.%ptr.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShift.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate.1(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr1.ref: ref <error> = name_ref arr1, file.%arr1
// CHECK:STDOUT:   %addr.loc15: <error> = addr_of %arr1.ref
// CHECK:STDOUT:   %arr1_p: %ptr.1 = bind_name arr1_p, <error>
// CHECK:STDOUT:   %arr2.ref: ref <error> = name_ref arr2, file.%arr2
// CHECK:STDOUT:   %addr.loc22: <error> = addr_of %arr2.ref
// CHECK:STDOUT:   %arr2_p: %ptr.2 = bind_name arr2_p, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %RightShift.type.1: type = fn_type @RightShift.1 [template]
// CHECK:STDOUT:   %RightShift: %RightShift.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Negate.type.1: type = fn_type @Negate.1 [template]
// CHECK:STDOUT:   %Negate: %Negate.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %Int = call %Convert.specific_fn.1(%int_1) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_31, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %Int = call %Convert.specific_fn.2(%int_31) [symbolic]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_32, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %Int = call %Convert.specific_fn.3(%int_32) [symbolic]
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_33, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.4: init %Int = call %Convert.specific_fn.4(%int_33) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .RightShift = %RightShift.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .size_1 = @__global_init.%size_1
// CHECK:STDOUT:     .size_2 = @__global_init.%size_2
// CHECK:STDOUT:     .size_3 = @__global_init.%size_3
// CHECK:STDOUT:     .negative = @__global_init.%negative
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %RightShift.decl: %RightShift.type.1 = fn_decl @RightShift.1 [template = constants.%RightShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_18: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_26: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_34: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_34: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: %Negate.type.1 = fn_decl @Negate.1 [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShift.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate.1(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %RightShift.ref.loc8: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [template = constants.%int_31]
// CHECK:STDOUT:   %impl.elem0.loc8_30: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_30: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8_30 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_30: <specific function> = specific_function %Convert.bound.loc8_30, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc8_30: init %Int = call %Convert.specific_fn.loc8_30(%int_1.loc8) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc8_30.1: %i32 = value_of_initializer %int.convert_checked.loc8_30 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc8_30.2: %i32 = converted %int_1.loc8, %.loc8_30.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc8_33: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_33: <bound method> = bound_method %int_31, %impl.elem0.loc8_33 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_33: <specific function> = specific_function %Convert.bound.loc8_33, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc8_33: init %Int = call %Convert.specific_fn.loc8_33(%int_31) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc8_33.1: %i32 = value_of_initializer %int.convert_checked.loc8_33 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc8_33.2: %i32 = converted %int_31, %.loc8_33.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.right_shift.loc8: init %i32 = call %RightShift.ref.loc8(%.loc8_30.2, %.loc8_33.2)
// CHECK:STDOUT:   %.loc8_36.1: %i32 = value_of_initializer %int.right_shift.loc8
// CHECK:STDOUT:   %.loc8_36.2: %i32 = converted %int.right_shift.loc8, %.loc8_36.1
// CHECK:STDOUT:   %size_1: %i32 = bind_name size_1, %.loc8_36.2
// CHECK:STDOUT:   %RightShift.ref.loc9: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %impl.elem0.loc9_30: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc9_30: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_30 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_30: <specific function> = specific_function %Convert.bound.loc9_30, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc9_30: init %Int = call %Convert.specific_fn.loc9_30(%int_1.loc9) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_30.1: %i32 = value_of_initializer %int.convert_checked.loc9_30 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_30.2: %i32 = converted %int_1.loc9, %.loc9_30.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc9_33: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc9_33: <bound method> = bound_method %int_32, %impl.elem0.loc9_33 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_33: <specific function> = specific_function %Convert.bound.loc9_33, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc9_33: init %Int = call %Convert.specific_fn.loc9_33(%int_32) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc9_33.1: %i32 = value_of_initializer %int.convert_checked.loc9_33 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc9_33.2: %i32 = converted %int_32, %.loc9_33.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.right_shift.loc9: init %i32 = call %RightShift.ref.loc9(%.loc9_30.2, %.loc9_33.2)
// CHECK:STDOUT:   %.loc9_36.1: %i32 = value_of_initializer %int.right_shift.loc9
// CHECK:STDOUT:   %.loc9_36.2: %i32 = converted %int.right_shift.loc9, %.loc9_36.1
// CHECK:STDOUT:   %size_2: %i32 = bind_name size_2, %.loc9_36.2
// CHECK:STDOUT:   %RightShift.ref.loc10: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [template = constants.%int_33]
// CHECK:STDOUT:   %impl.elem0.loc10_30: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc10_30: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10_30 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_30: <specific function> = specific_function %Convert.bound.loc10_30, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_30: init %Int = call %Convert.specific_fn.loc10_30(%int_1.loc10) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc10_30.1: %i32 = value_of_initializer %int.convert_checked.loc10_30 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc10_30.2: %i32 = converted %int_1.loc10, %.loc10_30.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc10_33: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc10_33: <bound method> = bound_method %int_33, %impl.elem0.loc10_33 [symbolic = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_33: <specific function> = specific_function %Convert.bound.loc10_33, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc10_33: init %Int = call %Convert.specific_fn.loc10_33(%int_33) [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc10_33.1: %i32 = value_of_initializer %int.convert_checked.loc10_33 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc10_33.2: %i32 = converted %int_33, %.loc10_33.1 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %int.right_shift.loc10: init %i32 = call %RightShift.ref.loc10(%.loc10_30.2, %.loc10_33.2)
// CHECK:STDOUT:   %.loc10_36.1: %i32 = value_of_initializer %int.right_shift.loc10
// CHECK:STDOUT:   %.loc10_36.2: %i32 = converted %int.right_shift.loc10, %.loc10_36.1
// CHECK:STDOUT:   %size_3: %i32 = bind_name size_3, %.loc10_36.2
// CHECK:STDOUT:   %RightShift.ref.loc13: %RightShift.type.1 = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc13_32: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %Negate.ref: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc13_42: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc13_42: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc13_42: <bound method> = bound_method %int_1.loc13_42, %impl.elem0.loc13_42 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_42: <specific function> = specific_function %Convert.bound.loc13_42, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_42: init %Int = call %Convert.specific_fn.loc13_42(%int_1.loc13_42) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc13_42.1: %i32 = value_of_initializer %int.convert_checked.loc13_42 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc13_42.2: %i32 = converted %int_1.loc13_42, %.loc13_42.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.snegate: init %i32 = call %Negate.ref(%.loc13_42.2)
// CHECK:STDOUT:   %impl.elem0.loc13_32: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc13_32: <bound method> = bound_method %int_1.loc13_32, %impl.elem0.loc13_32 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_32: <specific function> = specific_function %Convert.bound.loc13_32, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_32: init %Int = call %Convert.specific_fn.loc13_32(%int_1.loc13_32) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc13_32.1: %i32 = value_of_initializer %int.convert_checked.loc13_32 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc13_32.2: %i32 = converted %int_1.loc13_32, %.loc13_32.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc13_43.1: %i32 = value_of_initializer %int.snegate
// CHECK:STDOUT:   %.loc13_43.2: %i32 = converted %int.snegate, %.loc13_43.1
// CHECK:STDOUT:   %int.right_shift.loc13: init %i32 = call %RightShift.ref.loc13(%.loc13_32.2, %.loc13_43.2)
// CHECK:STDOUT:   %.loc13_45.1: %i32 = value_of_initializer %int.right_shift.loc13
// CHECK:STDOUT:   %.loc13_45.2: %i32 = converted %int.right_shift.loc13, %.loc13_45.1
// CHECK:STDOUT:   %negative: %i32 = bind_name negative, %.loc13_45.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
