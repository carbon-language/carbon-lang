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
var arr1: [i32; Negate(RightShift(Negate(1), 1))];
let arr1_p: [i32; 1]* = &arr1;

// -10 >> 2 is -3.
var arr2: [i32; Negate(RightShift(Negate(10), 2))];
let arr2_p: [i32; 3]* = &arr2;

// --- fail_bad_shift.carbon

package BadShift;

fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
fn Negate(a: i32) -> i32 = "int.snegate";

// Shift greater than size is disallowed.
let size_1: i32 = RightShift(1, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance not in range [0, 32) in 1 >> 32 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_2: i32 = RightShift(1, 32);
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let size_2: i32 = RightShift(1, 32);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance not in range [0, 32) in 1 >> 33 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_3: i32 = RightShift(1, 33);
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let size_3: i32 = RightShift(1, 33);

// Negative shifts aren't allowed either.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+3]]:21: error: shift distance not in range [0, 32) in 1 >> -1 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let negative: i32 = RightShift(1, Negate(1));
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~~~~~~~
let negative: i32 = RightShift(1, Negate(1));

// CHECK:STDOUT: --- int_right_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %RightShift.type: type = fn_type @RightShift [template]
// CHECK:STDOUT:   %RightShift: %RightShift.type = struct_value () [template]
// CHECK:STDOUT:   %int_22.1: Core.IntLiteral = int_value 22 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_22.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_22.2: %i32 = int_value 22 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int_5.1: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %interface.10: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_5.1, %Convert.15 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.4(%int_32) [template]
// CHECK:STDOUT:   %int_5.2: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_5.2, %i32 [template]
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
// CHECK:STDOUT:     .RightShift = %RightShift.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %RightShift.decl: %RightShift.type = fn_decl @RightShift [template = constants.%RightShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc2_18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_18: init type = call constants.%Int(%int_32.loc2_18) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_18.1: type = value_of_initializer %int.make_type_signed.loc2_18 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_18.2: type = converted %int.make_type_signed.loc2_18, %.loc2_18.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc2_26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_26: init type = call constants.%Int(%int_32.loc2_26) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_26.1: type = value_of_initializer %int.make_type_signed.loc2_26 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_26.2: type = converted %int.make_type_signed.loc2_26, %.loc2_26.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc2_34: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_34: init type = call constants.%Int(%int_32.loc2_34) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_34.1: type = value_of_initializer %int.make_type_signed.loc2_34 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_34.2: type = converted %int.make_type_signed.loc2_34, %.loc2_34.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc4: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%int_32.loc4) [template = constants.%i32]
// CHECK:STDOUT:   %RightShift.ref: %RightShift.type = name_ref RightShift, %RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [template = constants.%int_22.1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc4_27: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc4_27: <bound method> = bound_method %int_22, %impl.elem0.loc4_27 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc4_27: <specific function> = specific_function %Convert.bound.loc4_27, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc4_27: init %i32 = call %Convert.specific_fn.loc4_27(%int_22) [template = constants.%int_22.2]
// CHECK:STDOUT:   %.loc4_27.1: %i32 = value_of_initializer %int.convert_checked.loc4_27 [template = constants.%int_22.2]
// CHECK:STDOUT:   %.loc4_27.2: %i32 = converted %int_22, %.loc4_27.1 [template = constants.%int_22.2]
// CHECK:STDOUT:   %impl.elem0.loc4_31: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc4_31: <bound method> = bound_method %int_2, %impl.elem0.loc4_31 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc4_31: <specific function> = specific_function %Convert.bound.loc4_31, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc4_31: init %i32 = call %Convert.specific_fn.loc4_31(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc4_31.1: %i32 = value_of_initializer %int.convert_checked.loc4_31 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc4_31.2: %i32 = converted %int_2, %.loc4_31.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.right_shift: init %i32 = call %RightShift.ref(%.loc4_27.2, %.loc4_31.2) [template = constants.%int_5.1]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_signed.loc4, %.loc4_11.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc4_32: %Convert.type.6 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %Convert.bound.loc4_32: <bound method> = bound_method %int.right_shift, %impl.elem0.loc4_32 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc4_32: <specific function> = specific_function %Convert.bound.loc4_32, @Convert.4(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %.loc4_32.1: %i32 = value_of_initializer %int.right_shift [template = constants.%int_5.1]
// CHECK:STDOUT:   %.loc4_32.2: %i32 = converted %int.right_shift, %.loc4_32.1 [template = constants.%int_5.1]
// CHECK:STDOUT:   %int.convert_checked.loc4_32: init Core.IntLiteral = call %Convert.specific_fn.loc4_32(%.loc4_32.2) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc4_32.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4_32 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc4_32.4: Core.IntLiteral = converted %int.right_shift, %.loc4_32.3 [template = constants.%int_5.2]
// CHECK:STDOUT:   %array_type.loc4: type = array_type %.loc4_32.4, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %arr.var: ref %array_type = var arr
// CHECK:STDOUT:   %arr: ref %array_type = bind_name arr, %arr.var
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%int_32.loc5) [template = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc5_13.1: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_13.2: type = converted %int.make_type_signed.loc5, %.loc5_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type.loc5: type = array_type %int_5, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template = constants.%ptr]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc7_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc7_19: init type = call constants.%Int(%int_32.loc7_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.1: type = value_of_initializer %int.make_type_signed.loc7_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.2: type = converted %int.make_type_signed.loc7_19, %.loc7_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc7_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc7_27: init type = call constants.%Int(%int_32.loc7_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.1: type = value_of_initializer %int.make_type_signed.loc7_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.2: type = converted %int.make_type_signed.loc7_27, %.loc7_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc7_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc7_35: init type = call constants.%Int(%int_32.loc7_35) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_35.1: type = value_of_initializer %int.make_type_signed.loc7_35 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_35.2: type = converted %int.make_type_signed.loc7_35, %.loc7_35.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShift(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %RightShift.ref: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.right_shift: init %i32 = call %RightShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_26.1: %i32 = value_of_initializer %int.right_shift
// CHECK:STDOUT:   %.loc8_26.2: %i32 = converted %int.right_shift, %.loc8_26.1
// CHECK:STDOUT:   return %.loc8_26.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %array_type = name_ref arr, file.%arr
// CHECK:STDOUT:   %addr: %ptr = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %ptr = bind_name arr_p, %addr
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- arith_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %RightShift.type: type = fn_type @RightShift [template]
// CHECK:STDOUT:   %RightShift: %RightShift.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_-1: %i32 = int_value -1 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %interface.10: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_1.2, %Convert.15 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.4(%int_32) [template]
// CHECK:STDOUT:   %array_type.1: type = array_type %int_1.1, %i32 [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %array_type.1 [template]
// CHECK:STDOUT:   %int_10.1: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_10.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_10.2: %i32 = int_value 10 [template]
// CHECK:STDOUT:   %int_-10: %i32 = int_value -10 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int_-3: %i32 = int_value -3 [template]
// CHECK:STDOUT:   %int_3.1: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_3.1, %Convert.15 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.4(%int_32) [template]
// CHECK:STDOUT:   %int_3.2: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type.2: type = array_type %int_3.2, %i32 [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %array_type.2 [template]
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
// CHECK:STDOUT:     .RightShift = %RightShift.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .arr1 = %arr1
// CHECK:STDOUT:     .arr1_p = @__global_init.%arr1_p
// CHECK:STDOUT:     .arr2 = %arr2
// CHECK:STDOUT:     .arr2_p = @__global_init.%arr2_p
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %RightShift.decl: %RightShift.type = fn_decl @RightShift [template = constants.%RightShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6_18: init type = call constants.%Int(%int_32.loc6_18) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_18.1: type = value_of_initializer %int.make_type_signed.loc6_18 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_18.2: type = converted %int.make_type_signed.loc6_18, %.loc6_18.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc6_26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6_26: init type = call constants.%Int(%int_32.loc6_26) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_26.1: type = value_of_initializer %int.make_type_signed.loc6_26 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_26.2: type = converted %int.make_type_signed.loc6_26, %.loc6_26.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc6_34: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6_34: init type = call constants.%Int(%int_32.loc6_34) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_34.1: type = value_of_initializer %int.make_type_signed.loc6_34 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_34.2: type = converted %int.make_type_signed.loc6_34, %.loc6_34.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc7_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc7_14: init type = call constants.%Int(%int_32.loc7_14) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_14.1: type = value_of_initializer %int.make_type_signed.loc7_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_14.2: type = converted %int.make_type_signed.loc7_14, %.loc7_14.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc7_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc7_22: init type = call constants.%Int(%int_32.loc7_22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_22.1: type = value_of_initializer %int.make_type_signed.loc7_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_22.2: type = converted %int.make_type_signed.loc7_22, %.loc7_22.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc10: init type = call constants.%Int(%int_32.loc10) [template = constants.%i32]
// CHECK:STDOUT:   %Negate.ref.loc10_17: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %RightShift.ref.loc10: %RightShift.type = name_ref RightShift, %RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %Negate.ref.loc10_35: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc10_42: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc10_42: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc10_42: <bound method> = bound_method %int_1.loc10_42, %impl.elem0.loc10_42 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_42: <specific function> = specific_function %Convert.bound.loc10_42, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_42: init %i32 = call %Convert.specific_fn.loc10_42(%int_1.loc10_42) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_42.1: %i32 = value_of_initializer %int.convert_checked.loc10_42 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_42.2: %i32 = converted %int_1.loc10_42, %.loc10_42.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.snegate.loc10_43: init %i32 = call %Negate.ref.loc10_35(%.loc10_42.2) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_1.loc10_46: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc10_43.1: %i32 = value_of_initializer %int.snegate.loc10_43 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc10_43.2: %i32 = converted %int.snegate.loc10_43, %.loc10_43.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %impl.elem0.loc10_46: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc10_46: <bound method> = bound_method %int_1.loc10_46, %impl.elem0.loc10_46 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_46: <specific function> = specific_function %Convert.bound.loc10_46, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_46: init %i32 = call %Convert.specific_fn.loc10_46(%int_1.loc10_46) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_46.1: %i32 = value_of_initializer %int.convert_checked.loc10_46 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_46.2: %i32 = converted %int_1.loc10_46, %.loc10_46.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.right_shift.loc10: init %i32 = call %RightShift.ref.loc10(%.loc10_43.2, %.loc10_46.2) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc10_47.1: %i32 = value_of_initializer %int.right_shift.loc10 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc10_47.2: %i32 = converted %int.right_shift.loc10, %.loc10_47.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.snegate.loc10_48: init %i32 = call %Negate.ref.loc10_17(%.loc10_47.2) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_12.1: type = value_of_initializer %int.make_type_signed.loc10 [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_12.2: type = converted %int.make_type_signed.loc10, %.loc10_12.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc10_48: %Convert.type.6 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %Convert.bound.loc10_48: <bound method> = bound_method %int.snegate.loc10_48, %impl.elem0.loc10_48 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_48: <specific function> = specific_function %Convert.bound.loc10_48, @Convert.4(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %.loc10_48.1: %i32 = value_of_initializer %int.snegate.loc10_48 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc10_48.2: %i32 = converted %int.snegate.loc10_48, %.loc10_48.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.convert_checked.loc10_48: init Core.IntLiteral = call %Convert.specific_fn.loc10_48(%.loc10_48.2) [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc10_48.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc10_48 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc10_48.4: Core.IntLiteral = converted %int.snegate.loc10_48, %.loc10_48.3 [template = constants.%int_1.1]
// CHECK:STDOUT:   %array_type.loc10: type = array_type %.loc10_48.4, %i32 [template = constants.%array_type.1]
// CHECK:STDOUT:   %arr1.var: ref %array_type.1 = var arr1
// CHECK:STDOUT:   %arr1: ref %array_type.1 = bind_name arr1, %arr1.var
// CHECK:STDOUT:   %int_32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%int_32.loc11) [template = constants.%i32]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc11_14.1: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_14.2: type = converted %int.make_type_signed.loc11, %.loc11_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type.loc11: type = array_type %int_1.loc11, %i32 [template = constants.%array_type.1]
// CHECK:STDOUT:   %ptr.loc11: type = ptr_type %array_type.1 [template = constants.%ptr.1]
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14: init type = call constants.%Int(%int_32.loc14) [template = constants.%i32]
// CHECK:STDOUT:   %Negate.ref.loc14_17: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %RightShift.ref.loc14: %RightShift.type = name_ref RightShift, %RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %Negate.ref.loc14_35: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template = constants.%int_10.1]
// CHECK:STDOUT:   %impl.elem0.loc14_42: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc14_42: <bound method> = bound_method %int_10, %impl.elem0.loc14_42 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_42: <specific function> = specific_function %Convert.bound.loc14_42, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc14_42: init %i32 = call %Convert.specific_fn.loc14_42(%int_10) [template = constants.%int_10.2]
// CHECK:STDOUT:   %.loc14_42.1: %i32 = value_of_initializer %int.convert_checked.loc14_42 [template = constants.%int_10.2]
// CHECK:STDOUT:   %.loc14_42.2: %i32 = converted %int_10, %.loc14_42.1 [template = constants.%int_10.2]
// CHECK:STDOUT:   %int.snegate.loc14_44: init %i32 = call %Negate.ref.loc14_35(%.loc14_42.2) [template = constants.%int_-10]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc14_44.1: %i32 = value_of_initializer %int.snegate.loc14_44 [template = constants.%int_-10]
// CHECK:STDOUT:   %.loc14_44.2: %i32 = converted %int.snegate.loc14_44, %.loc14_44.1 [template = constants.%int_-10]
// CHECK:STDOUT:   %impl.elem0.loc14_47: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc14_47: <bound method> = bound_method %int_2, %impl.elem0.loc14_47 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_47: <specific function> = specific_function %Convert.bound.loc14_47, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc14_47: init %i32 = call %Convert.specific_fn.loc14_47(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_47.1: %i32 = value_of_initializer %int.convert_checked.loc14_47 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_47.2: %i32 = converted %int_2, %.loc14_47.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.right_shift.loc14: init %i32 = call %RightShift.ref.loc14(%.loc14_44.2, %.loc14_47.2) [template = constants.%int_-3]
// CHECK:STDOUT:   %.loc14_48.1: %i32 = value_of_initializer %int.right_shift.loc14 [template = constants.%int_-3]
// CHECK:STDOUT:   %.loc14_48.2: %i32 = converted %int.right_shift.loc14, %.loc14_48.1 [template = constants.%int_-3]
// CHECK:STDOUT:   %int.snegate.loc14_49: init %i32 = call %Negate.ref.loc14_17(%.loc14_48.2) [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc14_12.1: type = value_of_initializer %int.make_type_signed.loc14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_12.2: type = converted %int.make_type_signed.loc14, %.loc14_12.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc14_49: %Convert.type.6 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %Convert.bound.loc14_49: <bound method> = bound_method %int.snegate.loc14_49, %impl.elem0.loc14_49 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_49: <specific function> = specific_function %Convert.bound.loc14_49, @Convert.4(constants.%int_32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %.loc14_49.1: %i32 = value_of_initializer %int.snegate.loc14_49 [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc14_49.2: %i32 = converted %int.snegate.loc14_49, %.loc14_49.1 [template = constants.%int_3.1]
// CHECK:STDOUT:   %int.convert_checked.loc14_49: init Core.IntLiteral = call %Convert.specific_fn.loc14_49(%.loc14_49.2) [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc14_49.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14_49 [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc14_49.4: Core.IntLiteral = converted %int.snegate.loc14_49, %.loc14_49.3 [template = constants.%int_3.2]
// CHECK:STDOUT:   %array_type.loc14: type = array_type %.loc14_49.4, %i32 [template = constants.%array_type.2]
// CHECK:STDOUT:   %arr2.var: ref %array_type.2 = var arr2
// CHECK:STDOUT:   %arr2: ref %array_type.2 = bind_name arr2, %arr2.var
// CHECK:STDOUT:   %int_32.loc15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%int_32.loc15) [template = constants.%i32]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc15_14.1: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_14.2: type = converted %int.make_type_signed.loc15, %.loc15_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type.loc15: type = array_type %int_3, %i32 [template = constants.%array_type.2]
// CHECK:STDOUT:   %ptr.loc15: type = ptr_type %array_type.2 [template = constants.%ptr.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShift(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr1.ref: ref %array_type.1 = name_ref arr1, file.%arr1
// CHECK:STDOUT:   %addr.loc11: %ptr.1 = addr_of %arr1.ref
// CHECK:STDOUT:   %arr1_p: %ptr.1 = bind_name arr1_p, %addr.loc11
// CHECK:STDOUT:   %arr2.ref: ref %array_type.2 = name_ref arr2, file.%arr2
// CHECK:STDOUT:   %addr.loc15: %ptr.2 = addr_of %arr2.ref
// CHECK:STDOUT:   %arr2_p: %ptr.2 = bind_name arr2_p, %addr.loc15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32.1 [template]
// CHECK:STDOUT:   %RightShift.type: type = fn_type @RightShift [template]
// CHECK:STDOUT:   %RightShift: %RightShift.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_31.1: Core.IntLiteral = int_value 31 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_31.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_31.2: %i32 = int_value 31 [template]
// CHECK:STDOUT:   %int_0: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_32.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_32.2: %i32 = int_value 32 [template]
// CHECK:STDOUT:   %int_33.1: Core.IntLiteral = int_value 33 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_33.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_33.2: %i32 = int_value 33 [template]
// CHECK:STDOUT:   %int_-1: %i32 = int_value -1 [template]
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
// CHECK:STDOUT:     .RightShift = %RightShift.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .size_1 = @__global_init.%size_1
// CHECK:STDOUT:     .size_2 = @__global_init.%size_2
// CHECK:STDOUT:     .size_3 = @__global_init.%size_3
// CHECK:STDOUT:     .negative = @__global_init.%negative
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %RightShift.decl: %RightShift.type = fn_decl @RightShift [template = constants.%RightShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_18: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_18: init type = call constants.%Int(%int_32.loc4_18) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_18.1: type = value_of_initializer %int.make_type_signed.loc4_18 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_18.2: type = converted %int.make_type_signed.loc4_18, %.loc4_18.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_26: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_26: init type = call constants.%Int(%int_32.loc4_26) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_26.1: type = value_of_initializer %int.make_type_signed.loc4_26 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_26.2: type = converted %int.make_type_signed.loc4_26, %.loc4_26.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_34: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_34: init type = call constants.%Int(%int_32.loc4_34) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_34.1: type = value_of_initializer %int.make_type_signed.loc4_34 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_34.2: type = converted %int.make_type_signed.loc4_34, %.loc4_34.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5_14: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:     %int.make_type_signed.loc5_14: init type = call constants.%Int(%int_32.loc5_14) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_14.1: type = value_of_initializer %int.make_type_signed.loc5_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_14.2: type = converted %int.make_type_signed.loc5_14, %.loc5_14.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_22: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:     %int.make_type_signed.loc5_22: init type = call constants.%Int(%int_32.loc5_22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_22.1: type = value_of_initializer %int.make_type_signed.loc5_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_22.2: type = converted %int.make_type_signed.loc5_22, %.loc5_22.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%int_32.loc8) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_13.1: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_13.2: type = converted %int.make_type_signed.loc8, %.loc8_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%int_32.loc13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_13.1: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_13.2: type = converted %int.make_type_signed.loc13, %.loc13_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc18: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %int.make_type_signed.loc18: init type = call constants.%Int(%int_32.loc18) [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_13.1: type = value_of_initializer %int.make_type_signed.loc18 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_13.2: type = converted %int.make_type_signed.loc18, %.loc18_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc24: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %int.make_type_signed.loc24: init type = call constants.%Int(%int_32.loc24) [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_15.1: type = value_of_initializer %int.make_type_signed.loc24 [template = constants.%i32]
// CHECK:STDOUT:   %.loc24_15.2: type = converted %int.make_type_signed.loc24, %.loc24_15.1 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RightShift(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.right_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %RightShift.ref.loc8: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [template = constants.%int_31.1]
// CHECK:STDOUT:   %impl.elem0.loc8_30: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc8_30: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8_30 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_30: <specific function> = specific_function %Convert.bound.loc8_30, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc8_30: init %i32 = call %Convert.specific_fn.loc8_30(%int_1.loc8) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc8_30.1: %i32 = value_of_initializer %int.convert_checked.loc8_30 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc8_30.2: %i32 = converted %int_1.loc8, %.loc8_30.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc8_33: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc8_33: <bound method> = bound_method %int_31, %impl.elem0.loc8_33 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_33: <specific function> = specific_function %Convert.bound.loc8_33, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc8_33: init %i32 = call %Convert.specific_fn.loc8_33(%int_31) [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc8_33.1: %i32 = value_of_initializer %int.convert_checked.loc8_33 [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc8_33.2: %i32 = converted %int_31, %.loc8_33.1 [template = constants.%int_31.2]
// CHECK:STDOUT:   %int.right_shift.loc8: init %i32 = call %RightShift.ref.loc8(%.loc8_30.2, %.loc8_33.2) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc8_36.1: %i32 = value_of_initializer %int.right_shift.loc8 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc8_36.2: %i32 = converted %int.right_shift.loc8, %.loc8_36.1 [template = constants.%int_0]
// CHECK:STDOUT:   %size_1: %i32 = bind_name size_1, %.loc8_36.2
// CHECK:STDOUT:   %RightShift.ref.loc13: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %impl.elem0.loc13_30: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc13_30: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_30 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_30: <specific function> = specific_function %Convert.bound.loc13_30, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_30: init %i32 = call %Convert.specific_fn.loc13_30(%int_1.loc13) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_30.1: %i32 = value_of_initializer %int.convert_checked.loc13_30 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_30.2: %i32 = converted %int_1.loc13, %.loc13_30.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc13_33: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc13_33: <bound method> = bound_method %int_32, %impl.elem0.loc13_33 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_33: <specific function> = specific_function %Convert.bound.loc13_33, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc13_33: init %i32 = call %Convert.specific_fn.loc13_33(%int_32) [template = constants.%int_32.2]
// CHECK:STDOUT:   %.loc13_33.1: %i32 = value_of_initializer %int.convert_checked.loc13_33 [template = constants.%int_32.2]
// CHECK:STDOUT:   %.loc13_33.2: %i32 = converted %int_32, %.loc13_33.1 [template = constants.%int_32.2]
// CHECK:STDOUT:   %int.right_shift.loc13: init %i32 = call %RightShift.ref.loc13(%.loc13_30.2, %.loc13_33.2) [template = <error>]
// CHECK:STDOUT:   %.loc13_36.1: %i32 = value_of_initializer %int.right_shift.loc13 [template = <error>]
// CHECK:STDOUT:   %.loc13_36.2: %i32 = converted %int.right_shift.loc13, %.loc13_36.1 [template = <error>]
// CHECK:STDOUT:   %size_2: %i32 = bind_name size_2, %.loc13_36.2
// CHECK:STDOUT:   %RightShift.ref.loc18: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [template = constants.%int_33.1]
// CHECK:STDOUT:   %impl.elem0.loc18_30: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc18_30: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_30 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_30: <specific function> = specific_function %Convert.bound.loc18_30, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc18_30: init %i32 = call %Convert.specific_fn.loc18_30(%int_1.loc18) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc18_30.1: %i32 = value_of_initializer %int.convert_checked.loc18_30 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc18_30.2: %i32 = converted %int_1.loc18, %.loc18_30.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc18_33: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc18_33: <bound method> = bound_method %int_33, %impl.elem0.loc18_33 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_33: <specific function> = specific_function %Convert.bound.loc18_33, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc18_33: init %i32 = call %Convert.specific_fn.loc18_33(%int_33) [template = constants.%int_33.2]
// CHECK:STDOUT:   %.loc18_33.1: %i32 = value_of_initializer %int.convert_checked.loc18_33 [template = constants.%int_33.2]
// CHECK:STDOUT:   %.loc18_33.2: %i32 = converted %int_33, %.loc18_33.1 [template = constants.%int_33.2]
// CHECK:STDOUT:   %int.right_shift.loc18: init %i32 = call %RightShift.ref.loc18(%.loc18_30.2, %.loc18_33.2) [template = <error>]
// CHECK:STDOUT:   %.loc18_36.1: %i32 = value_of_initializer %int.right_shift.loc18 [template = <error>]
// CHECK:STDOUT:   %.loc18_36.2: %i32 = converted %int.right_shift.loc18, %.loc18_36.1 [template = <error>]
// CHECK:STDOUT:   %size_3: %i32 = bind_name size_3, %.loc18_36.2
// CHECK:STDOUT:   %RightShift.ref.loc24: %RightShift.type = name_ref RightShift, file.%RightShift.decl [template = constants.%RightShift]
// CHECK:STDOUT:   %int_1.loc24_32: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc24_42: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc24_42: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc24_42: <bound method> = bound_method %int_1.loc24_42, %impl.elem0.loc24_42 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc24_42: <specific function> = specific_function %Convert.bound.loc24_42, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc24_42: init %i32 = call %Convert.specific_fn.loc24_42(%int_1.loc24_42) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc24_42.1: %i32 = value_of_initializer %int.convert_checked.loc24_42 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc24_42.2: %i32 = converted %int_1.loc24_42, %.loc24_42.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.snegate: init %i32 = call %Negate.ref(%.loc24_42.2) [template = constants.%int_-1]
// CHECK:STDOUT:   %impl.elem0.loc24_32: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc24_32: <bound method> = bound_method %int_1.loc24_32, %impl.elem0.loc24_32 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc24_32: <specific function> = specific_function %Convert.bound.loc24_32, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc24_32: init %i32 = call %Convert.specific_fn.loc24_32(%int_1.loc24_32) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc24_32.1: %i32 = value_of_initializer %int.convert_checked.loc24_32 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc24_32.2: %i32 = converted %int_1.loc24_32, %.loc24_32.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc24_43.1: %i32 = value_of_initializer %int.snegate [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc24_43.2: %i32 = converted %int.snegate, %.loc24_43.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.right_shift.loc24: init %i32 = call %RightShift.ref.loc24(%.loc24_32.2, %.loc24_43.2) [template = <error>]
// CHECK:STDOUT:   %.loc24_45.1: %i32 = value_of_initializer %int.right_shift.loc24 [template = <error>]
// CHECK:STDOUT:   %.loc24_45.2: %i32 = converted %int.right_shift.loc24, %.loc24_45.1 [template = <error>]
// CHECK:STDOUT:   %negative: %i32 = bind_name negative, %.loc24_45.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
