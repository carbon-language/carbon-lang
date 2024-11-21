// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/left_shift.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/left_shift.carbon

// --- int_left_shift.carbon

fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";

var arr: [i32; LeftShift(5, 2)];
let arr_p: [i32; 20]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return LeftShift(a, b);
}

// TODO: Test mixed types for LHS and RHS.

// --- fail_bad_shift.carbon

package BadShift;

fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
fn Negate(a: i32) -> i32 = "int.snegate";

// Shift greater than size is disallowed.
let size_1: i32 = LeftShift(1, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance not in range [0, 32) in 1 << 32 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_2: i32 = LeftShift(1, 32);
// CHECK:STDERR:                   ^~~~~~~~~~
// CHECK:STDERR:
let size_2: i32 = LeftShift(1, 32);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance not in range [0, 32) in 1 << 33 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_3: i32 = LeftShift(1, 33);
// CHECK:STDERR:                   ^~~~~~~~~~
// CHECK:STDERR:
let size_3: i32 = LeftShift(1, 33);

// Overflow is allowed if the shift distance is in bounds.
let overflow_1: i32 = LeftShift(1000, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:23: error: shift distance not in range [0, 32) in 1000 << 32 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let overflow_2: i32 = LeftShift(1000, 32);
// CHECK:STDERR:                       ^~~~~~~~~~
// CHECK:STDERR:
let overflow_2: i32 = LeftShift(1000, 32);

// Oversize shifts aren't allowed even if there's no overflow.
let no_overflow_1: i32 = LeftShift(0, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:26: error: shift distance not in range [0, 32) in 0 << 32 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let no_overflow_2: i32 = LeftShift(0, 32);
// CHECK:STDERR:                          ^~~~~~~~~~
// CHECK:STDERR:
let no_overflow_2: i32 = LeftShift(0, 32);

// Negative shifts aren't allowed either.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+3]]:21: error: shift distance not in range [0, 32) in 1 << -1 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let negative: i32 = LeftShift(1, Negate(1));
// CHECK:STDERR:                     ^~~~~~~~~~
let negative: i32 = LeftShift(1, Negate(1));

// CHECK:STDOUT: --- int_left_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %LeftShift.type: type = fn_type @LeftShift [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.30: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 20 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%.1) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.35: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.34, %Convert.15 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.4(%.1) [template]
// CHECK:STDOUT:   %.38: Core.IntLiteral = int_value 20 [template]
// CHECK:STDOUT:   %.39: type = array_type %.38, %i32 [template]
// CHECK:STDOUT:   %.40: type = ptr_type %.39 [template]
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
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LeftShift.decl: %LeftShift.type = fn_decl @LeftShift [template = constants.%LeftShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc2_17.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_17: init type = call constants.%Int(%.loc2_17.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_17.2: type = value_of_initializer %int.make_type_signed.loc2_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_17.3: type = converted %int.make_type_signed.loc2_17, %.loc2_17.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_25.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_25: init type = call constants.%Int(%.loc2_25.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_25.2: type = value_of_initializer %int.make_type_signed.loc2_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_25.3: type = converted %int.make_type_signed.loc2_25, %.loc2_25.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_33.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_33: init type = call constants.%Int(%.loc2_33.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_33.2: type = value_of_initializer %int.make_type_signed.loc2_33 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_33.3: type = converted %int.make_type_signed.loc2_33, %.loc2_33.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%.loc4_11.1) [template = constants.%i32]
// CHECK:STDOUT:   %LeftShift.ref: %LeftShift.type = name_ref LeftShift, %LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc4_26.1: Core.IntLiteral = int_value 5 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_29.1: Core.IntLiteral = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_26.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_26.3: <bound method> = bound_method %.loc4_26.1, %.loc4_26.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc4_26.4: <specific function> = specific_function %.loc4_26.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc4_26: init %i32 = call %.loc4_26.4(%.loc4_26.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_26.5: %i32 = value_of_initializer %int.convert_checked.loc4_26 [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_26.6: %i32 = converted %.loc4_26.1, %.loc4_26.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_29.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_29.3: <bound method> = bound_method %.loc4_29.1, %.loc4_29.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc4_29.4: <specific function> = specific_function %.loc4_29.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc4_29: init %i32 = call %.loc4_29.4(%.loc4_29.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_29.5: %i32 = value_of_initializer %int.convert_checked.loc4_29 [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_29.6: %i32 = converted %.loc4_29.1, %.loc4_29.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.left_shift: init %i32 = call %LeftShift.ref(%.loc4_26.6, %.loc4_29.6) [template = constants.%.34]
// CHECK:STDOUT:   %.loc4_11.2: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_11.3: type = converted %int.make_type_signed.loc4, %.loc4_11.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_25.1: %Convert.type.6 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc4_25.2: <bound method> = bound_method %int.left_shift, %.loc4_25.1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc4_25.3: <specific function> = specific_function %.loc4_25.2, @Convert.4(constants.%.1) [template = constants.%.37]
// CHECK:STDOUT:   %.loc4_25.4: %i32 = value_of_initializer %int.left_shift [template = constants.%.34]
// CHECK:STDOUT:   %.loc4_25.5: %i32 = converted %int.left_shift, %.loc4_25.4 [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc4_25: init Core.IntLiteral = call %.loc4_25.3(%.loc4_25.5) [template = constants.%.38]
// CHECK:STDOUT:   %.loc4_25.6: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4_25 [template = constants.%.38]
// CHECK:STDOUT:   %.loc4_25.7: Core.IntLiteral = converted %int.left_shift, %.loc4_25.6 [template = constants.%.38]
// CHECK:STDOUT:   %.loc4_31: type = array_type %.loc4_25.7, %i32 [template = constants.%.39]
// CHECK:STDOUT:   %arr.var: ref %.39 = var arr
// CHECK:STDOUT:   %arr: ref %.39 = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc5_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%.loc5_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_18: Core.IntLiteral = int_value 20 [template = constants.%.38]
// CHECK:STDOUT:   %.loc5_13.2: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_13.3: type = converted %int.make_type_signed.loc5, %.loc5_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_20: type = array_type %.loc5_18, %i32 [template = constants.%.39]
// CHECK:STDOUT:   %.loc5_21: type = ptr_type %.39 [template = constants.%.40]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc7_19: init type = call constants.%Int(%.loc7_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.2: type = value_of_initializer %int.make_type_signed.loc7_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.3: type = converted %int.make_type_signed.loc7_19, %.loc7_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc7_27: init type = call constants.%Int(%.loc7_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.2: type = value_of_initializer %int.make_type_signed.loc7_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.3: type = converted %int.make_type_signed.loc7_27, %.loc7_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_35.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc7_35: init type = call constants.%Int(%.loc7_35.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_35.2: type = value_of_initializer %int.make_type_signed.loc7_35 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_35.3: type = converted %int.make_type_signed.loc7_35, %.loc7_35.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.left_shift: init %i32 = call %LeftShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_25.1: %i32 = value_of_initializer %int.left_shift
// CHECK:STDOUT:   %.loc8_25.2: %i32 = converted %int.left_shift, %.loc8_25.1
// CHECK:STDOUT:   return %.loc8_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %.39 = name_ref arr, file.%arr
// CHECK:STDOUT:   %.loc5: %.40 = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %.40 = bind_name arr_p, %.loc5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %LeftShift.type: type = fn_type @LeftShift [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 31 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.30: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 31 [template]
// CHECK:STDOUT:   %.34: %i32 = int_value -2147483648 [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.1, %Convert.14 [template]
// CHECK:STDOUT:   %.36: <specific function> = specific_function %.35, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.37: %i32 = int_value 32 [template]
// CHECK:STDOUT:   %.38: Core.IntLiteral = int_value 33 [template]
// CHECK:STDOUT:   %.39: <bound method> = bound_method %.38, %Convert.14 [template]
// CHECK:STDOUT:   %.40: <specific function> = specific_function %.39, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.41: %i32 = int_value 33 [template]
// CHECK:STDOUT:   %.42: Core.IntLiteral = int_value 1000 [template]
// CHECK:STDOUT:   %.43: <bound method> = bound_method %.42, %Convert.14 [template]
// CHECK:STDOUT:   %.44: <specific function> = specific_function %.43, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.45: %i32 = int_value 1000 [template]
// CHECK:STDOUT:   %.46: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %.47: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.48: <bound method> = bound_method %.47, %Convert.14 [template]
// CHECK:STDOUT:   %.49: <specific function> = specific_function %.48, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.50: %i32 = int_value -1 [template]
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
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .size_1 = @__global_init.%size_1
// CHECK:STDOUT:     .size_2 = @__global_init.%size_2
// CHECK:STDOUT:     .size_3 = @__global_init.%size_3
// CHECK:STDOUT:     .overflow_1 = @__global_init.%overflow_1
// CHECK:STDOUT:     .overflow_2 = @__global_init.%overflow_2
// CHECK:STDOUT:     .no_overflow_1 = @__global_init.%no_overflow_1
// CHECK:STDOUT:     .no_overflow_2 = @__global_init.%no_overflow_2
// CHECK:STDOUT:     .negative = @__global_init.%negative
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LeftShift.decl: %LeftShift.type = fn_decl @LeftShift [template = constants.%LeftShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_17: init type = call constants.%Int(%.loc4_17.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.2: type = value_of_initializer %int.make_type_signed.loc4_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.3: type = converted %int.make_type_signed.loc4_17, %.loc4_17.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_25.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_25: init type = call constants.%Int(%.loc4_25.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_25.2: type = value_of_initializer %int.make_type_signed.loc4_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_25.3: type = converted %int.make_type_signed.loc4_25, %.loc4_25.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_33.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_33: init type = call constants.%Int(%.loc4_33.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_33.2: type = value_of_initializer %int.make_type_signed.loc4_33 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_33.3: type = converted %int.make_type_signed.loc4_33, %.loc4_33.2 [template = constants.%i32]
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
// CHECK:STDOUT:     %.loc5_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc5_14: init type = call constants.%Int(%.loc5_14.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_14.2: type = value_of_initializer %int.make_type_signed.loc5_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_14.3: type = converted %int.make_type_signed.loc5_14, %.loc5_14.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc5_22: init type = call constants.%Int(%.loc5_22.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_22.2: type = value_of_initializer %int.make_type_signed.loc5_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_22.3: type = converted %int.make_type_signed.loc5_22, %.loc5_22.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%.loc8_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_13.2: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_13.3: type = converted %int.make_type_signed.loc8, %.loc8_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%.loc13_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_13.2: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_13.3: type = converted %int.make_type_signed.loc13, %.loc13_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc18: init type = call constants.%Int(%.loc18_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_13.2: type = value_of_initializer %int.make_type_signed.loc18 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_13.3: type = converted %int.make_type_signed.loc18, %.loc18_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_17.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc21: init type = call constants.%Int(%.loc21_17.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_17.2: type = value_of_initializer %int.make_type_signed.loc21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_17.3: type = converted %int.make_type_signed.loc21, %.loc21_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc26_17.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc26: init type = call constants.%Int(%.loc26_17.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc26_17.2: type = value_of_initializer %int.make_type_signed.loc26 [template = constants.%i32]
// CHECK:STDOUT:   %.loc26_17.3: type = converted %int.make_type_signed.loc26, %.loc26_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc29_20.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc29: init type = call constants.%Int(%.loc29_20.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc29_20.2: type = value_of_initializer %int.make_type_signed.loc29 [template = constants.%i32]
// CHECK:STDOUT:   %.loc29_20.3: type = converted %int.make_type_signed.loc29, %.loc29_20.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc34_20.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc34: init type = call constants.%Int(%.loc34_20.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc34_20.2: type = value_of_initializer %int.make_type_signed.loc34 [template = constants.%i32]
// CHECK:STDOUT:   %.loc34_20.3: type = converted %int.make_type_signed.loc34, %.loc34_20.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc40_15.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc40: init type = call constants.%Int(%.loc40_15.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc40_15.2: type = value_of_initializer %int.make_type_signed.loc40 [template = constants.%i32]
// CHECK:STDOUT:   %.loc40_15.3: type = converted %int.make_type_signed.loc40, %.loc40_15.2 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref.loc8: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc8_29.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_32.1: Core.IntLiteral = int_value 31 [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_29.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc8_29.3: <bound method> = bound_method %.loc8_29.1, %.loc8_29.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc8_29.4: <specific function> = specific_function %.loc8_29.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc8_29: init %i32 = call %.loc8_29.4(%.loc8_29.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc8_29.5: %i32 = value_of_initializer %int.convert_checked.loc8_29 [template = constants.%.30]
// CHECK:STDOUT:   %.loc8_29.6: %i32 = converted %.loc8_29.1, %.loc8_29.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc8_32.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc8_32.3: <bound method> = bound_method %.loc8_32.1, %.loc8_32.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc8_32.4: <specific function> = specific_function %.loc8_32.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc8_32: init %i32 = call %.loc8_32.4(%.loc8_32.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc8_32.5: %i32 = value_of_initializer %int.convert_checked.loc8_32 [template = constants.%.33]
// CHECK:STDOUT:   %.loc8_32.6: %i32 = converted %.loc8_32.1, %.loc8_32.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.left_shift.loc8: init %i32 = call %LeftShift.ref.loc8(%.loc8_29.6, %.loc8_32.6) [template = constants.%.34]
// CHECK:STDOUT:   %.loc8_35.1: %i32 = value_of_initializer %int.left_shift.loc8 [template = constants.%.34]
// CHECK:STDOUT:   %.loc8_35.2: %i32 = converted %int.left_shift.loc8, %.loc8_35.1 [template = constants.%.34]
// CHECK:STDOUT:   %size_1: %i32 = bind_name size_1, %.loc8_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc13: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc13_29.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_32.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_29.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_29.3: <bound method> = bound_method %.loc13_29.1, %.loc13_29.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc13_29.4: <specific function> = specific_function %.loc13_29.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc13_29: init %i32 = call %.loc13_29.4(%.loc13_29.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc13_29.5: %i32 = value_of_initializer %int.convert_checked.loc13_29 [template = constants.%.30]
// CHECK:STDOUT:   %.loc13_29.6: %i32 = converted %.loc13_29.1, %.loc13_29.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc13_32.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_32.3: <bound method> = bound_method %.loc13_32.1, %.loc13_32.2 [template = constants.%.35]
// CHECK:STDOUT:   %.loc13_32.4: <specific function> = specific_function %.loc13_32.3, @Convert.2(constants.%.1) [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc13_32: init %i32 = call %.loc13_32.4(%.loc13_32.1) [template = constants.%.37]
// CHECK:STDOUT:   %.loc13_32.5: %i32 = value_of_initializer %int.convert_checked.loc13_32 [template = constants.%.37]
// CHECK:STDOUT:   %.loc13_32.6: %i32 = converted %.loc13_32.1, %.loc13_32.5 [template = constants.%.37]
// CHECK:STDOUT:   %int.left_shift.loc13: init %i32 = call %LeftShift.ref.loc13(%.loc13_29.6, %.loc13_32.6) [template = <error>]
// CHECK:STDOUT:   %.loc13_35.1: %i32 = value_of_initializer %int.left_shift.loc13 [template = <error>]
// CHECK:STDOUT:   %.loc13_35.2: %i32 = converted %int.left_shift.loc13, %.loc13_35.1 [template = <error>]
// CHECK:STDOUT:   %size_2: %i32 = bind_name size_2, %.loc13_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc18: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc18_29.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc18_32.1: Core.IntLiteral = int_value 33 [template = constants.%.38]
// CHECK:STDOUT:   %.loc18_29.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc18_29.3: <bound method> = bound_method %.loc18_29.1, %.loc18_29.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc18_29.4: <specific function> = specific_function %.loc18_29.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc18_29: init %i32 = call %.loc18_29.4(%.loc18_29.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc18_29.5: %i32 = value_of_initializer %int.convert_checked.loc18_29 [template = constants.%.30]
// CHECK:STDOUT:   %.loc18_29.6: %i32 = converted %.loc18_29.1, %.loc18_29.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc18_32.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc18_32.3: <bound method> = bound_method %.loc18_32.1, %.loc18_32.2 [template = constants.%.39]
// CHECK:STDOUT:   %.loc18_32.4: <specific function> = specific_function %.loc18_32.3, @Convert.2(constants.%.1) [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc18_32: init %i32 = call %.loc18_32.4(%.loc18_32.1) [template = constants.%.41]
// CHECK:STDOUT:   %.loc18_32.5: %i32 = value_of_initializer %int.convert_checked.loc18_32 [template = constants.%.41]
// CHECK:STDOUT:   %.loc18_32.6: %i32 = converted %.loc18_32.1, %.loc18_32.5 [template = constants.%.41]
// CHECK:STDOUT:   %int.left_shift.loc18: init %i32 = call %LeftShift.ref.loc18(%.loc18_29.6, %.loc18_32.6) [template = <error>]
// CHECK:STDOUT:   %.loc18_35.1: %i32 = value_of_initializer %int.left_shift.loc18 [template = <error>]
// CHECK:STDOUT:   %.loc18_35.2: %i32 = converted %int.left_shift.loc18, %.loc18_35.1 [template = <error>]
// CHECK:STDOUT:   %size_3: %i32 = bind_name size_3, %.loc18_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc21: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc21_33.1: Core.IntLiteral = int_value 1000 [template = constants.%.42]
// CHECK:STDOUT:   %.loc21_39.1: Core.IntLiteral = int_value 31 [template = constants.%.3]
// CHECK:STDOUT:   %.loc21_33.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc21_33.3: <bound method> = bound_method %.loc21_33.1, %.loc21_33.2 [template = constants.%.43]
// CHECK:STDOUT:   %.loc21_33.4: <specific function> = specific_function %.loc21_33.3, @Convert.2(constants.%.1) [template = constants.%.44]
// CHECK:STDOUT:   %int.convert_checked.loc21_33: init %i32 = call %.loc21_33.4(%.loc21_33.1) [template = constants.%.45]
// CHECK:STDOUT:   %.loc21_33.5: %i32 = value_of_initializer %int.convert_checked.loc21_33 [template = constants.%.45]
// CHECK:STDOUT:   %.loc21_33.6: %i32 = converted %.loc21_33.1, %.loc21_33.5 [template = constants.%.45]
// CHECK:STDOUT:   %.loc21_39.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc21_39.3: <bound method> = bound_method %.loc21_39.1, %.loc21_39.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc21_39.4: <specific function> = specific_function %.loc21_39.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc21_39: init %i32 = call %.loc21_39.4(%.loc21_39.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc21_39.5: %i32 = value_of_initializer %int.convert_checked.loc21_39 [template = constants.%.33]
// CHECK:STDOUT:   %.loc21_39.6: %i32 = converted %.loc21_39.1, %.loc21_39.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.left_shift.loc21: init %i32 = call %LeftShift.ref.loc21(%.loc21_33.6, %.loc21_39.6) [template = constants.%.46]
// CHECK:STDOUT:   %.loc21_42.1: %i32 = value_of_initializer %int.left_shift.loc21 [template = constants.%.46]
// CHECK:STDOUT:   %.loc21_42.2: %i32 = converted %int.left_shift.loc21, %.loc21_42.1 [template = constants.%.46]
// CHECK:STDOUT:   %overflow_1: %i32 = bind_name overflow_1, %.loc21_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc26: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc26_33.1: Core.IntLiteral = int_value 1000 [template = constants.%.42]
// CHECK:STDOUT:   %.loc26_39.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %.loc26_33.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc26_33.3: <bound method> = bound_method %.loc26_33.1, %.loc26_33.2 [template = constants.%.43]
// CHECK:STDOUT:   %.loc26_33.4: <specific function> = specific_function %.loc26_33.3, @Convert.2(constants.%.1) [template = constants.%.44]
// CHECK:STDOUT:   %int.convert_checked.loc26_33: init %i32 = call %.loc26_33.4(%.loc26_33.1) [template = constants.%.45]
// CHECK:STDOUT:   %.loc26_33.5: %i32 = value_of_initializer %int.convert_checked.loc26_33 [template = constants.%.45]
// CHECK:STDOUT:   %.loc26_33.6: %i32 = converted %.loc26_33.1, %.loc26_33.5 [template = constants.%.45]
// CHECK:STDOUT:   %.loc26_39.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc26_39.3: <bound method> = bound_method %.loc26_39.1, %.loc26_39.2 [template = constants.%.35]
// CHECK:STDOUT:   %.loc26_39.4: <specific function> = specific_function %.loc26_39.3, @Convert.2(constants.%.1) [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc26_39: init %i32 = call %.loc26_39.4(%.loc26_39.1) [template = constants.%.37]
// CHECK:STDOUT:   %.loc26_39.5: %i32 = value_of_initializer %int.convert_checked.loc26_39 [template = constants.%.37]
// CHECK:STDOUT:   %.loc26_39.6: %i32 = converted %.loc26_39.1, %.loc26_39.5 [template = constants.%.37]
// CHECK:STDOUT:   %int.left_shift.loc26: init %i32 = call %LeftShift.ref.loc26(%.loc26_33.6, %.loc26_39.6) [template = <error>]
// CHECK:STDOUT:   %.loc26_42.1: %i32 = value_of_initializer %int.left_shift.loc26 [template = <error>]
// CHECK:STDOUT:   %.loc26_42.2: %i32 = converted %int.left_shift.loc26, %.loc26_42.1 [template = <error>]
// CHECK:STDOUT:   %overflow_2: %i32 = bind_name overflow_2, %.loc26_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc29: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc29_36.1: Core.IntLiteral = int_value 0 [template = constants.%.47]
// CHECK:STDOUT:   %.loc29_39.1: Core.IntLiteral = int_value 31 [template = constants.%.3]
// CHECK:STDOUT:   %.loc29_36.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc29_36.3: <bound method> = bound_method %.loc29_36.1, %.loc29_36.2 [template = constants.%.48]
// CHECK:STDOUT:   %.loc29_36.4: <specific function> = specific_function %.loc29_36.3, @Convert.2(constants.%.1) [template = constants.%.49]
// CHECK:STDOUT:   %int.convert_checked.loc29_36: init %i32 = call %.loc29_36.4(%.loc29_36.1) [template = constants.%.46]
// CHECK:STDOUT:   %.loc29_36.5: %i32 = value_of_initializer %int.convert_checked.loc29_36 [template = constants.%.46]
// CHECK:STDOUT:   %.loc29_36.6: %i32 = converted %.loc29_36.1, %.loc29_36.5 [template = constants.%.46]
// CHECK:STDOUT:   %.loc29_39.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc29_39.3: <bound method> = bound_method %.loc29_39.1, %.loc29_39.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc29_39.4: <specific function> = specific_function %.loc29_39.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc29_39: init %i32 = call %.loc29_39.4(%.loc29_39.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc29_39.5: %i32 = value_of_initializer %int.convert_checked.loc29_39 [template = constants.%.33]
// CHECK:STDOUT:   %.loc29_39.6: %i32 = converted %.loc29_39.1, %.loc29_39.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.left_shift.loc29: init %i32 = call %LeftShift.ref.loc29(%.loc29_36.6, %.loc29_39.6) [template = constants.%.46]
// CHECK:STDOUT:   %.loc29_42.1: %i32 = value_of_initializer %int.left_shift.loc29 [template = constants.%.46]
// CHECK:STDOUT:   %.loc29_42.2: %i32 = converted %int.left_shift.loc29, %.loc29_42.1 [template = constants.%.46]
// CHECK:STDOUT:   %no_overflow_1: %i32 = bind_name no_overflow_1, %.loc29_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc34: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc34_36.1: Core.IntLiteral = int_value 0 [template = constants.%.47]
// CHECK:STDOUT:   %.loc34_39.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %.loc34_36.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc34_36.3: <bound method> = bound_method %.loc34_36.1, %.loc34_36.2 [template = constants.%.48]
// CHECK:STDOUT:   %.loc34_36.4: <specific function> = specific_function %.loc34_36.3, @Convert.2(constants.%.1) [template = constants.%.49]
// CHECK:STDOUT:   %int.convert_checked.loc34_36: init %i32 = call %.loc34_36.4(%.loc34_36.1) [template = constants.%.46]
// CHECK:STDOUT:   %.loc34_36.5: %i32 = value_of_initializer %int.convert_checked.loc34_36 [template = constants.%.46]
// CHECK:STDOUT:   %.loc34_36.6: %i32 = converted %.loc34_36.1, %.loc34_36.5 [template = constants.%.46]
// CHECK:STDOUT:   %.loc34_39.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc34_39.3: <bound method> = bound_method %.loc34_39.1, %.loc34_39.2 [template = constants.%.35]
// CHECK:STDOUT:   %.loc34_39.4: <specific function> = specific_function %.loc34_39.3, @Convert.2(constants.%.1) [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc34_39: init %i32 = call %.loc34_39.4(%.loc34_39.1) [template = constants.%.37]
// CHECK:STDOUT:   %.loc34_39.5: %i32 = value_of_initializer %int.convert_checked.loc34_39 [template = constants.%.37]
// CHECK:STDOUT:   %.loc34_39.6: %i32 = converted %.loc34_39.1, %.loc34_39.5 [template = constants.%.37]
// CHECK:STDOUT:   %int.left_shift.loc34: init %i32 = call %LeftShift.ref.loc34(%.loc34_36.6, %.loc34_39.6) [template = <error>]
// CHECK:STDOUT:   %.loc34_42.1: %i32 = value_of_initializer %int.left_shift.loc34 [template = <error>]
// CHECK:STDOUT:   %.loc34_42.2: %i32 = converted %int.left_shift.loc34, %.loc34_42.1 [template = <error>]
// CHECK:STDOUT:   %no_overflow_2: %i32 = bind_name no_overflow_2, %.loc34_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc40: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %.loc40_31.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc40_41.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc40_41.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc40_41.3: <bound method> = bound_method %.loc40_41.1, %.loc40_41.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc40_41.4: <specific function> = specific_function %.loc40_41.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc40_41: init %i32 = call %.loc40_41.4(%.loc40_41.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc40_41.5: %i32 = value_of_initializer %int.convert_checked.loc40_41 [template = constants.%.30]
// CHECK:STDOUT:   %.loc40_41.6: %i32 = converted %.loc40_41.1, %.loc40_41.5 [template = constants.%.30]
// CHECK:STDOUT:   %int.snegate: init %i32 = call %Negate.ref(%.loc40_41.6) [template = constants.%.50]
// CHECK:STDOUT:   %.loc40_31.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc40_31.3: <bound method> = bound_method %.loc40_31.1, %.loc40_31.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc40_31.4: <specific function> = specific_function %.loc40_31.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc40_31: init %i32 = call %.loc40_31.4(%.loc40_31.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc40_31.5: %i32 = value_of_initializer %int.convert_checked.loc40_31 [template = constants.%.30]
// CHECK:STDOUT:   %.loc40_31.6: %i32 = converted %.loc40_31.1, %.loc40_31.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc40_40.1: %i32 = value_of_initializer %int.snegate [template = constants.%.50]
// CHECK:STDOUT:   %.loc40_40.2: %i32 = converted %int.snegate, %.loc40_40.1 [template = constants.%.50]
// CHECK:STDOUT:   %int.left_shift.loc40: init %i32 = call %LeftShift.ref.loc40(%.loc40_31.6, %.loc40_40.2) [template = <error>]
// CHECK:STDOUT:   %.loc40_44.1: %i32 = value_of_initializer %int.left_shift.loc40 [template = <error>]
// CHECK:STDOUT:   %.loc40_44.2: %i32 = converted %int.left_shift.loc40, %.loc40_44.1 [template = <error>]
// CHECK:STDOUT:   %negative: %i32 = bind_name negative, %.loc40_44.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
