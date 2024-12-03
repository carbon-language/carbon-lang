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
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let size_2: i32 = LeftShift(1, 32);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance not in range [0, 32) in 1 << 33 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_3: i32 = LeftShift(1, 33);
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let size_3: i32 = LeftShift(1, 33);

// Overflow is allowed if the shift distance is in bounds.
let overflow_1: i32 = LeftShift(1000, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:23: error: shift distance not in range [0, 32) in 1000 << 32 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let overflow_2: i32 = LeftShift(1000, 32);
// CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let overflow_2: i32 = LeftShift(1000, 32);

// Oversize shifts aren't allowed even if there's no overflow.
let no_overflow_1: i32 = LeftShift(0, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:26: error: shift distance not in range [0, 32) in 0 << 32 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let no_overflow_2: i32 = LeftShift(0, 32);
// CHECK:STDERR:                          ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let no_overflow_2: i32 = LeftShift(0, 32);

// Negative shifts aren't allowed either.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+3]]:21: error: shift distance not in range [0, 32) in 1 << -1 [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let negative: i32 = LeftShift(1, Negate(1));
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~~~~~~
let negative: i32 = LeftShift(1, Negate(1));

// CHECK:STDOUT: --- int_left_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %LeftShift.type: type = fn_type @LeftShift [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type = struct_value () [template]
// CHECK:STDOUT:   %int.5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int.2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int.5.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int.20.1: %i32 = int_value 20 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%int.32) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %interface.10: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int.20.1, %Convert.15 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.4(%int.32) [template]
// CHECK:STDOUT:   %int.20.2: Core.IntLiteral = int_value 20 [template]
// CHECK:STDOUT:   %.5: type = array_type %int.20.2, %i32 [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %.5 [template]
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
// CHECK:STDOUT:     %int.32.loc2_17: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_17: init type = call constants.%Int(%int.32.loc2_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_17.1: type = value_of_initializer %int.make_type_signed.loc2_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_17.2: type = converted %int.make_type_signed.loc2_17, %.loc2_17.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc2_25: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_25: init type = call constants.%Int(%int.32.loc2_25) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_25.1: type = value_of_initializer %int.make_type_signed.loc2_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_25.2: type = converted %int.make_type_signed.loc2_25, %.loc2_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc2_33: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_33: init type = call constants.%Int(%int.32.loc2_33) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_33.1: type = value_of_initializer %int.make_type_signed.loc2_33 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_33.2: type = converted %int.make_type_signed.loc2_33, %.loc2_33.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.32.loc4: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%int.32.loc4) [template = constants.%i32]
// CHECK:STDOUT:   %LeftShift.ref: %LeftShift.type = name_ref LeftShift, %LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int.5: Core.IntLiteral = int_value 5 [template = constants.%int.5.1]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc4_26.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc4_26: <bound method> = bound_method %int.5, %.loc4_26.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc4_26: <specific function> = specific_function %Convert.bound.loc4_26, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc4_26: init %i32 = call %Convert.specific_fn.loc4_26(%int.5) [template = constants.%int.5.2]
// CHECK:STDOUT:   %.loc4_26.2: %i32 = value_of_initializer %int.convert_checked.loc4_26 [template = constants.%int.5.2]
// CHECK:STDOUT:   %.loc4_26.3: %i32 = converted %int.5, %.loc4_26.2 [template = constants.%int.5.2]
// CHECK:STDOUT:   %.loc4_29.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc4_29: <bound method> = bound_method %int.2, %.loc4_29.1 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc4_29: <specific function> = specific_function %Convert.bound.loc4_29, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc4_29: init %i32 = call %Convert.specific_fn.loc4_29(%int.2) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc4_29.2: %i32 = value_of_initializer %int.convert_checked.loc4_29 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc4_29.3: %i32 = converted %int.2, %.loc4_29.2 [template = constants.%int.2.2]
// CHECK:STDOUT:   %int.left_shift: init %i32 = call %LeftShift.ref(%.loc4_26.3, %.loc4_29.3) [template = constants.%int.20.1]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_signed.loc4, %.loc4_11.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_30.1: %Convert.type.6 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %Convert.bound.loc4_30: <bound method> = bound_method %int.left_shift, %.loc4_30.1 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc4_30: <specific function> = specific_function %Convert.bound.loc4_30, @Convert.4(constants.%int.32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %.loc4_30.2: %i32 = value_of_initializer %int.left_shift [template = constants.%int.20.1]
// CHECK:STDOUT:   %.loc4_30.3: %i32 = converted %int.left_shift, %.loc4_30.2 [template = constants.%int.20.1]
// CHECK:STDOUT:   %int.convert_checked.loc4_30: init Core.IntLiteral = call %Convert.specific_fn.loc4_30(%.loc4_30.3) [template = constants.%int.20.2]
// CHECK:STDOUT:   %.loc4_30.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4_30 [template = constants.%int.20.2]
// CHECK:STDOUT:   %.loc4_30.5: Core.IntLiteral = converted %int.left_shift, %.loc4_30.4 [template = constants.%int.20.2]
// CHECK:STDOUT:   %.loc4_31: type = array_type %.loc4_30.5, %i32 [template = constants.%.5]
// CHECK:STDOUT:   %arr.var: ref %.5 = var arr
// CHECK:STDOUT:   %arr: ref %.5 = bind_name arr, %arr.var
// CHECK:STDOUT:   %int.32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%int.32.loc5) [template = constants.%i32]
// CHECK:STDOUT:   %int.20: Core.IntLiteral = int_value 20 [template = constants.%int.20.2]
// CHECK:STDOUT:   %.loc5_13.1: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_13.2: type = converted %int.make_type_signed.loc5, %.loc5_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_20: type = array_type %int.20, %i32 [template = constants.%.5]
// CHECK:STDOUT:   %ptr: type = ptr_type %.5 [template = constants.%ptr]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32.loc7_19: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc7_19: init type = call constants.%Int(%int.32.loc7_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.1: type = value_of_initializer %int.make_type_signed.loc7_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.2: type = converted %int.make_type_signed.loc7_19, %.loc7_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc7_27: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc7_27: init type = call constants.%Int(%int.32.loc7_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.1: type = value_of_initializer %int.make_type_signed.loc7_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.2: type = converted %int.make_type_signed.loc7_27, %.loc7_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc7_35: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc7_35: init type = call constants.%Int(%int.32.loc7_35) [template = constants.%i32]
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
// CHECK:STDOUT:   %arr.ref: ref %.5 = name_ref arr, file.%arr
// CHECK:STDOUT:   %.loc5: %ptr = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %ptr = bind_name arr_p, %.loc5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32.1 [template]
// CHECK:STDOUT:   %LeftShift.type: type = fn_type @LeftShift [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int.31.1: Core.IntLiteral = int_value 31 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int.32.1) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int.31.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int.32.1) [template]
// CHECK:STDOUT:   %int.31.2: %i32 = int_value 31 [template]
// CHECK:STDOUT:   %int.-2147483648: %i32 = int_value -2147483648 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int.32.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int.32.1) [template]
// CHECK:STDOUT:   %int.32.2: %i32 = int_value 32 [template]
// CHECK:STDOUT:   %int.33.1: Core.IntLiteral = int_value 33 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int.33.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int.32.1) [template]
// CHECK:STDOUT:   %int.33.2: %i32 = int_value 33 [template]
// CHECK:STDOUT:   %int.1000.1: Core.IntLiteral = int_value 1000 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int.1000.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.2(%int.32.1) [template]
// CHECK:STDOUT:   %int.1000.2: %i32 = int_value 1000 [template]
// CHECK:STDOUT:   %int.0.1: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %int.0.2: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int.0.2, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.6: <specific function> = specific_function %Convert.bound.6, @Convert.2(%int.32.1) [template]
// CHECK:STDOUT:   %int.-1: %i32 = int_value -1 [template]
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
// CHECK:STDOUT:     %int.32.loc4_17: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_17: init type = call constants.%Int(%int.32.loc4_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.1: type = value_of_initializer %int.make_type_signed.loc4_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.2: type = converted %int.make_type_signed.loc4_17, %.loc4_17.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc4_25: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_25: init type = call constants.%Int(%int.32.loc4_25) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_25.1: type = value_of_initializer %int.make_type_signed.loc4_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_25.2: type = converted %int.make_type_signed.loc4_25, %.loc4_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc4_33: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_33: init type = call constants.%Int(%int.32.loc4_33) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_33.1: type = value_of_initializer %int.make_type_signed.loc4_33 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_33.2: type = converted %int.make_type_signed.loc4_33, %.loc4_33.1 [template = constants.%i32]
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
// CHECK:STDOUT:     %int.32.loc5_14: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:     %int.make_type_signed.loc5_14: init type = call constants.%Int(%int.32.loc5_14) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_14.1: type = value_of_initializer %int.make_type_signed.loc5_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_14.2: type = converted %int.make_type_signed.loc5_14, %.loc5_14.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc5_22: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:     %int.make_type_signed.loc5_22: init type = call constants.%Int(%int.32.loc5_22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_22.1: type = value_of_initializer %int.make_type_signed.loc5_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_22.2: type = converted %int.make_type_signed.loc5_22, %.loc5_22.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%int.32.loc8) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_13.1: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_13.2: type = converted %int.make_type_signed.loc8, %.loc8_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %int.32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%int.32.loc13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_13.1: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_13.2: type = converted %int.make_type_signed.loc13, %.loc13_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %int.32.loc18: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:   %int.make_type_signed.loc18: init type = call constants.%Int(%int.32.loc18) [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_13.1: type = value_of_initializer %int.make_type_signed.loc18 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_13.2: type = converted %int.make_type_signed.loc18, %.loc18_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %int.32.loc21: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:   %int.make_type_signed.loc21: init type = call constants.%Int(%int.32.loc21) [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_17.1: type = value_of_initializer %int.make_type_signed.loc21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc21_17.2: type = converted %int.make_type_signed.loc21, %.loc21_17.1 [template = constants.%i32]
// CHECK:STDOUT:   %int.32.loc26: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:   %int.make_type_signed.loc26: init type = call constants.%Int(%int.32.loc26) [template = constants.%i32]
// CHECK:STDOUT:   %.loc26_17.1: type = value_of_initializer %int.make_type_signed.loc26 [template = constants.%i32]
// CHECK:STDOUT:   %.loc26_17.2: type = converted %int.make_type_signed.loc26, %.loc26_17.1 [template = constants.%i32]
// CHECK:STDOUT:   %int.32.loc29: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:   %int.make_type_signed.loc29: init type = call constants.%Int(%int.32.loc29) [template = constants.%i32]
// CHECK:STDOUT:   %.loc29_20.1: type = value_of_initializer %int.make_type_signed.loc29 [template = constants.%i32]
// CHECK:STDOUT:   %.loc29_20.2: type = converted %int.make_type_signed.loc29, %.loc29_20.1 [template = constants.%i32]
// CHECK:STDOUT:   %int.32.loc34: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:   %int.make_type_signed.loc34: init type = call constants.%Int(%int.32.loc34) [template = constants.%i32]
// CHECK:STDOUT:   %.loc34_20.1: type = value_of_initializer %int.make_type_signed.loc34 [template = constants.%i32]
// CHECK:STDOUT:   %.loc34_20.2: type = converted %int.make_type_signed.loc34, %.loc34_20.1 [template = constants.%i32]
// CHECK:STDOUT:   %int.32.loc40: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:   %int.make_type_signed.loc40: init type = call constants.%Int(%int.32.loc40) [template = constants.%i32]
// CHECK:STDOUT:   %.loc40_15.1: type = value_of_initializer %int.make_type_signed.loc40 [template = constants.%i32]
// CHECK:STDOUT:   %.loc40_15.2: type = converted %int.make_type_signed.loc40, %.loc40_15.1 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref.loc8: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int.1.loc8: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.31.loc8: Core.IntLiteral = int_value 31 [template = constants.%int.31.1]
// CHECK:STDOUT:   %.loc8_29.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc8_29: <bound method> = bound_method %int.1.loc8, %.loc8_29.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_29: <specific function> = specific_function %Convert.bound.loc8_29, @Convert.2(constants.%int.32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc8_29: init %i32 = call %Convert.specific_fn.loc8_29(%int.1.loc8) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc8_29.2: %i32 = value_of_initializer %int.convert_checked.loc8_29 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc8_29.3: %i32 = converted %int.1.loc8, %.loc8_29.2 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc8_32.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc8_32: <bound method> = bound_method %int.31.loc8, %.loc8_32.1 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_32: <specific function> = specific_function %Convert.bound.loc8_32, @Convert.2(constants.%int.32.1) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc8_32: init %i32 = call %Convert.specific_fn.loc8_32(%int.31.loc8) [template = constants.%int.31.2]
// CHECK:STDOUT:   %.loc8_32.2: %i32 = value_of_initializer %int.convert_checked.loc8_32 [template = constants.%int.31.2]
// CHECK:STDOUT:   %.loc8_32.3: %i32 = converted %int.31.loc8, %.loc8_32.2 [template = constants.%int.31.2]
// CHECK:STDOUT:   %int.left_shift.loc8: init %i32 = call %LeftShift.ref.loc8(%.loc8_29.3, %.loc8_32.3) [template = constants.%int.-2147483648]
// CHECK:STDOUT:   %.loc8_35.1: %i32 = value_of_initializer %int.left_shift.loc8 [template = constants.%int.-2147483648]
// CHECK:STDOUT:   %.loc8_35.2: %i32 = converted %int.left_shift.loc8, %.loc8_35.1 [template = constants.%int.-2147483648]
// CHECK:STDOUT:   %size_1: %i32 = bind_name size_1, %.loc8_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc13: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int.1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:   %.loc13_29.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc13_29: <bound method> = bound_method %int.1.loc13, %.loc13_29.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_29: <specific function> = specific_function %Convert.bound.loc13_29, @Convert.2(constants.%int.32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_29: init %i32 = call %Convert.specific_fn.loc13_29(%int.1.loc13) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc13_29.2: %i32 = value_of_initializer %int.convert_checked.loc13_29 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc13_29.3: %i32 = converted %int.1.loc13, %.loc13_29.2 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc13_32.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc13_32: <bound method> = bound_method %int.32.loc13, %.loc13_32.1 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_32: <specific function> = specific_function %Convert.bound.loc13_32, @Convert.2(constants.%int.32.1) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc13_32: init %i32 = call %Convert.specific_fn.loc13_32(%int.32.loc13) [template = constants.%int.32.2]
// CHECK:STDOUT:   %.loc13_32.2: %i32 = value_of_initializer %int.convert_checked.loc13_32 [template = constants.%int.32.2]
// CHECK:STDOUT:   %.loc13_32.3: %i32 = converted %int.32.loc13, %.loc13_32.2 [template = constants.%int.32.2]
// CHECK:STDOUT:   %int.left_shift.loc13: init %i32 = call %LeftShift.ref.loc13(%.loc13_29.3, %.loc13_32.3) [template = <error>]
// CHECK:STDOUT:   %.loc13_35.1: %i32 = value_of_initializer %int.left_shift.loc13 [template = <error>]
// CHECK:STDOUT:   %.loc13_35.2: %i32 = converted %int.left_shift.loc13, %.loc13_35.1 [template = <error>]
// CHECK:STDOUT:   %size_2: %i32 = bind_name size_2, %.loc13_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc18: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int.1.loc18: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.33: Core.IntLiteral = int_value 33 [template = constants.%int.33.1]
// CHECK:STDOUT:   %.loc18_29.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc18_29: <bound method> = bound_method %int.1.loc18, %.loc18_29.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_29: <specific function> = specific_function %Convert.bound.loc18_29, @Convert.2(constants.%int.32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc18_29: init %i32 = call %Convert.specific_fn.loc18_29(%int.1.loc18) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc18_29.2: %i32 = value_of_initializer %int.convert_checked.loc18_29 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc18_29.3: %i32 = converted %int.1.loc18, %.loc18_29.2 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc18_32.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc18_32: <bound method> = bound_method %int.33, %.loc18_32.1 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_32: <specific function> = specific_function %Convert.bound.loc18_32, @Convert.2(constants.%int.32.1) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc18_32: init %i32 = call %Convert.specific_fn.loc18_32(%int.33) [template = constants.%int.33.2]
// CHECK:STDOUT:   %.loc18_32.2: %i32 = value_of_initializer %int.convert_checked.loc18_32 [template = constants.%int.33.2]
// CHECK:STDOUT:   %.loc18_32.3: %i32 = converted %int.33, %.loc18_32.2 [template = constants.%int.33.2]
// CHECK:STDOUT:   %int.left_shift.loc18: init %i32 = call %LeftShift.ref.loc18(%.loc18_29.3, %.loc18_32.3) [template = <error>]
// CHECK:STDOUT:   %.loc18_35.1: %i32 = value_of_initializer %int.left_shift.loc18 [template = <error>]
// CHECK:STDOUT:   %.loc18_35.2: %i32 = converted %int.left_shift.loc18, %.loc18_35.1 [template = <error>]
// CHECK:STDOUT:   %size_3: %i32 = bind_name size_3, %.loc18_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc21: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int.1000.loc21: Core.IntLiteral = int_value 1000 [template = constants.%int.1000.1]
// CHECK:STDOUT:   %int.31.loc21: Core.IntLiteral = int_value 31 [template = constants.%int.31.1]
// CHECK:STDOUT:   %.loc21_33.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21_33: <bound method> = bound_method %int.1000.loc21, %.loc21_33.1 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_33: <specific function> = specific_function %Convert.bound.loc21_33, @Convert.2(constants.%int.32.1) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc21_33: init %i32 = call %Convert.specific_fn.loc21_33(%int.1000.loc21) [template = constants.%int.1000.2]
// CHECK:STDOUT:   %.loc21_33.2: %i32 = value_of_initializer %int.convert_checked.loc21_33 [template = constants.%int.1000.2]
// CHECK:STDOUT:   %.loc21_33.3: %i32 = converted %int.1000.loc21, %.loc21_33.2 [template = constants.%int.1000.2]
// CHECK:STDOUT:   %.loc21_39.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21_39: <bound method> = bound_method %int.31.loc21, %.loc21_39.1 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_39: <specific function> = specific_function %Convert.bound.loc21_39, @Convert.2(constants.%int.32.1) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc21_39: init %i32 = call %Convert.specific_fn.loc21_39(%int.31.loc21) [template = constants.%int.31.2]
// CHECK:STDOUT:   %.loc21_39.2: %i32 = value_of_initializer %int.convert_checked.loc21_39 [template = constants.%int.31.2]
// CHECK:STDOUT:   %.loc21_39.3: %i32 = converted %int.31.loc21, %.loc21_39.2 [template = constants.%int.31.2]
// CHECK:STDOUT:   %int.left_shift.loc21: init %i32 = call %LeftShift.ref.loc21(%.loc21_33.3, %.loc21_39.3) [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc21_42.1: %i32 = value_of_initializer %int.left_shift.loc21 [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc21_42.2: %i32 = converted %int.left_shift.loc21, %.loc21_42.1 [template = constants.%int.0.1]
// CHECK:STDOUT:   %overflow_1: %i32 = bind_name overflow_1, %.loc21_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc26: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int.1000.loc26: Core.IntLiteral = int_value 1000 [template = constants.%int.1000.1]
// CHECK:STDOUT:   %int.32.loc26: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:   %.loc26_33.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc26_33: <bound method> = bound_method %int.1000.loc26, %.loc26_33.1 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc26_33: <specific function> = specific_function %Convert.bound.loc26_33, @Convert.2(constants.%int.32.1) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc26_33: init %i32 = call %Convert.specific_fn.loc26_33(%int.1000.loc26) [template = constants.%int.1000.2]
// CHECK:STDOUT:   %.loc26_33.2: %i32 = value_of_initializer %int.convert_checked.loc26_33 [template = constants.%int.1000.2]
// CHECK:STDOUT:   %.loc26_33.3: %i32 = converted %int.1000.loc26, %.loc26_33.2 [template = constants.%int.1000.2]
// CHECK:STDOUT:   %.loc26_39.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc26_39: <bound method> = bound_method %int.32.loc26, %.loc26_39.1 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc26_39: <specific function> = specific_function %Convert.bound.loc26_39, @Convert.2(constants.%int.32.1) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc26_39: init %i32 = call %Convert.specific_fn.loc26_39(%int.32.loc26) [template = constants.%int.32.2]
// CHECK:STDOUT:   %.loc26_39.2: %i32 = value_of_initializer %int.convert_checked.loc26_39 [template = constants.%int.32.2]
// CHECK:STDOUT:   %.loc26_39.3: %i32 = converted %int.32.loc26, %.loc26_39.2 [template = constants.%int.32.2]
// CHECK:STDOUT:   %int.left_shift.loc26: init %i32 = call %LeftShift.ref.loc26(%.loc26_33.3, %.loc26_39.3) [template = <error>]
// CHECK:STDOUT:   %.loc26_42.1: %i32 = value_of_initializer %int.left_shift.loc26 [template = <error>]
// CHECK:STDOUT:   %.loc26_42.2: %i32 = converted %int.left_shift.loc26, %.loc26_42.1 [template = <error>]
// CHECK:STDOUT:   %overflow_2: %i32 = bind_name overflow_2, %.loc26_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc29: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int.0.loc29: Core.IntLiteral = int_value 0 [template = constants.%int.0.2]
// CHECK:STDOUT:   %int.31.loc29: Core.IntLiteral = int_value 31 [template = constants.%int.31.1]
// CHECK:STDOUT:   %.loc29_36.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc29_36: <bound method> = bound_method %int.0.loc29, %.loc29_36.1 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc29_36: <specific function> = specific_function %Convert.bound.loc29_36, @Convert.2(constants.%int.32.1) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc29_36: init %i32 = call %Convert.specific_fn.loc29_36(%int.0.loc29) [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc29_36.2: %i32 = value_of_initializer %int.convert_checked.loc29_36 [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc29_36.3: %i32 = converted %int.0.loc29, %.loc29_36.2 [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc29_39.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc29_39: <bound method> = bound_method %int.31.loc29, %.loc29_39.1 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc29_39: <specific function> = specific_function %Convert.bound.loc29_39, @Convert.2(constants.%int.32.1) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc29_39: init %i32 = call %Convert.specific_fn.loc29_39(%int.31.loc29) [template = constants.%int.31.2]
// CHECK:STDOUT:   %.loc29_39.2: %i32 = value_of_initializer %int.convert_checked.loc29_39 [template = constants.%int.31.2]
// CHECK:STDOUT:   %.loc29_39.3: %i32 = converted %int.31.loc29, %.loc29_39.2 [template = constants.%int.31.2]
// CHECK:STDOUT:   %int.left_shift.loc29: init %i32 = call %LeftShift.ref.loc29(%.loc29_36.3, %.loc29_39.3) [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc29_42.1: %i32 = value_of_initializer %int.left_shift.loc29 [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc29_42.2: %i32 = converted %int.left_shift.loc29, %.loc29_42.1 [template = constants.%int.0.1]
// CHECK:STDOUT:   %no_overflow_1: %i32 = bind_name no_overflow_1, %.loc29_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc34: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int.0.loc34: Core.IntLiteral = int_value 0 [template = constants.%int.0.2]
// CHECK:STDOUT:   %int.32.loc34: Core.IntLiteral = int_value 32 [template = constants.%int.32.1]
// CHECK:STDOUT:   %.loc34_36.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc34_36: <bound method> = bound_method %int.0.loc34, %.loc34_36.1 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc34_36: <specific function> = specific_function %Convert.bound.loc34_36, @Convert.2(constants.%int.32.1) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc34_36: init %i32 = call %Convert.specific_fn.loc34_36(%int.0.loc34) [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc34_36.2: %i32 = value_of_initializer %int.convert_checked.loc34_36 [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc34_36.3: %i32 = converted %int.0.loc34, %.loc34_36.2 [template = constants.%int.0.1]
// CHECK:STDOUT:   %.loc34_39.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc34_39: <bound method> = bound_method %int.32.loc34, %.loc34_39.1 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc34_39: <specific function> = specific_function %Convert.bound.loc34_39, @Convert.2(constants.%int.32.1) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc34_39: init %i32 = call %Convert.specific_fn.loc34_39(%int.32.loc34) [template = constants.%int.32.2]
// CHECK:STDOUT:   %.loc34_39.2: %i32 = value_of_initializer %int.convert_checked.loc34_39 [template = constants.%int.32.2]
// CHECK:STDOUT:   %.loc34_39.3: %i32 = converted %int.32.loc34, %.loc34_39.2 [template = constants.%int.32.2]
// CHECK:STDOUT:   %int.left_shift.loc34: init %i32 = call %LeftShift.ref.loc34(%.loc34_36.3, %.loc34_39.3) [template = <error>]
// CHECK:STDOUT:   %.loc34_42.1: %i32 = value_of_initializer %int.left_shift.loc34 [template = <error>]
// CHECK:STDOUT:   %.loc34_42.2: %i32 = converted %int.left_shift.loc34, %.loc34_42.1 [template = <error>]
// CHECK:STDOUT:   %no_overflow_2: %i32 = bind_name no_overflow_2, %.loc34_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc40: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int.1.loc40_31: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int.1.loc40_41: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc40_41.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc40_41: <bound method> = bound_method %int.1.loc40_41, %.loc40_41.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc40_41: <specific function> = specific_function %Convert.bound.loc40_41, @Convert.2(constants.%int.32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc40_41: init %i32 = call %Convert.specific_fn.loc40_41(%int.1.loc40_41) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc40_41.2: %i32 = value_of_initializer %int.convert_checked.loc40_41 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc40_41.3: %i32 = converted %int.1.loc40_41, %.loc40_41.2 [template = constants.%int.1.2]
// CHECK:STDOUT:   %int.snegate: init %i32 = call %Negate.ref(%.loc40_41.3) [template = constants.%int.-1]
// CHECK:STDOUT:   %.loc40_31.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc40_31: <bound method> = bound_method %int.1.loc40_31, %.loc40_31.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc40_31: <specific function> = specific_function %Convert.bound.loc40_31, @Convert.2(constants.%int.32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc40_31: init %i32 = call %Convert.specific_fn.loc40_31(%int.1.loc40_31) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc40_31.2: %i32 = value_of_initializer %int.convert_checked.loc40_31 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc40_31.3: %i32 = converted %int.1.loc40_31, %.loc40_31.2 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc40_42.1: %i32 = value_of_initializer %int.snegate [template = constants.%int.-1]
// CHECK:STDOUT:   %.loc40_42.2: %i32 = converted %int.snegate, %.loc40_42.1 [template = constants.%int.-1]
// CHECK:STDOUT:   %int.left_shift.loc40: init %i32 = call %LeftShift.ref.loc40(%.loc40_31.3, %.loc40_42.2) [template = <error>]
// CHECK:STDOUT:   %.loc40_44.1: %i32 = value_of_initializer %int.left_shift.loc40 [template = <error>]
// CHECK:STDOUT:   %.loc40_44.2: %i32 = converted %int.left_shift.loc40, %.loc40_44.1 [template = <error>]
// CHECK:STDOUT:   %negative: %i32 = bind_name negative, %.loc40_44.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
