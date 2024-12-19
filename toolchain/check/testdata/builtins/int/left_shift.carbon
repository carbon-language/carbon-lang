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

library "[[@TEST_NAME]]";

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
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance >= type width of 32 in `1 << 32` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_2: i32 = LeftShift(1, 32);
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let size_2: i32 = LeftShift(1, 32);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:19: error: shift distance >= type width of 32 in `1 << 33` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let size_3: i32 = LeftShift(1, 33);
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let size_3: i32 = LeftShift(1, 33);

// Overflow is allowed if the shift distance is in bounds.
let overflow_1: i32 = LeftShift(1000, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:23: error: shift distance >= type width of 32 in `1000 << 32` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let overflow_2: i32 = LeftShift(1000, 32);
// CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let overflow_2: i32 = LeftShift(1000, 32);

// Oversize shifts aren't allowed even if there's no overflow.
let no_overflow_1: i32 = LeftShift(0, 31);
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:26: error: shift distance >= type width of 32 in `0 << 32` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let no_overflow_2: i32 = LeftShift(0, 32);
// CHECK:STDERR:                          ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let no_overflow_2: i32 = LeftShift(0, 32);

// Negative shifts aren't allowed either.
// CHECK:STDERR: fail_bad_shift.carbon:[[@LINE+4]]:21: error: shift distance >= type width of 32 in `1 << -1` [CompileTimeShiftOutOfRange]
// CHECK:STDERR: let negative: i32 = LeftShift(1, Negate(1));
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let negative: i32 = LeftShift(1, Negate(1));

// --- literal.carbon

library "[[@TEST_NAME]]";

fn LeftShift(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.left_shift";

class Expect(N:! Core.IntLiteral()) {}
// CHECK:STDERR: literal.carbon:[[@LINE+4]]:13: error: expression cannot be used as a value [UseOfNonExprAsValue]
// CHECK:STDERR: fn Test(N:! Core.IntLiteral) -> Expect(N) { return {}; }
// CHECK:STDERR:             ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Test(N:! Core.IntLiteral) -> Expect(N) { return {}; }

fn F() {
  // Zero can be shifted by any amount.
  Test(LeftShift(0, 0)) as Expect(0);
  Test(LeftShift(0, 1000000000000)) as Expect(0);

  // Positive numbers can be shifted.
  Test(LeftShift(1, 0)) as Expect(1);
  Test(LeftShift(1, 1)) as Expect(2);
  Test(LeftShift(2, 1)) as Expect(4);
  Test(LeftShift(1, 2)) as Expect(4);
  Test(LeftShift(3, 1)) as Expect(12);

  // Negative numbers can be shifted too.
  Test(LeftShift(-1, 0)) as Expect(-1);
  Test(LeftShift(-1, 1)) as Expect(-2);
  Test(LeftShift(-2, 1)) as Expect(-4);
  Test(LeftShift(-3, 1)) as Expect(-5);

  // Large numbers can be shifted.
  Test(LeftShift(0x7FFF_FFFF_FFFF_FFFF, 1)) as Expect(0xFFFF_FFFF_FFFF_FFFE);
  Test(LeftShift(0xFFFF_FFFF_FFFF_FFFF, 1)) as Expect(0x1_FFFF_FFFF_FFFF_FFFE);
}
// --- literal_negative.carbon

library "[[@TEST_NAME]]";

fn LeftShift(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.left_shift";

// CHECK:STDERR: literal_negative.carbon:[[@LINE+4]]:16: error: shift distance negative in `1 << -1` [CompileTimeShiftNegative]
// CHECK:STDERR: let bad: i32 = LeftShift(1, -1);
// CHECK:STDERR:                ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
let bad: i32 = LeftShift(1, -1);

// --- literal_overflow.carbon

library "[[@TEST_NAME]]";

fn LeftShift(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.left_shift";

// CHECK:STDERR: literal_overflow.carbon:[[@LINE+4]]:16: error: shift distance of 4000000000 would result in an integer whose width is greater than the maximum supported width of 8388608 [CompileTimeUnsizedShiftOutOfRange]
// CHECK:STDERR: let bad: i32 = LeftShift(1, 4000_000_000);
// CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let bad: i32 = LeftShift(1, 4000_000_000);

// CHECK:STDERR: literal_overflow.carbon:[[@LINE+3]]:25: error: shift distance of 4000000000 would result in an integer whose width is greater than the maximum supported width of 8388608 [CompileTimeUnsizedShiftOutOfRange]
// CHECK:STDERR: let bad_negative: i32 = LeftShift(-1, 4000_000_000);
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~
let bad_negative: i32 = LeftShift(-1, 4000_000_000);

// CHECK:STDOUT: --- int_left_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %LeftShift.type.1: type = fn_type @LeftShift.1 [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_5.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int_20.1: %i32 = int_value 20 [template]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.3, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [template]
// CHECK:STDOUT:   %interface.6: <witness> = interface_witness (%Convert.11) [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_20.1, %Convert.11 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.3(%int_32) [template]
// CHECK:STDOUT:   %int_20.2: Core.IntLiteral = int_value 20 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_20.2, %i32 [template]
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
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LeftShift.decl: %LeftShift.type.1 = fn_decl @LeftShift.1 [template = constants.%LeftShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_17: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_25: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_33: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %LeftShift.ref: %LeftShift.type.1 = name_ref LeftShift, %LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc6_26: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc6_26: <bound method> = bound_method %int_5, %impl.elem0.loc6_26 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_26: <specific function> = specific_function %Convert.bound.loc6_26, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc6_26: init %i32 = call %Convert.specific_fn.loc6_26(%int_5) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc6_26.1: %i32 = value_of_initializer %int.convert_checked.loc6_26 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc6_26.2: %i32 = converted %int_5, %.loc6_26.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %impl.elem0.loc6_29: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc6_29: <bound method> = bound_method %int_2, %impl.elem0.loc6_29 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_29: <specific function> = specific_function %Convert.bound.loc6_29, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc6_29: init %i32 = call %Convert.specific_fn.loc6_29(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc6_29.1: %i32 = value_of_initializer %int.convert_checked.loc6_29 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc6_29.2: %i32 = converted %int_2, %.loc6_29.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.left_shift: init %i32 = call %LeftShift.ref(%.loc6_26.2, %.loc6_29.2) [template = constants.%int_20.1]
// CHECK:STDOUT:   %impl.elem0.loc6_30: %Convert.type.5 = interface_witness_access constants.%interface.6, element0 [template = constants.%Convert.11]
// CHECK:STDOUT:   %Convert.bound.loc6_30: <bound method> = bound_method %int.left_shift, %impl.elem0.loc6_30 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_30: <specific function> = specific_function %Convert.bound.loc6_30, @Convert.3(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %.loc6_30.1: %i32 = value_of_initializer %int.left_shift [template = constants.%int_20.1]
// CHECK:STDOUT:   %.loc6_30.2: %i32 = converted %int.left_shift, %.loc6_30.1 [template = constants.%int_20.1]
// CHECK:STDOUT:   %int.convert_checked.loc6_30: init Core.IntLiteral = call %Convert.specific_fn.loc6_30(%.loc6_30.2) [template = constants.%int_20.2]
// CHECK:STDOUT:   %.loc6_30.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc6_30 [template = constants.%int_20.2]
// CHECK:STDOUT:   %.loc6_30.4: Core.IntLiteral = converted %int.left_shift, %.loc6_30.3 [template = constants.%int_20.2]
// CHECK:STDOUT:   %array_type.loc6: type = array_type %.loc6_30.4, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %arr.var: ref %array_type = var arr
// CHECK:STDOUT:   %arr: ref %array_type = bind_name arr, %arr.var
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [template = constants.%int_20.2]
// CHECK:STDOUT:   %array_type.loc7: type = array_type %int_20, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template = constants.%ptr]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc9_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc9_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc9_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9_35: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.left_shift: init %i32 = call %LeftShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc10_25.1: %i32 = value_of_initializer %int.left_shift
// CHECK:STDOUT:   %.loc10_25.2: %i32 = converted %int.left_shift, %.loc10_25.1
// CHECK:STDOUT:   return %.loc10_25.2
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
// CHECK:STDOUT: --- fail_bad_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32.1) [template]
// CHECK:STDOUT:   %LeftShift.type.1: type = fn_type @LeftShift.1 [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Negate.type.1: type = fn_type @Negate.1 [template]
// CHECK:STDOUT:   %Negate: %Negate.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_31.1: Core.IntLiteral = int_value 31 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32.1) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_31.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_31.2: %i32 = int_value 31 [template]
// CHECK:STDOUT:   %int_-2147483648: %i32 = int_value -2147483648 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_32.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_32.2: %i32 = int_value 32 [template]
// CHECK:STDOUT:   %int_33.1: Core.IntLiteral = int_value 33 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_33.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_33.2: %i32 = int_value 33 [template]
// CHECK:STDOUT:   %int_1000.1: Core.IntLiteral = int_value 1000 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_1000.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_1000.2: %i32 = int_value 1000 [template]
// CHECK:STDOUT:   %int_0.1: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %int_0.2: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_0.2, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.6: <specific function> = specific_function %Convert.bound.6, @Convert.2(%int_32.1) [template]
// CHECK:STDOUT:   %int_-1: %i32 = int_value -1 [template]
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
// CHECK:STDOUT:   %LeftShift.decl: %LeftShift.type.1 = fn_decl @LeftShift.1 [template = constants.%LeftShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_17: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:     %i32.loc4_17: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_25: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:     %i32.loc4_25: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_33: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:     %i32.loc4_33: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
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
// CHECK:STDOUT:     %int_32.loc5_14: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:     %i32.loc5_14: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_22: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:     %i32.loc5_22: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %i32.loc8: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc18: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %i32.loc18: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc21: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %i32.loc21: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc26: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %i32.loc26: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc29: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %i32.loc29: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc34: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %i32.loc34: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc41: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %i32.loc41: type = class_type @Int, @Int(constants.%int_32.1) [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate.1(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref.loc8: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_31.loc8: Core.IntLiteral = int_value 31 [template = constants.%int_31.1]
// CHECK:STDOUT:   %impl.elem0.loc8_29: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc8_29: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8_29 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_29: <specific function> = specific_function %Convert.bound.loc8_29, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc8_29: init %i32 = call %Convert.specific_fn.loc8_29(%int_1.loc8) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc8_29.1: %i32 = value_of_initializer %int.convert_checked.loc8_29 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc8_29.2: %i32 = converted %int_1.loc8, %.loc8_29.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc8_32: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc8_32: <bound method> = bound_method %int_31.loc8, %impl.elem0.loc8_32 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_32: <specific function> = specific_function %Convert.bound.loc8_32, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc8_32: init %i32 = call %Convert.specific_fn.loc8_32(%int_31.loc8) [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc8_32.1: %i32 = value_of_initializer %int.convert_checked.loc8_32 [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc8_32.2: %i32 = converted %int_31.loc8, %.loc8_32.1 [template = constants.%int_31.2]
// CHECK:STDOUT:   %int.left_shift.loc8: init %i32 = call %LeftShift.ref.loc8(%.loc8_29.2, %.loc8_32.2) [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc8_35.1: %i32 = value_of_initializer %int.left_shift.loc8 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc8_35.2: %i32 = converted %int.left_shift.loc8, %.loc8_35.1 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %size_1: %i32 = bind_name size_1, %.loc8_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc13: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %impl.elem0.loc13_29: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_29: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_29 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_29: <specific function> = specific_function %Convert.bound.loc13_29, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_29: init %i32 = call %Convert.specific_fn.loc13_29(%int_1.loc13) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_29.1: %i32 = value_of_initializer %int.convert_checked.loc13_29 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_29.2: %i32 = converted %int_1.loc13, %.loc13_29.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc13_32: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_32: <bound method> = bound_method %int_32.loc13, %impl.elem0.loc13_32 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_32: <specific function> = specific_function %Convert.bound.loc13_32, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc13_32: init %i32 = call %Convert.specific_fn.loc13_32(%int_32.loc13) [template = constants.%int_32.2]
// CHECK:STDOUT:   %.loc13_32.1: %i32 = value_of_initializer %int.convert_checked.loc13_32 [template = constants.%int_32.2]
// CHECK:STDOUT:   %.loc13_32.2: %i32 = converted %int_32.loc13, %.loc13_32.1 [template = constants.%int_32.2]
// CHECK:STDOUT:   %int.left_shift.loc13: init %i32 = call %LeftShift.ref.loc13(%.loc13_29.2, %.loc13_32.2) [template = <error>]
// CHECK:STDOUT:   %.loc13_35.1: %i32 = value_of_initializer %int.left_shift.loc13 [template = <error>]
// CHECK:STDOUT:   %.loc13_35.2: %i32 = converted %int.left_shift.loc13, %.loc13_35.1 [template = <error>]
// CHECK:STDOUT:   %size_2: %i32 = bind_name size_2, %.loc13_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc18: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [template = constants.%int_33.1]
// CHECK:STDOUT:   %impl.elem0.loc18_29: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc18_29: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_29 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_29: <specific function> = specific_function %Convert.bound.loc18_29, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc18_29: init %i32 = call %Convert.specific_fn.loc18_29(%int_1.loc18) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc18_29.1: %i32 = value_of_initializer %int.convert_checked.loc18_29 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc18_29.2: %i32 = converted %int_1.loc18, %.loc18_29.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc18_32: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc18_32: <bound method> = bound_method %int_33, %impl.elem0.loc18_32 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_32: <specific function> = specific_function %Convert.bound.loc18_32, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc18_32: init %i32 = call %Convert.specific_fn.loc18_32(%int_33) [template = constants.%int_33.2]
// CHECK:STDOUT:   %.loc18_32.1: %i32 = value_of_initializer %int.convert_checked.loc18_32 [template = constants.%int_33.2]
// CHECK:STDOUT:   %.loc18_32.2: %i32 = converted %int_33, %.loc18_32.1 [template = constants.%int_33.2]
// CHECK:STDOUT:   %int.left_shift.loc18: init %i32 = call %LeftShift.ref.loc18(%.loc18_29.2, %.loc18_32.2) [template = <error>]
// CHECK:STDOUT:   %.loc18_35.1: %i32 = value_of_initializer %int.left_shift.loc18 [template = <error>]
// CHECK:STDOUT:   %.loc18_35.2: %i32 = converted %int.left_shift.loc18, %.loc18_35.1 [template = <error>]
// CHECK:STDOUT:   %size_3: %i32 = bind_name size_3, %.loc18_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc21: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1000.loc21: Core.IntLiteral = int_value 1000 [template = constants.%int_1000.1]
// CHECK:STDOUT:   %int_31.loc21: Core.IntLiteral = int_value 31 [template = constants.%int_31.1]
// CHECK:STDOUT:   %impl.elem0.loc21_33: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc21_33: <bound method> = bound_method %int_1000.loc21, %impl.elem0.loc21_33 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_33: <specific function> = specific_function %Convert.bound.loc21_33, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc21_33: init %i32 = call %Convert.specific_fn.loc21_33(%int_1000.loc21) [template = constants.%int_1000.2]
// CHECK:STDOUT:   %.loc21_33.1: %i32 = value_of_initializer %int.convert_checked.loc21_33 [template = constants.%int_1000.2]
// CHECK:STDOUT:   %.loc21_33.2: %i32 = converted %int_1000.loc21, %.loc21_33.1 [template = constants.%int_1000.2]
// CHECK:STDOUT:   %impl.elem0.loc21_39: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc21_39: <bound method> = bound_method %int_31.loc21, %impl.elem0.loc21_39 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_39: <specific function> = specific_function %Convert.bound.loc21_39, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc21_39: init %i32 = call %Convert.specific_fn.loc21_39(%int_31.loc21) [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc21_39.1: %i32 = value_of_initializer %int.convert_checked.loc21_39 [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc21_39.2: %i32 = converted %int_31.loc21, %.loc21_39.1 [template = constants.%int_31.2]
// CHECK:STDOUT:   %int.left_shift.loc21: init %i32 = call %LeftShift.ref.loc21(%.loc21_33.2, %.loc21_39.2) [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc21_42.1: %i32 = value_of_initializer %int.left_shift.loc21 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc21_42.2: %i32 = converted %int.left_shift.loc21, %.loc21_42.1 [template = constants.%int_0.1]
// CHECK:STDOUT:   %overflow_1: %i32 = bind_name overflow_1, %.loc21_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc26: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1000.loc26: Core.IntLiteral = int_value 1000 [template = constants.%int_1000.1]
// CHECK:STDOUT:   %int_32.loc26: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %impl.elem0.loc26_33: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc26_33: <bound method> = bound_method %int_1000.loc26, %impl.elem0.loc26_33 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc26_33: <specific function> = specific_function %Convert.bound.loc26_33, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc26_33: init %i32 = call %Convert.specific_fn.loc26_33(%int_1000.loc26) [template = constants.%int_1000.2]
// CHECK:STDOUT:   %.loc26_33.1: %i32 = value_of_initializer %int.convert_checked.loc26_33 [template = constants.%int_1000.2]
// CHECK:STDOUT:   %.loc26_33.2: %i32 = converted %int_1000.loc26, %.loc26_33.1 [template = constants.%int_1000.2]
// CHECK:STDOUT:   %impl.elem0.loc26_39: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc26_39: <bound method> = bound_method %int_32.loc26, %impl.elem0.loc26_39 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc26_39: <specific function> = specific_function %Convert.bound.loc26_39, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc26_39: init %i32 = call %Convert.specific_fn.loc26_39(%int_32.loc26) [template = constants.%int_32.2]
// CHECK:STDOUT:   %.loc26_39.1: %i32 = value_of_initializer %int.convert_checked.loc26_39 [template = constants.%int_32.2]
// CHECK:STDOUT:   %.loc26_39.2: %i32 = converted %int_32.loc26, %.loc26_39.1 [template = constants.%int_32.2]
// CHECK:STDOUT:   %int.left_shift.loc26: init %i32 = call %LeftShift.ref.loc26(%.loc26_33.2, %.loc26_39.2) [template = <error>]
// CHECK:STDOUT:   %.loc26_42.1: %i32 = value_of_initializer %int.left_shift.loc26 [template = <error>]
// CHECK:STDOUT:   %.loc26_42.2: %i32 = converted %int.left_shift.loc26, %.loc26_42.1 [template = <error>]
// CHECK:STDOUT:   %overflow_2: %i32 = bind_name overflow_2, %.loc26_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc29: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc29: Core.IntLiteral = int_value 0 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int_31.loc29: Core.IntLiteral = int_value 31 [template = constants.%int_31.1]
// CHECK:STDOUT:   %impl.elem0.loc29_36: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc29_36: <bound method> = bound_method %int_0.loc29, %impl.elem0.loc29_36 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc29_36: <specific function> = specific_function %Convert.bound.loc29_36, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc29_36: init %i32 = call %Convert.specific_fn.loc29_36(%int_0.loc29) [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc29_36.1: %i32 = value_of_initializer %int.convert_checked.loc29_36 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc29_36.2: %i32 = converted %int_0.loc29, %.loc29_36.1 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc29_39: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc29_39: <bound method> = bound_method %int_31.loc29, %impl.elem0.loc29_39 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc29_39: <specific function> = specific_function %Convert.bound.loc29_39, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc29_39: init %i32 = call %Convert.specific_fn.loc29_39(%int_31.loc29) [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc29_39.1: %i32 = value_of_initializer %int.convert_checked.loc29_39 [template = constants.%int_31.2]
// CHECK:STDOUT:   %.loc29_39.2: %i32 = converted %int_31.loc29, %.loc29_39.1 [template = constants.%int_31.2]
// CHECK:STDOUT:   %int.left_shift.loc29: init %i32 = call %LeftShift.ref.loc29(%.loc29_36.2, %.loc29_39.2) [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc29_42.1: %i32 = value_of_initializer %int.left_shift.loc29 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc29_42.2: %i32 = converted %int.left_shift.loc29, %.loc29_42.1 [template = constants.%int_0.1]
// CHECK:STDOUT:   %no_overflow_1: %i32 = bind_name no_overflow_1, %.loc29_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc34: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc34: Core.IntLiteral = int_value 0 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int_32.loc34: Core.IntLiteral = int_value 32 [template = constants.%int_32.1]
// CHECK:STDOUT:   %impl.elem0.loc34_36: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc34_36: <bound method> = bound_method %int_0.loc34, %impl.elem0.loc34_36 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc34_36: <specific function> = specific_function %Convert.bound.loc34_36, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc34_36: init %i32 = call %Convert.specific_fn.loc34_36(%int_0.loc34) [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc34_36.1: %i32 = value_of_initializer %int.convert_checked.loc34_36 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc34_36.2: %i32 = converted %int_0.loc34, %.loc34_36.1 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc34_39: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc34_39: <bound method> = bound_method %int_32.loc34, %impl.elem0.loc34_39 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc34_39: <specific function> = specific_function %Convert.bound.loc34_39, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc34_39: init %i32 = call %Convert.specific_fn.loc34_39(%int_32.loc34) [template = constants.%int_32.2]
// CHECK:STDOUT:   %.loc34_39.1: %i32 = value_of_initializer %int.convert_checked.loc34_39 [template = constants.%int_32.2]
// CHECK:STDOUT:   %.loc34_39.2: %i32 = converted %int_32.loc34, %.loc34_39.1 [template = constants.%int_32.2]
// CHECK:STDOUT:   %int.left_shift.loc34: init %i32 = call %LeftShift.ref.loc34(%.loc34_36.2, %.loc34_39.2) [template = <error>]
// CHECK:STDOUT:   %.loc34_42.1: %i32 = value_of_initializer %int.left_shift.loc34 [template = <error>]
// CHECK:STDOUT:   %.loc34_42.2: %i32 = converted %int.left_shift.loc34, %.loc34_42.1 [template = <error>]
// CHECK:STDOUT:   %no_overflow_2: %i32 = bind_name no_overflow_2, %.loc34_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc41: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc41_31: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %Negate.ref: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc41_41: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc41_41: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc41_41: <bound method> = bound_method %int_1.loc41_41, %impl.elem0.loc41_41 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc41_41: <specific function> = specific_function %Convert.bound.loc41_41, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc41_41: init %i32 = call %Convert.specific_fn.loc41_41(%int_1.loc41_41) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc41_41.1: %i32 = value_of_initializer %int.convert_checked.loc41_41 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc41_41.2: %i32 = converted %int_1.loc41_41, %.loc41_41.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.snegate: init %i32 = call %Negate.ref(%.loc41_41.2) [template = constants.%int_-1]
// CHECK:STDOUT:   %impl.elem0.loc41_31: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc41_31: <bound method> = bound_method %int_1.loc41_31, %impl.elem0.loc41_31 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc41_31: <specific function> = specific_function %Convert.bound.loc41_31, @Convert.2(constants.%int_32.1) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc41_31: init %i32 = call %Convert.specific_fn.loc41_31(%int_1.loc41_31) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc41_31.1: %i32 = value_of_initializer %int.convert_checked.loc41_31 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc41_31.2: %i32 = converted %int_1.loc41_31, %.loc41_31.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc41_42.1: %i32 = value_of_initializer %int.snegate [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc41_42.2: %i32 = converted %int.snegate, %.loc41_42.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.left_shift.loc41: init %i32 = call %LeftShift.ref.loc41(%.loc41_31.2, %.loc41_42.2) [template = <error>]
// CHECK:STDOUT:   %.loc41_44.1: %i32 = value_of_initializer %int.left_shift.loc41 [template = <error>]
// CHECK:STDOUT:   %.loc41_44.2: %i32 = converted %int.left_shift.loc41, %.loc41_44.1 [template = <error>]
// CHECK:STDOUT:   %negative: %i32 = bind_name negative, %.loc41_44.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %LeftShift.type: type = fn_type @LeftShift [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Expect.type: type = generic_class_type @Expect [template]
// CHECK:STDOUT:   %Expect.generic: %Expect.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.1: type = class_type @Expect, @Expect(%N) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %N.patt.2: <error> = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%int_0) [template]
// CHECK:STDOUT:   %int_1000000000000: Core.IntLiteral = int_value 1000000000000 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%int_1) [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Expect.4: type = class_type @Expect, @Expect(%int_2) [template]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Expect.5: type = class_type @Expect, @Expect(%int_4) [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %Expect.6: type = class_type @Expect, @Expect(%int_12) [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound.1: <bound method> = bound_method %int_1, %Op.2 [template]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Expect.7: type = class_type @Expect, @Expect(%int_-1) [template]
// CHECK:STDOUT:   %int_-2: Core.IntLiteral = int_value -2 [template]
// CHECK:STDOUT:   %Op.bound.2: <bound method> = bound_method %int_2, %Op.2 [template]
// CHECK:STDOUT:   %Expect.8: type = class_type @Expect, @Expect(%int_-2) [template]
// CHECK:STDOUT:   %int_-4: Core.IntLiteral = int_value -4 [template]
// CHECK:STDOUT:   %Op.bound.3: <bound method> = bound_method %int_4, %Op.2 [template]
// CHECK:STDOUT:   %Expect.9: type = class_type @Expect, @Expect(%int_-4) [template]
// CHECK:STDOUT:   %Op.bound.4: <bound method> = bound_method %int_3, %Op.2 [template]
// CHECK:STDOUT:   %int_-3: Core.IntLiteral = int_value -3 [template]
// CHECK:STDOUT:   %int_-6: Core.IntLiteral = int_value -6 [template]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %Op.bound.5: <bound method> = bound_method %int_5, %Op.2 [template]
// CHECK:STDOUT:   %int_-5: Core.IntLiteral = int_value -5 [template]
// CHECK:STDOUT:   %Expect.10: type = class_type @Expect, @Expect(%int_-5) [template]
// CHECK:STDOUT:   %int_9223372036854775807: Core.IntLiteral = int_value 9223372036854775807 [template]
// CHECK:STDOUT:   %int_18446744073709551614: Core.IntLiteral = int_value 18446744073709551614 [template]
// CHECK:STDOUT:   %Expect.11: type = class_type @Expect, @Expect(%int_18446744073709551614) [template]
// CHECK:STDOUT:   %int_18446744073709551615: Core.IntLiteral = int_value 18446744073709551615 [template]
// CHECK:STDOUT:   %int_36893488147419103230: Core.IntLiteral = int_value 36893488147419103230 [template]
// CHECK:STDOUT:   %Expect.12: type = class_type @Expect, @Expect(%int_36893488147419103230) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Negate = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LeftShift.decl: %LeftShift.type = fn_decl @LeftShift [template = constants.%LeftShift] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_17: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_21: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_33: init type = call %IntLiteral.ref.loc4_21() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_33.1: type = value_of_initializer %int_literal.make_type.loc4_33 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_33.2: type = converted %int_literal.make_type.loc4_33, %.loc4_33.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_39: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_43: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_55: init type = call %IntLiteral.ref.loc4_43() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_55.1: type = value_of_initializer %int_literal.make_type.loc4_55 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_55.2: type = converted %int_literal.make_type.loc4_55, %.loc4_55.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_61: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_65: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_77: init type = call %IntLiteral.ref.loc4_65() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_77.1: type = value_of_initializer %int_literal.make_type.loc4_77 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_77.2: type = converted %int_literal.make_type.loc4_77, %.loc4_77.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %N.patt.loc6_14.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_14.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc6_14.1, runtime_param<invalid> [symbolic = %N.patt.loc6_14.2 (constants.%N.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_34.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_34.2: type = converted %int_literal.make_type, %.loc6_34.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc6_14.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_14.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %N.patt.loc11_9.1: <error> = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc11_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: <error> = value_param_pattern %N.patt.loc11_9.1, runtime_param<invalid> [symbolic = %N.patt.loc11_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %N.ref: <error> = name_ref N, %N [template = <error>]
// CHECK:STDOUT:     %N.param: <error> = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N: <error> = bind_symbolic_name N, 0, %N.param [template = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Expect(%N.loc6_14.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc6_14.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc6_14.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_14.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_14.2 (constants.%N.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Expect.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%N: <error>) {
// CHECK:STDOUT:   %N.patt.loc11_9.2: <error> = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc11_9.2 (constants.%N.patt.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: <error>) -> <error> {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc11: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref.loc15: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc15: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc15_18: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_0.loc15_21: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int.left_shift.loc15: init Core.IntLiteral = call %LeftShift.ref.loc15(%int_0.loc15_18, %int_0.loc15_21) [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.ref.loc15: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc15_35: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc15: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %Test.ref.loc16: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc16: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc16_18: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_1000000000000: Core.IntLiteral = int_value 1000000000000 [template = constants.%int_1000000000000]
// CHECK:STDOUT:   %int.left_shift.loc16: init Core.IntLiteral = call %LeftShift.ref.loc16(%int_0.loc16_18, %int_1000000000000) [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.ref.loc16: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc16_47: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc16: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.2]
// CHECK:STDOUT:   %Test.ref.loc19: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc19: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc19_18: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_0.loc19: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int.left_shift.loc19: init Core.IntLiteral = call %LeftShift.ref.loc19(%int_1.loc19_18, %int_0.loc19) [template = constants.%int_1]
// CHECK:STDOUT:   %Expect.ref.loc19: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1.loc19_35: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %Expect.loc19: type = class_type @Expect, @Expect(constants.%int_1) [template = constants.%Expect.3]
// CHECK:STDOUT:   %Test.ref.loc20: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc20: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc20_18: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_1.loc20_21: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.left_shift.loc20: init Core.IntLiteral = call %LeftShift.ref.loc20(%int_1.loc20_18, %int_1.loc20_21) [template = constants.%int_2]
// CHECK:STDOUT:   %Expect.ref.loc20: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2.loc20: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %Expect.loc20: type = class_type @Expect, @Expect(constants.%int_2) [template = constants.%Expect.4]
// CHECK:STDOUT:   %Test.ref.loc21: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc21: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_2.loc21: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.left_shift.loc21: init Core.IntLiteral = call %LeftShift.ref.loc21(%int_2.loc21, %int_1.loc21) [template = constants.%int_4]
// CHECK:STDOUT:   %Expect.ref.loc21: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_4.loc21: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %Expect.loc21: type = class_type @Expect, @Expect(constants.%int_4) [template = constants.%Expect.5]
// CHECK:STDOUT:   %Test.ref.loc22: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc22: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc22: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc22: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %int.left_shift.loc22: init Core.IntLiteral = call %LeftShift.ref.loc22(%int_1.loc22, %int_2.loc22) [template = constants.%int_4]
// CHECK:STDOUT:   %Expect.ref.loc22: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_4.loc22: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %Expect.loc22: type = class_type @Expect, @Expect(constants.%int_4) [template = constants.%Expect.5]
// CHECK:STDOUT:   %Test.ref.loc23: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc23: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_3.loc23: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.left_shift.loc23: init Core.IntLiteral = call %LeftShift.ref.loc23(%int_3.loc23, %int_1.loc23) [template = constants.%int_6]
// CHECK:STDOUT:   %Expect.ref.loc23: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12]
// CHECK:STDOUT:   %Expect.loc23: type = class_type @Expect, @Expect(constants.%int_12) [template = constants.%Expect.6]
// CHECK:STDOUT:   %Test.ref.loc26: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc26: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc26_19: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc26_18: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc26_18: <bound method> = bound_method %int_1.loc26_19, %impl.elem0.loc26_18 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc26_18: init Core.IntLiteral = call %Op.bound.loc26_18(%int_1.loc26_19) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_0.loc26: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc26_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc26_18 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc26_18.2: Core.IntLiteral = converted %int.snegate.loc26_18, %.loc26_18.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.left_shift.loc26: init Core.IntLiteral = call %LeftShift.ref.loc26(%.loc26_18.2, %int_0.loc26) [template = constants.%int_-1]
// CHECK:STDOUT:   %Expect.ref.loc26: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1.loc26_37: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc26_36: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc26_36: <bound method> = bound_method %int_1.loc26_37, %impl.elem0.loc26_36 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc26_36: init Core.IntLiteral = call %Op.bound.loc26_36(%int_1.loc26_37) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc26_38.1: Core.IntLiteral = value_of_initializer %int.snegate.loc26_36 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc26_38.2: Core.IntLiteral = converted %int.snegate.loc26_36, %.loc26_38.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Expect.loc26: type = class_type @Expect, @Expect(constants.%int_-1) [template = constants.%Expect.7]
// CHECK:STDOUT:   %Test.ref.loc27: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc27: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc27_19: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc27_18: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc27_18: <bound method> = bound_method %int_1.loc27_19, %impl.elem0.loc27_18 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc27_18: init Core.IntLiteral = call %Op.bound.loc27_18(%int_1.loc27_19) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_1.loc27_22: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc27_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc27_18 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc27_18.2: Core.IntLiteral = converted %int.snegate.loc27_18, %.loc27_18.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.left_shift.loc27: init Core.IntLiteral = call %LeftShift.ref.loc27(%.loc27_18.2, %int_1.loc27_22) [template = constants.%int_-2]
// CHECK:STDOUT:   %Expect.ref.loc27: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2.loc27: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %impl.elem0.loc27_36: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc27_36: <bound method> = bound_method %int_2.loc27, %impl.elem0.loc27_36 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc27_36: init Core.IntLiteral = call %Op.bound.loc27_36(%int_2.loc27) [template = constants.%int_-2]
// CHECK:STDOUT:   %.loc27_38.1: Core.IntLiteral = value_of_initializer %int.snegate.loc27_36 [template = constants.%int_-2]
// CHECK:STDOUT:   %.loc27_38.2: Core.IntLiteral = converted %int.snegate.loc27_36, %.loc27_38.1 [template = constants.%int_-2]
// CHECK:STDOUT:   %Expect.loc27: type = class_type @Expect, @Expect(constants.%int_-2) [template = constants.%Expect.8]
// CHECK:STDOUT:   %Test.ref.loc28: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc28: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_2.loc28: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %impl.elem0.loc28_18: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc28_18: <bound method> = bound_method %int_2.loc28, %impl.elem0.loc28_18 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc28_18: init Core.IntLiteral = call %Op.bound.loc28_18(%int_2.loc28) [template = constants.%int_-2]
// CHECK:STDOUT:   %int_1.loc28: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc28_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc28_18 [template = constants.%int_-2]
// CHECK:STDOUT:   %.loc28_18.2: Core.IntLiteral = converted %int.snegate.loc28_18, %.loc28_18.1 [template = constants.%int_-2]
// CHECK:STDOUT:   %int.left_shift.loc28: init Core.IntLiteral = call %LeftShift.ref.loc28(%.loc28_18.2, %int_1.loc28) [template = constants.%int_-4]
// CHECK:STDOUT:   %Expect.ref.loc28: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_4.loc28: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %impl.elem0.loc28_36: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc28_36: <bound method> = bound_method %int_4.loc28, %impl.elem0.loc28_36 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc28_36: init Core.IntLiteral = call %Op.bound.loc28_36(%int_4.loc28) [template = constants.%int_-4]
// CHECK:STDOUT:   %.loc28_38.1: Core.IntLiteral = value_of_initializer %int.snegate.loc28_36 [template = constants.%int_-4]
// CHECK:STDOUT:   %.loc28_38.2: Core.IntLiteral = converted %int.snegate.loc28_36, %.loc28_38.1 [template = constants.%int_-4]
// CHECK:STDOUT:   %Expect.loc28: type = class_type @Expect, @Expect(constants.%int_-4) [template = constants.%Expect.9]
// CHECK:STDOUT:   %Test.ref.loc29: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc29: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_3.loc29: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %impl.elem0.loc29_18: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc29_18: <bound method> = bound_method %int_3.loc29, %impl.elem0.loc29_18 [template = constants.%Op.bound.4]
// CHECK:STDOUT:   %int.snegate.loc29_18: init Core.IntLiteral = call %Op.bound.loc29_18(%int_3.loc29) [template = constants.%int_-3]
// CHECK:STDOUT:   %int_1.loc29: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc29_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc29_18 [template = constants.%int_-3]
// CHECK:STDOUT:   %.loc29_18.2: Core.IntLiteral = converted %int.snegate.loc29_18, %.loc29_18.1 [template = constants.%int_-3]
// CHECK:STDOUT:   %int.left_shift.loc29: init Core.IntLiteral = call %LeftShift.ref.loc29(%.loc29_18.2, %int_1.loc29) [template = constants.%int_-6]
// CHECK:STDOUT:   %Expect.ref.loc29: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %impl.elem0.loc29_36: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc29_36: <bound method> = bound_method %int_5, %impl.elem0.loc29_36 [template = constants.%Op.bound.5]
// CHECK:STDOUT:   %int.snegate.loc29_36: init Core.IntLiteral = call %Op.bound.loc29_36(%int_5) [template = constants.%int_-5]
// CHECK:STDOUT:   %.loc29_38.1: Core.IntLiteral = value_of_initializer %int.snegate.loc29_36 [template = constants.%int_-5]
// CHECK:STDOUT:   %.loc29_38.2: Core.IntLiteral = converted %int.snegate.loc29_36, %.loc29_38.1 [template = constants.%int_-5]
// CHECK:STDOUT:   %Expect.loc29: type = class_type @Expect, @Expect(constants.%int_-5) [template = constants.%Expect.10]
// CHECK:STDOUT:   %Test.ref.loc32: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc32: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_9223372036854775807: Core.IntLiteral = int_value 9223372036854775807 [template = constants.%int_9223372036854775807]
// CHECK:STDOUT:   %int_1.loc32: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.left_shift.loc32: init Core.IntLiteral = call %LeftShift.ref.loc32(%int_9223372036854775807, %int_1.loc32) [template = constants.%int_18446744073709551614]
// CHECK:STDOUT:   %Expect.ref.loc32: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_18446744073709551614: Core.IntLiteral = int_value 18446744073709551614 [template = constants.%int_18446744073709551614]
// CHECK:STDOUT:   %Expect.loc32: type = class_type @Expect, @Expect(constants.%int_18446744073709551614) [template = constants.%Expect.11]
// CHECK:STDOUT:   %Test.ref.loc33: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %LeftShift.ref.loc33: %LeftShift.type = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_18446744073709551615: Core.IntLiteral = int_value 18446744073709551615 [template = constants.%int_18446744073709551615]
// CHECK:STDOUT:   %int_1.loc33: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.left_shift.loc33: init Core.IntLiteral = call %LeftShift.ref.loc33(%int_18446744073709551615, %int_1.loc33) [template = constants.%int_36893488147419103230]
// CHECK:STDOUT:   %Expect.ref.loc33: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_36893488147419103230: Core.IntLiteral = int_value 36893488147419103230 [template = constants.%int_36893488147419103230]
// CHECK:STDOUT:   %Expect.loc33: type = class_type @Expect, @Expect(constants.%int_36893488147419103230) [template = constants.%Expect.12]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%N) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(<error>) {
// CHECK:STDOUT:   %N.patt.loc11_9.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_0) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_0
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_1) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_1
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_2) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_2
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_4) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_4
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_12) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_12
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-1) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-1
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-2) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-2
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-4) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-4
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-5) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-5
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_18446744073709551614) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_18446744073709551614
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_18446744073709551614
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_36893488147419103230) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_36893488147419103230
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_36893488147419103230
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal_negative.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %LeftShift.type.1: type = fn_type @LeftShift.1 [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface.1: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_1, %Op.2 [template]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.7(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.6: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     .Negate = %import_ref.6
// CHECK:STDOUT:     .ImplicitAs = %import_ref.44
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .bad = @__global_init.%bad
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LeftShift.decl: %LeftShift.type.1 = fn_decl @LeftShift.1 [template = constants.%LeftShift] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_17: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_21: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_33: init type = call %IntLiteral.ref.loc4_21() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_33.1: type = value_of_initializer %int_literal.make_type.loc4_33 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_33.2: type = converted %int_literal.make_type.loc4_33, %.loc4_33.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_39: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_43: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_55: init type = call %IntLiteral.ref.loc4_43() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_55.1: type = value_of_initializer %int_literal.make_type.loc4_55 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_55.2: type = converted %int_literal.make_type.loc4_55, %.loc4_55.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_61: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_65: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_77: init type = call %IntLiteral.ref.loc4_65() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_77.1: type = value_of_initializer %int_literal.make_type.loc4_77 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_77.2: type = converted %int_literal.make_type.loc4_77, %.loc4_77.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift.1(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc10_26: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_1.loc10_30: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc10_29: %Op.type.1 = interface_witness_access constants.%interface.1, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_1.loc10_30, %impl.elem0.loc10_29 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate: init Core.IntLiteral = call %Op.bound(%int_1.loc10_30) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc10_29.1: Core.IntLiteral = value_of_initializer %int.snegate [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc10_29.2: Core.IntLiteral = converted %int.snegate, %.loc10_29.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.left_shift: init Core.IntLiteral = call %LeftShift.ref(%int_1.loc10_26, %.loc10_29.2) [template = <error>]
// CHECK:STDOUT:   %impl.elem0.loc10_32: %Convert.type.2 = interface_witness_access constants.%interface.6, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int.left_shift, %impl.elem0.loc10_32 [template = <error>]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = <error>]
// CHECK:STDOUT:   %.loc10_31.1: Core.IntLiteral = value_of_initializer %int.left_shift [template = <error>]
// CHECK:STDOUT:   %.loc10_31.2: Core.IntLiteral = converted %int.left_shift, %.loc10_31.1 [template = <error>]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%.loc10_31.2) [template = <error>]
// CHECK:STDOUT:   %.loc10_32.1: %i32 = value_of_initializer %int.convert_checked [template = <error>]
// CHECK:STDOUT:   %.loc10_32.2: %i32 = converted %int.left_shift, %.loc10_32.1 [template = <error>]
// CHECK:STDOUT:   %bad: %i32 = bind_name bad, %.loc10_32.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %LeftShift.type.1: type = fn_type @LeftShift.1 [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_4000000000: Core.IntLiteral = int_value 4000000000 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface.6: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_1, %Op.2 [template]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     .Negate = %import_ref.230
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .bad = @__global_init.%bad
// CHECK:STDOUT:     .bad_negative = @__global_init.%bad_negative
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LeftShift.decl: %LeftShift.type.1 = fn_decl @LeftShift.1 [template = constants.%LeftShift] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_17: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_21: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_33: init type = call %IntLiteral.ref.loc4_21() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_33.1: type = value_of_initializer %int_literal.make_type.loc4_33 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_33.2: type = converted %int_literal.make_type.loc4_33, %.loc4_33.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_39: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_43: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_55: init type = call %IntLiteral.ref.loc4_43() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_55.1: type = value_of_initializer %int_literal.make_type.loc4_55 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_55.2: type = converted %int_literal.make_type.loc4_55, %.loc4_55.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_61: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_65: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_77: init type = call %IntLiteral.ref.loc4_65() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_77.1: type = value_of_initializer %int_literal.make_type.loc4_77 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_77.2: type = converted %int_literal.make_type.loc4_77, %.loc4_77.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift.1(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref.loc10: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_4000000000.loc10: Core.IntLiteral = int_value 4000000000 [template = constants.%int_4000000000]
// CHECK:STDOUT:   %int.left_shift.loc10: init Core.IntLiteral = call %LeftShift.ref.loc10(%int_1.loc10, %int_4000000000.loc10) [template = <error>]
// CHECK:STDOUT:   %impl.elem0.loc10: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10: <bound method> = bound_method %int.left_shift.loc10, %impl.elem0.loc10 [template = <error>]
// CHECK:STDOUT:   %Convert.specific_fn.loc10: <specific function> = specific_function %Convert.bound.loc10, @Convert.2(constants.%int_32) [template = <error>]
// CHECK:STDOUT:   %.loc10_41.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc10 [template = <error>]
// CHECK:STDOUT:   %.loc10_41.2: Core.IntLiteral = converted %int.left_shift.loc10, %.loc10_41.1 [template = <error>]
// CHECK:STDOUT:   %int.convert_checked.loc10: init %i32 = call %Convert.specific_fn.loc10(%.loc10_41.2) [template = <error>]
// CHECK:STDOUT:   %.loc10_42.1: %i32 = value_of_initializer %int.convert_checked.loc10 [template = <error>]
// CHECK:STDOUT:   %.loc10_42.2: %i32 = converted %int.left_shift.loc10, %.loc10_42.1 [template = <error>]
// CHECK:STDOUT:   %bad: %i32 = bind_name bad, %.loc10_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc15: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc15_35: %Op.type.1 = interface_witness_access constants.%interface.6, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15_35 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate: init Core.IntLiteral = call %Op.bound(%int_1.loc15) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_4000000000.loc15: Core.IntLiteral = int_value 4000000000 [template = constants.%int_4000000000]
// CHECK:STDOUT:   %.loc15_35.1: Core.IntLiteral = value_of_initializer %int.snegate [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc15_35.2: Core.IntLiteral = converted %int.snegate, %.loc15_35.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.left_shift.loc15: init Core.IntLiteral = call %LeftShift.ref.loc15(%.loc15_35.2, %int_4000000000.loc15) [template = <error>]
// CHECK:STDOUT:   %impl.elem0.loc15_52: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15: <bound method> = bound_method %int.left_shift.loc15, %impl.elem0.loc15_52 [template = <error>]
// CHECK:STDOUT:   %Convert.specific_fn.loc15: <specific function> = specific_function %Convert.bound.loc15, @Convert.2(constants.%int_32) [template = <error>]
// CHECK:STDOUT:   %.loc15_51.1: Core.IntLiteral = value_of_initializer %int.left_shift.loc15 [template = <error>]
// CHECK:STDOUT:   %.loc15_51.2: Core.IntLiteral = converted %int.left_shift.loc15, %.loc15_51.1 [template = <error>]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32 = call %Convert.specific_fn.loc15(%.loc15_51.2) [template = <error>]
// CHECK:STDOUT:   %.loc15_52.1: %i32 = value_of_initializer %int.convert_checked.loc15 [template = <error>]
// CHECK:STDOUT:   %.loc15_52.2: %i32 = converted %int.left_shift.loc15, %.loc15_52.1 [template = <error>]
// CHECK:STDOUT:   %bad_negative: %i32 = bind_name bad_negative, %.loc15_52.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
