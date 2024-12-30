// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/sdiv.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/sdiv.carbon

// --- int_div.carbon

fn Div(a: i32, b: i32) -> i32 = "int.sdiv";

var arr: [i32; Div(3, 2)];
let arr_p: [i32; 1]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Div(a, b);
}

// --- fail_overflow.carbon

package FailOverflow;

fn Div(a: i32, b: i32) -> i32 = "int.sdiv";
fn Sub(a: i32, b: i32) -> i32 = "int.ssub";
fn Negate(a: i32) -> i32 = "int.snegate";

// -0x7FFF_FFFF / -1 is OK.
let a: i32 = Div(-0x7FFF_FFFF, -1);

// -0x8000_0000 / 1 is OK.
let b: i32 = Div(-0x8000_0000, 1);

// -0x8000_0000 / -1 overflows.
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:14: error: integer overflow in calculation `-2147483648 / -1` [CompileTimeIntegerOverflow]
// CHECK:STDERR: let c: i32 = Div(-0x8000_0000, -1);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let c: i32 = Div(-0x8000_0000, -1);

// --- literal_no_overflow.carbon

library "[[@TEST_NAME]]";

fn Div(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.sdiv";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(Div(-0x8000_0000, -1)) as Expect(0x8000_0000);
  Test(Div(-0x8000_0000_0000_0000, -1)) as Expect(0x8000_0000_0000_0000);
}

// --- fail_div_by_zero.carbon

package FailDivByZero;

fn Div(a: i32, b: i32) -> i32 = "int.sdiv";
fn DivLit(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.sdiv";

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+4]]:14: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let a: i32 = Div(1, 0);
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR:
let a: i32 = Div(1, 0);

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+4]]:14: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let b: i32 = Div(0, 0);
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR:
let b: i32 = Div(0, 0);

// IntLiteral allows "overflow" by widening its representation, but not overflow to infinity.
// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+4]]:28: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let c: Core.IntLiteral() = DivLit(1, 0);
// CHECK:STDERR:                            ^~~~~~~~~~~~
// CHECK:STDERR:
let c: Core.IntLiteral() = DivLit(1, 0);

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+3]]:28: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let d: Core.IntLiteral() = DivLit(0, 0);
// CHECK:STDERR:                            ^~~~~~~~~~~~
let d: Core.IntLiteral() = DivLit(0, 0);

// CHECK:STDOUT: --- int_div.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Div.type.1: type = fn_type @Div.1 [template]
// CHECK:STDOUT:   %Div: %Div.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_1.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.2, %i32 [template]
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
// CHECK:STDOUT:     .Div = %Div.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Div.decl: %Div.type.1 = fn_decl @Div.1 [template = constants.%Div] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc2_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc2_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc2_11: type = splice_block %i32.loc2_11 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc2_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc2_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc2_19: type = splice_block %i32.loc2_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc2_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc2_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr.var: ref %array_type = var arr
// CHECK:STDOUT:   %arr: ref %array_type = bind_name arr, %arr.var
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc7_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7_35: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc7_19: type = splice_block %i32.loc7_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc7_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc7_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc7_27: type = splice_block %i32.loc7_27 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc7_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc7_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Div.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.sdiv";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Div.ref: %Div.type.1 = name_ref Div, file.%Div.decl [template = constants.%Div]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.sdiv: init %i32 = call %Div.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19.1: %i32 = value_of_initializer %int.sdiv
// CHECK:STDOUT:   %.loc8_19.2: %i32 = converted %int.sdiv, %.loc8_19.1
// CHECK:STDOUT:   return %.loc8_19.2
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
// CHECK:STDOUT: --- fail_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Div.type.1: type = fn_type @Div.1 [template]
// CHECK:STDOUT:   %Div: %Div.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Sub.type.1: type = fn_type @Sub.1 [template]
// CHECK:STDOUT:   %Sub: %Sub.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Negate.type.1: type = fn_type @Negate.1 [template]
// CHECK:STDOUT:   %Negate: %Negate.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_2147483647.1: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface.1: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound.1: <bound method> = bound_method %int_2147483647.1, %Op.2 [template]
// CHECK:STDOUT:   %int_-2147483647.1: Core.IntLiteral = int_value -2147483647 [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Op.bound.2: <bound method> = bound_method %int_1.1, %Op.2 [template]
// CHECK:STDOUT:   %int_-1.1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.7(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.20: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_-2147483647.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_-2147483647.2: %i32 = int_value -2147483647 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_-1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_-1.2: %i32 = int_value -1 [template]
// CHECK:STDOUT:   %int_2147483647.2: %i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %int_2147483648: Core.IntLiteral = int_value 2147483648 [template]
// CHECK:STDOUT:   %Op.bound.3: <bound method> = bound_method %int_2147483648, %Op.2 [template]
// CHECK:STDOUT:   %int_-2147483648.1: Core.IntLiteral = int_value -2147483648 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_-2147483648.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_-2147483648.2: %i32 = int_value -2147483648 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .Negate = %import_ref.5
// CHECK:STDOUT:     .ImplicitAs = %import_ref.43
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Div = %Div.decl
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Div.decl: %Div.type.1 = fn_decl @Div.1 [template = constants.%Div] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_11: type = splice_block %i32.loc4_11 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_19: type = splice_block %i32.loc4_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Sub.decl: %Sub.type.1 = fn_decl @Sub.1 [template = constants.%Sub] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc5_11: type = splice_block %i32.loc5_11 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc5_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc5_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc5_19: type = splice_block %i32.loc5_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc5_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc5_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
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
// CHECK:STDOUT:     %int_32.loc6_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %i32.loc6_14 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc6_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Div.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.sdiv";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.ssub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate.1(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Div.ref.loc9: %Div.type.1 = name_ref Div, file.%Div.decl [template = constants.%Div]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %impl.elem0.loc9_18.1: %Op.type.1 = interface_witness_access constants.%interface.1, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc9_18: <bound method> = bound_method %int_2147483647, %impl.elem0.loc9_18.1 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc9_18: init Core.IntLiteral = call %Op.bound.loc9_18(%int_2147483647) [template = constants.%int_-2147483647.1]
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc9_32.1: %Op.type.1 = interface_witness_access constants.%interface.1, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc9_32: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_32.1 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc9_32: init Core.IntLiteral = call %Op.bound.loc9_32(%int_1.loc9) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc9_18.2: %Convert.type.2 = interface_witness_access constants.%interface.20, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc9_18: <bound method> = bound_method %int.snegate.loc9_18, %impl.elem0.loc9_18.2 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_18: <specific function> = specific_function %Convert.bound.loc9_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %.loc9_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc9_18 [template = constants.%int_-2147483647.1]
// CHECK:STDOUT:   %.loc9_18.2: Core.IntLiteral = converted %int.snegate.loc9_18, %.loc9_18.1 [template = constants.%int_-2147483647.1]
// CHECK:STDOUT:   %int.convert_checked.loc9_18: init %i32 = call %Convert.specific_fn.loc9_18(%.loc9_18.2) [template = constants.%int_-2147483647.2]
// CHECK:STDOUT:   %.loc9_18.3: %i32 = value_of_initializer %int.convert_checked.loc9_18 [template = constants.%int_-2147483647.2]
// CHECK:STDOUT:   %.loc9_18.4: %i32 = converted %int.snegate.loc9_18, %.loc9_18.3 [template = constants.%int_-2147483647.2]
// CHECK:STDOUT:   %impl.elem0.loc9_32.2: %Convert.type.2 = interface_witness_access constants.%interface.20, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc9_32: <bound method> = bound_method %int.snegate.loc9_32, %impl.elem0.loc9_32.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_32: <specific function> = specific_function %Convert.bound.loc9_32, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %.loc9_32.1: Core.IntLiteral = value_of_initializer %int.snegate.loc9_32 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc9_32.2: Core.IntLiteral = converted %int.snegate.loc9_32, %.loc9_32.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc9_32: init %i32 = call %Convert.specific_fn.loc9_32(%.loc9_32.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc9_32.3: %i32 = value_of_initializer %int.convert_checked.loc9_32 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc9_32.4: %i32 = converted %int.snegate.loc9_32, %.loc9_32.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.sdiv.loc9: init %i32 = call %Div.ref.loc9(%.loc9_18.4, %.loc9_32.4) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc9_35.1: %i32 = value_of_initializer %int.sdiv.loc9 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc9_35.2: %i32 = converted %int.sdiv.loc9, %.loc9_35.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc9_35.2
// CHECK:STDOUT:   %Div.ref.loc12: %Div.type.1 = name_ref Div, file.%Div.decl [template = constants.%Div]
// CHECK:STDOUT:   %int_2147483648.loc12: Core.IntLiteral = int_value 2147483648 [template = constants.%int_2147483648]
// CHECK:STDOUT:   %impl.elem0.loc12_18.1: %Op.type.1 = interface_witness_access constants.%interface.1, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc12: <bound method> = bound_method %int_2147483648.loc12, %impl.elem0.loc12_18.1 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc12: init Core.IntLiteral = call %Op.bound.loc12(%int_2147483648.loc12) [template = constants.%int_-2147483648.1]
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc12_18.2: %Convert.type.2 = interface_witness_access constants.%interface.20, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_18: <bound method> = bound_method %int.snegate.loc12, %impl.elem0.loc12_18.2 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_18: <specific function> = specific_function %Convert.bound.loc12_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %.loc12_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc12 [template = constants.%int_-2147483648.1]
// CHECK:STDOUT:   %.loc12_18.2: Core.IntLiteral = converted %int.snegate.loc12, %.loc12_18.1 [template = constants.%int_-2147483648.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_18: init %i32 = call %Convert.specific_fn.loc12_18(%.loc12_18.2) [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %.loc12_18.3: %i32 = value_of_initializer %int.convert_checked.loc12_18 [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %.loc12_18.4: %i32 = converted %int.snegate.loc12, %.loc12_18.3 [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %impl.elem0.loc12_32: %Convert.type.2 = interface_witness_access constants.%interface.20, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_32: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12_32 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_32: <specific function> = specific_function %Convert.bound.loc12_32, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc12_32: init %i32 = call %Convert.specific_fn.loc12_32(%int_1.loc12) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_32.1: %i32 = value_of_initializer %int.convert_checked.loc12_32 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_32.2: %i32 = converted %int_1.loc12, %.loc12_32.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.sdiv.loc12: init %i32 = call %Div.ref.loc12(%.loc12_18.4, %.loc12_32.2) [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %.loc12_34.1: %i32 = value_of_initializer %int.sdiv.loc12 [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %.loc12_34.2: %i32 = converted %int.sdiv.loc12, %.loc12_34.1 [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc12_34.2
// CHECK:STDOUT:   %Div.ref.loc19: %Div.type.1 = name_ref Div, file.%Div.decl [template = constants.%Div]
// CHECK:STDOUT:   %int_2147483648.loc19: Core.IntLiteral = int_value 2147483648 [template = constants.%int_2147483648]
// CHECK:STDOUT:   %impl.elem0.loc19_18.1: %Op.type.1 = interface_witness_access constants.%interface.1, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc19_18: <bound method> = bound_method %int_2147483648.loc19, %impl.elem0.loc19_18.1 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc19_18: init Core.IntLiteral = call %Op.bound.loc19_18(%int_2147483648.loc19) [template = constants.%int_-2147483648.1]
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc19_32.1: %Op.type.1 = interface_witness_access constants.%interface.1, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc19_32: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_32.1 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc19_32: init Core.IntLiteral = call %Op.bound.loc19_32(%int_1.loc19) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc19_18.2: %Convert.type.2 = interface_witness_access constants.%interface.20, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc19_18: <bound method> = bound_method %int.snegate.loc19_18, %impl.elem0.loc19_18.2 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc19_18: <specific function> = specific_function %Convert.bound.loc19_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %.loc19_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc19_18 [template = constants.%int_-2147483648.1]
// CHECK:STDOUT:   %.loc19_18.2: Core.IntLiteral = converted %int.snegate.loc19_18, %.loc19_18.1 [template = constants.%int_-2147483648.1]
// CHECK:STDOUT:   %int.convert_checked.loc19_18: init %i32 = call %Convert.specific_fn.loc19_18(%.loc19_18.2) [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %.loc19_18.3: %i32 = value_of_initializer %int.convert_checked.loc19_18 [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %.loc19_18.4: %i32 = converted %int.snegate.loc19_18, %.loc19_18.3 [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %impl.elem0.loc19_32.2: %Convert.type.2 = interface_witness_access constants.%interface.20, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc19_32: <bound method> = bound_method %int.snegate.loc19_32, %impl.elem0.loc19_32.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc19_32: <specific function> = specific_function %Convert.bound.loc19_32, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %.loc19_32.1: Core.IntLiteral = value_of_initializer %int.snegate.loc19_32 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc19_32.2: Core.IntLiteral = converted %int.snegate.loc19_32, %.loc19_32.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc19_32: init %i32 = call %Convert.specific_fn.loc19_32(%.loc19_32.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc19_32.3: %i32 = value_of_initializer %int.convert_checked.loc19_32 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc19_32.4: %i32 = converted %int.snegate.loc19_32, %.loc19_32.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.sdiv.loc19: init %i32 = call %Div.ref.loc19(%.loc19_18.4, %.loc19_32.4) [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %.loc19_35.1: %i32 = value_of_initializer %int.sdiv.loc19 [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %.loc19_35.2: %i32 = converted %int.sdiv.loc19, %.loc19_35.1 [template = constants.%int_-2147483648.2]
// CHECK:STDOUT:   %c: %i32 = bind_name c, %.loc19_35.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal_no_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Div.type.1: type = fn_type @Div.1 [template]
// CHECK:STDOUT:   %Div: %Div.type.1 = struct_value () [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Expect.type: type = generic_class_type @Expect [template]
// CHECK:STDOUT:   %Expect.generic: %Expect.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.1: type = class_type @Expect, @Expect(%N) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Expect.1 [symbolic]
// CHECK:STDOUT:   %Expect.val.1: %Expect.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_2147483648: Core.IntLiteral = int_value 2147483648 [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound.1: <bound method> = bound_method %int_2147483648, %Op.2 [template]
// CHECK:STDOUT:   %int_-2147483648: Core.IntLiteral = int_value -2147483648 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Op.bound.2: <bound method> = bound_method %int_1, %Op.2 [template]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%int_2147483648) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%int_2147483648) [template]
// CHECK:STDOUT:   %int_9223372036854775808: Core.IntLiteral = int_value 9223372036854775808 [template]
// CHECK:STDOUT:   %Op.bound.3: <bound method> = bound_method %int_9223372036854775808, %Op.2 [template]
// CHECK:STDOUT:   %int_-9223372036854775808: Core.IntLiteral = int_value -9223372036854775808 [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%int_9223372036854775808) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%int_9223372036854775808) [template]
// CHECK:STDOUT:   %Expect.val.2: %Expect.2 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.3: %Expect.3 = struct_value () [template]
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
// CHECK:STDOUT:     .Div = %Div.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Div.decl: %Div.type.1 = fn_decl @Div.1 [template = constants.%Div] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_55: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_59: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_71: init type = call %IntLiteral.ref.loc4_59() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_71.1: type = value_of_initializer %int_literal.make_type.loc4_71 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_71.2: type = converted %int_literal.make_type.loc4_71, %.loc4_71.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_27.1: type = splice_block %.loc4_27.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_11: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_15: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_27: init type = call %IntLiteral.ref.loc4_15() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_27.2: type = value_of_initializer %int_literal.make_type.loc4_27 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_27.3: type = converted %int_literal.make_type.loc4_27, %.loc4_27.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_49.1: type = splice_block %.loc4_49.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_33: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_37: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_49: init type = call %IntLiteral.ref.loc4_37() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_49.2: type = value_of_initializer %int_literal.make_type.loc4_49 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_49.3: type = converted %int_literal.make_type.loc4_49, %.loc4_49.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %N.patt.loc6_14.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_14.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc6_14.1, runtime_param<invalid> [symbolic = %N.patt.loc6_14.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6_34.1: type = splice_block %.loc6_34.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_34.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_34.3: type = converted %int_literal.make_type, %.loc6_34.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_14.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_14.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %N.patt.loc7_9.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_9.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc7_9.1, runtime_param<invalid> [symbolic = %N.patt.loc7_9.2 (constants.%N.patt)]
// CHECK:STDOUT:     %return.patt: @Test.%Expect.loc7_43.2 (%Expect.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Test.%Expect.loc7_43.2 (%Expect.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %N.ref: Core.IntLiteral = name_ref N, %N.loc7_9.1 [symbolic = %N.loc7_9.2 (constants.%N)]
// CHECK:STDOUT:     %Expect.loc7_43.1: type = class_type @Expect, @Expect(constants.%N) [symbolic = %Expect.loc7_43.2 (constants.%Expect.1)]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc7_29.1: type = splice_block %.loc7_29.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_29.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_29.3: type = converted %int_literal.make_type, %.loc7_29.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc7_9.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc7_9.2 (constants.%N)]
// CHECK:STDOUT:     %return.param: ref @Test.%Expect.loc7_43.2 (%Expect.1) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Test.%Expect.loc7_43.2 (%Expect.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Expect(%N.loc6_14.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc6_14.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc6_14.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_14.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_14.2 (constants.%N.patt)]
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
// CHECK:STDOUT: fn @Div.1(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.sdiv";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%N.loc7_9.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc7_9.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc7_9.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc7_9.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_9.2 (constants.%N.patt)]
// CHECK:STDOUT:   %Expect.loc7_43.2: type = class_type @Expect, @Expect(%N.loc7_9.2) [symbolic = %Expect.loc7_43.2 (constants.%Expect.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Test.%Expect.loc7_43.2 (%Expect.1) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Expect.val: @Test.%Expect.loc7_43.2 (%Expect.1) = struct_value () [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: Core.IntLiteral) -> %return.param_patt: @Test.%Expect.loc7_43.2 (%Expect.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc7_55.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_55.2: init @Test.%Expect.loc7_43.2 (%Expect.1) = class_init (), %return [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     %.loc7_56: init @Test.%Expect.loc7_43.2 (%Expect.1) = converted %.loc7_55.1, %.loc7_55.2 [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     return %.loc7_56 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref.loc10: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Div.ref.loc10: %Div.type.1 = name_ref Div, file.%Div.decl [template = constants.%Div]
// CHECK:STDOUT:   %int_2147483648.loc10_13: Core.IntLiteral = int_value 2147483648 [template = constants.%int_2147483648]
// CHECK:STDOUT:   %impl.elem0.loc10_12: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc10_12: <bound method> = bound_method %int_2147483648.loc10_13, %impl.elem0.loc10_12 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc10_12: init Core.IntLiteral = call %Op.bound.loc10_12(%int_2147483648.loc10_13) [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc10_26: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc10_26: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10_26 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc10_26: init Core.IntLiteral = call %Op.bound.loc10_26(%int_1.loc10) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc10_12.1: Core.IntLiteral = value_of_initializer %int.snegate.loc10_12 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc10_12.2: Core.IntLiteral = converted %int.snegate.loc10_12, %.loc10_12.1 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc10_26.1: Core.IntLiteral = value_of_initializer %int.snegate.loc10_26 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc10_26.2: Core.IntLiteral = converted %int.snegate.loc10_26, %.loc10_26.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.sdiv.loc10: init Core.IntLiteral = call %Div.ref.loc10(%.loc10_12.2, %.loc10_26.2) [template = constants.%int_2147483648]
// CHECK:STDOUT:   %.loc10_29.1: Core.IntLiteral = value_of_initializer %int.sdiv.loc10 [template = constants.%int_2147483648]
// CHECK:STDOUT:   %.loc10_29.2: Core.IntLiteral = converted %int.sdiv.loc10, %.loc10_29.1 [template = constants.%int_2147483648]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%int_2147483648) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_29.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_29.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2147483648.loc10_41: Core.IntLiteral = int_value 2147483648 [template = constants.%int_2147483648]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%int_2147483648) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_29.4: ref %Expect.2 = temporary %.loc10_29.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Div.ref.loc11: %Div.type.1 = name_ref Div, file.%Div.decl [template = constants.%Div]
// CHECK:STDOUT:   %int_9223372036854775808.loc11_13: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %impl.elem0.loc11_12: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc11_12: <bound method> = bound_method %int_9223372036854775808.loc11_13, %impl.elem0.loc11_12 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc11_12: init Core.IntLiteral = call %Op.bound.loc11_12(%int_9223372036854775808.loc11_13) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc11_36: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc11_36: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_36 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc11_36: init Core.IntLiteral = call %Op.bound.loc11_36(%int_1.loc11) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc11_12.1: Core.IntLiteral = value_of_initializer %int.snegate.loc11_12 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc11_12.2: Core.IntLiteral = converted %int.snegate.loc11_12, %.loc11_12.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc11_36.1: Core.IntLiteral = value_of_initializer %int.snegate.loc11_36 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc11_36.2: Core.IntLiteral = converted %int.snegate.loc11_36, %.loc11_36.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.sdiv.loc11: init Core.IntLiteral = call %Div.ref.loc11(%.loc11_12.2, %.loc11_36.2) [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %.loc11_39.1: Core.IntLiteral = value_of_initializer %int.sdiv.loc11 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %.loc11_39.2: Core.IntLiteral = converted %int.sdiv.loc11, %.loc11_39.1 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%int_9223372036854775808) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc11_39.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.3 = call %Test.specific_fn.loc11() to %.loc11_39.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_9223372036854775808.loc11_51: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%int_9223372036854775808) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc11_39.4: ref %Expect.3 = temporary %.loc11_39.3, %Test.call.loc11
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%N) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(@Test.%N.loc7_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%N) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%N
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_2147483648) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_2147483648
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_2147483648
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_2147483648) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_2147483648
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_2147483648
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_9223372036854775808) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_9223372036854775808
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_9223372036854775808
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_9223372036854775808) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_9223372036854775808
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_9223372036854775808
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_div_by_zero.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Div.type.1: type = fn_type @Div.1 [template]
// CHECK:STDOUT:   %Div: %Div.type.1 = struct_value () [template]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %DivLit.type: type = fn_type @DivLit [template]
// CHECK:STDOUT:   %DivLit: %DivLit.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_0.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .IntLiteral = %import_ref.5
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.5: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Div = %Div.decl
// CHECK:STDOUT:     .DivLit = %DivLit.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:     .d = @__global_init.%d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Div.decl: %Div.type.1 = fn_decl @Div.1 [template = constants.%Div] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_11: type = splice_block %i32.loc4_11 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_19: type = splice_block %i32.loc4_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DivLit.decl: %DivLit.type = fn_decl @DivLit [template = constants.%DivLit] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc5_58: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc5_62: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc5_74: init type = call %IntLiteral.ref.loc5_62() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_74.1: type = value_of_initializer %int_literal.make_type.loc5_74 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc5_74.2: type = converted %int_literal.make_type.loc5_74, %.loc5_74.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc5_30.1: type = splice_block %.loc5_30.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc5_14: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc5_18: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc5_30: init type = call %IntLiteral.ref.loc5_18() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_30.2: type = value_of_initializer %int_literal.make_type.loc5_30 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_30.3: type = converted %int_literal.make_type.loc5_30, %.loc5_30.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %.loc5_52.1: type = splice_block %.loc5_52.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc5_36: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc5_40: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.5 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc5_52: init type = call %IntLiteral.ref.loc5_40() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_52.2: type = value_of_initializer %int_literal.make_type.loc5_52 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_52.3: type = converted %int_literal.make_type.loc5_52, %.loc5_52.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Div.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.sdiv";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DivLit(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.sdiv";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Div.ref.loc11: %Div.type.1 = name_ref Div, file.%Div.decl [template = constants.%Div]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_0.loc11: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc11_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_18: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_18 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_18: <specific function> = specific_function %Convert.bound.loc11_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_18: init %i32 = call %Convert.specific_fn.loc11_18(%int_1.loc11) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_18.1: %i32 = value_of_initializer %int.convert_checked.loc11_18 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_18.2: %i32 = converted %int_1.loc11, %.loc11_18.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc11_21: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_21: <bound method> = bound_method %int_0.loc11, %impl.elem0.loc11_21 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_21: <specific function> = specific_function %Convert.bound.loc11_21, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_21: init %i32 = call %Convert.specific_fn.loc11_21(%int_0.loc11) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_21.1: %i32 = value_of_initializer %int.convert_checked.loc11_21 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc11_21.2: %i32 = converted %int_0.loc11, %.loc11_21.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int.sdiv.loc11: init %i32 = call %Div.ref.loc11(%.loc11_18.2, %.loc11_21.2) [template = <error>]
// CHECK:STDOUT:   %.loc11_23.1: %i32 = value_of_initializer %int.sdiv.loc11 [template = <error>]
// CHECK:STDOUT:   %.loc11_23.2: %i32 = converted %int.sdiv.loc11, %.loc11_23.1 [template = <error>]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc11_23.2
// CHECK:STDOUT:   %Div.ref.loc17: %Div.type.1 = name_ref Div, file.%Div.decl [template = constants.%Div]
// CHECK:STDOUT:   %int_0.loc17_18: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_0.loc17_21: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc17_18: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc17_18: <bound method> = bound_method %int_0.loc17_18, %impl.elem0.loc17_18 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_18: <specific function> = specific_function %Convert.bound.loc17_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc17_18: init %i32 = call %Convert.specific_fn.loc17_18(%int_0.loc17_18) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc17_18.1: %i32 = value_of_initializer %int.convert_checked.loc17_18 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc17_18.2: %i32 = converted %int_0.loc17_18, %.loc17_18.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %impl.elem0.loc17_21: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc17_21: <bound method> = bound_method %int_0.loc17_21, %impl.elem0.loc17_21 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_21: <specific function> = specific_function %Convert.bound.loc17_21, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc17_21: init %i32 = call %Convert.specific_fn.loc17_21(%int_0.loc17_21) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc17_21.1: %i32 = value_of_initializer %int.convert_checked.loc17_21 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc17_21.2: %i32 = converted %int_0.loc17_21, %.loc17_21.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int.sdiv.loc17: init %i32 = call %Div.ref.loc17(%.loc17_18.2, %.loc17_21.2) [template = <error>]
// CHECK:STDOUT:   %.loc17_23.1: %i32 = value_of_initializer %int.sdiv.loc17 [template = <error>]
// CHECK:STDOUT:   %.loc17_23.2: %i32 = converted %int.sdiv.loc17, %.loc17_23.1 [template = <error>]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc17_23.2
// CHECK:STDOUT:   %DivLit.ref.loc24: %DivLit.type = name_ref DivLit, file.%DivLit.decl [template = constants.%DivLit]
// CHECK:STDOUT:   %int_1.loc24: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_0.loc24: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int.sdiv.loc24: init Core.IntLiteral = call %DivLit.ref.loc24(%int_1.loc24, %int_0.loc24) [template = <error>]
// CHECK:STDOUT:   %.loc24_40.1: Core.IntLiteral = value_of_initializer %int.sdiv.loc24 [template = <error>]
// CHECK:STDOUT:   %.loc24_40.2: Core.IntLiteral = converted %int.sdiv.loc24, %.loc24_40.1 [template = <error>]
// CHECK:STDOUT:   %c: Core.IntLiteral = bind_name c, %.loc24_40.2
// CHECK:STDOUT:   %DivLit.ref.loc29: %DivLit.type = name_ref DivLit, file.%DivLit.decl [template = constants.%DivLit]
// CHECK:STDOUT:   %int_0.loc29_35: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_0.loc29_38: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int.sdiv.loc29: init Core.IntLiteral = call %DivLit.ref.loc29(%int_0.loc29_35, %int_0.loc29_38) [template = <error>]
// CHECK:STDOUT:   %.loc29_40.1: Core.IntLiteral = value_of_initializer %int.sdiv.loc29 [template = <error>]
// CHECK:STDOUT:   %.loc29_40.2: Core.IntLiteral = converted %int.sdiv.loc29, %.loc29_40.1 [template = <error>]
// CHECK:STDOUT:   %d: Core.IntLiteral = bind_name d, %.loc29_40.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
