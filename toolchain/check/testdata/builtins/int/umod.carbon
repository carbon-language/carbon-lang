// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/umod.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/umod.carbon

// --- int_div.carbon

fn Mod(a: i32, b: i32) -> i32 = "int.umod";

var arr: [i32; Mod(5, 3)];
let arr_p: [i32; 2]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Mod(a, b);
}

// --- overflow.carbon

package Overflow;

fn Mod(a: i32, b: i32) -> i32 = "int.umod";
fn Sub(a: i32, b: i32) -> i32 = "int.usub";
fn Negate(a: i32) -> i32 = "int.unegate";

// -0x7FFF_FFFF % -1 is OK.
let a: i32 = Mod(Negate(0x7FFF_FFFF), Negate(1));

// -0x8000_0000 % 1 is OK.
let b: i32 = Mod(Sub(Negate(0x7FFF_FFFF), 1), 1);

// -0x8000_0000 / -1 is OK.
let c: i32 = Mod(Sub(Negate(0x7FFF_FFFF), 1), Negate(1));

// --- fail_div_by_zero.carbon

package FailDivByZero;

fn Mod(a: i32, b: i32) -> i32 = "int.umod";

// Remainder of division by zero is not defined.

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+4]]:14: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let a: i32 = Mod(1, 0);
// CHECK:STDERR:              ^~~~~~~~~
// CHECK:STDERR:
let a: i32 = Mod(1, 0);

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+3]]:14: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let b: i32 = Mod(0, 0);
// CHECK:STDERR:              ^~~~~~~~~
let b: i32 = Mod(0, 0);

// CHECK:STDOUT: --- int_div.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Mod.type.1: type = fn_type @Mod.1 [template]
// CHECK:STDOUT:   %Mod: %Mod.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_2.2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_2.2, %i32 [template]
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
// CHECK:STDOUT:     .Mod = %Mod.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mod.decl: %Mod.type.1 = fn_decl @Mod.1 [template = constants.%Mod] {
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
// CHECK:STDOUT: fn @Mod.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.umod";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mod.ref: %Mod.type.1 = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.umod: init %i32 = call %Mod.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19.1: %i32 = value_of_initializer %int.umod
// CHECK:STDOUT:   %.loc8_19.2: %i32 = converted %int.umod, %.loc8_19.1
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
// CHECK:STDOUT: --- overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Mod.type.1: type = fn_type @Mod.1 [template]
// CHECK:STDOUT:   %Mod: %Mod.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Sub.type.1: type = fn_type @Sub.1 [template]
// CHECK:STDOUT:   %Sub: %Sub.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Negate.type.1: type = fn_type @Negate.1 [template]
// CHECK:STDOUT:   %Negate: %Negate.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_2147483647.1: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_2147483647.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2147483647.2: %i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %int_-2147483647: %i32 = int_value -2147483647 [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_-1: %i32 = int_value -1 [template]
// CHECK:STDOUT:   %int_-2147483648: %i32 = int_value -2147483648 [template]
// CHECK:STDOUT:   %int_0: %i32 = int_value 0 [template]
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
// CHECK:STDOUT:     .Mod = %Mod.decl
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mod.decl: %Mod.type.1 = fn_decl @Mod.1 [template = constants.%Mod] {
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
// CHECK:STDOUT: fn @Mod.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.umod";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.usub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate.1(%a.param_patt: %i32) -> %i32 = "int.unegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mod.ref.loc9: %Mod.type.1 = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %Negate.ref.loc9_18: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_2147483647.loc9: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %impl.elem0.loc9_25: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc9_25: <bound method> = bound_method %int_2147483647.loc9, %impl.elem0.loc9_25 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_25: <specific function> = specific_function %Convert.bound.loc9_25, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc9_25: init %i32 = call %Convert.specific_fn.loc9_25(%int_2147483647.loc9) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc9_25.1: %i32 = value_of_initializer %int.convert_checked.loc9_25 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc9_25.2: %i32 = converted %int_2147483647.loc9, %.loc9_25.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %int.unegate.loc9_36: init %i32 = call %Negate.ref.loc9_18(%.loc9_25.2) [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %Negate.ref.loc9_39: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc9_46: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc9_46: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_46 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_46: <specific function> = specific_function %Convert.bound.loc9_46, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc9_46: init %i32 = call %Convert.specific_fn.loc9_46(%int_1.loc9) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc9_46.1: %i32 = value_of_initializer %int.convert_checked.loc9_46 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc9_46.2: %i32 = converted %int_1.loc9, %.loc9_46.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.unegate.loc9_47: init %i32 = call %Negate.ref.loc9_39(%.loc9_46.2) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc9_36.1: %i32 = value_of_initializer %int.unegate.loc9_36 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %.loc9_36.2: %i32 = converted %int.unegate.loc9_36, %.loc9_36.1 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %.loc9_47.1: %i32 = value_of_initializer %int.unegate.loc9_47 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc9_47.2: %i32 = converted %int.unegate.loc9_47, %.loc9_47.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.umod.loc9: init %i32 = call %Mod.ref.loc9(%.loc9_36.2, %.loc9_47.2) [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %.loc9_49.1: %i32 = value_of_initializer %int.umod.loc9 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %.loc9_49.2: %i32 = converted %int.umod.loc9, %.loc9_49.1 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc9_49.2
// CHECK:STDOUT:   %Mod.ref.loc12: %Mod.type.1 = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %Sub.ref.loc12: %Sub.type.1 = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Negate.ref.loc12: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_2147483647.loc12: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %impl.elem0.loc12_29: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_29: <bound method> = bound_method %int_2147483647.loc12, %impl.elem0.loc12_29 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_29: <specific function> = specific_function %Convert.bound.loc12_29, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_29: init %i32 = call %Convert.specific_fn.loc12_29(%int_2147483647.loc12) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc12_29.1: %i32 = value_of_initializer %int.convert_checked.loc12_29 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc12_29.2: %i32 = converted %int_2147483647.loc12, %.loc12_29.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %int.unegate.loc12: init %i32 = call %Negate.ref.loc12(%.loc12_29.2) [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %int_1.loc12_43: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc12_40.1: %i32 = value_of_initializer %int.unegate.loc12 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %.loc12_40.2: %i32 = converted %int.unegate.loc12, %.loc12_40.1 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %impl.elem0.loc12_43: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_43: <bound method> = bound_method %int_1.loc12_43, %impl.elem0.loc12_43 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_43: <specific function> = specific_function %Convert.bound.loc12_43, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc12_43: init %i32 = call %Convert.specific_fn.loc12_43(%int_1.loc12_43) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_43.1: %i32 = value_of_initializer %int.convert_checked.loc12_43 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_43.2: %i32 = converted %int_1.loc12_43, %.loc12_43.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.usub.loc12: init %i32 = call %Sub.ref.loc12(%.loc12_40.2, %.loc12_43.2) [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %int_1.loc12_47: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc12_44.1: %i32 = value_of_initializer %int.usub.loc12 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc12_44.2: %i32 = converted %int.usub.loc12, %.loc12_44.1 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %impl.elem0.loc12_47: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_47: <bound method> = bound_method %int_1.loc12_47, %impl.elem0.loc12_47 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_47: <specific function> = specific_function %Convert.bound.loc12_47, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc12_47: init %i32 = call %Convert.specific_fn.loc12_47(%int_1.loc12_47) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_47.1: %i32 = value_of_initializer %int.convert_checked.loc12_47 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_47.2: %i32 = converted %int_1.loc12_47, %.loc12_47.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.umod.loc12: init %i32 = call %Mod.ref.loc12(%.loc12_44.2, %.loc12_47.2) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc12_49.1: %i32 = value_of_initializer %int.umod.loc12 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc12_49.2: %i32 = converted %int.umod.loc12, %.loc12_49.1 [template = constants.%int_0]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc12_49.2
// CHECK:STDOUT:   %Mod.ref.loc15: %Mod.type.1 = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %Sub.ref.loc15: %Sub.type.1 = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Negate.ref.loc15_22: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_2147483647.loc15: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %impl.elem0.loc15_29: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_29: <bound method> = bound_method %int_2147483647.loc15, %impl.elem0.loc15_29 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_29: <specific function> = specific_function %Convert.bound.loc15_29, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc15_29: init %i32 = call %Convert.specific_fn.loc15_29(%int_2147483647.loc15) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc15_29.1: %i32 = value_of_initializer %int.convert_checked.loc15_29 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc15_29.2: %i32 = converted %int_2147483647.loc15, %.loc15_29.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %int.unegate.loc15_40: init %i32 = call %Negate.ref.loc15_22(%.loc15_29.2) [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %int_1.loc15_43: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc15_40.1: %i32 = value_of_initializer %int.unegate.loc15_40 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %.loc15_40.2: %i32 = converted %int.unegate.loc15_40, %.loc15_40.1 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %impl.elem0.loc15_43: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_43: <bound method> = bound_method %int_1.loc15_43, %impl.elem0.loc15_43 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_43: <specific function> = specific_function %Convert.bound.loc15_43, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc15_43: init %i32 = call %Convert.specific_fn.loc15_43(%int_1.loc15_43) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_43.1: %i32 = value_of_initializer %int.convert_checked.loc15_43 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_43.2: %i32 = converted %int_1.loc15_43, %.loc15_43.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.usub.loc15: init %i32 = call %Sub.ref.loc15(%.loc15_40.2, %.loc15_43.2) [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %Negate.ref.loc15_47: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc15_54: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_54: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_54: <bound method> = bound_method %int_1.loc15_54, %impl.elem0.loc15_54 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_54: <specific function> = specific_function %Convert.bound.loc15_54, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc15_54: init %i32 = call %Convert.specific_fn.loc15_54(%int_1.loc15_54) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_54.1: %i32 = value_of_initializer %int.convert_checked.loc15_54 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_54.2: %i32 = converted %int_1.loc15_54, %.loc15_54.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.unegate.loc15_55: init %i32 = call %Negate.ref.loc15_47(%.loc15_54.2) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc15_44.1: %i32 = value_of_initializer %int.usub.loc15 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc15_44.2: %i32 = converted %int.usub.loc15, %.loc15_44.1 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc15_55.1: %i32 = value_of_initializer %int.unegate.loc15_55 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc15_55.2: %i32 = converted %int.unegate.loc15_55, %.loc15_55.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.umod.loc15: init %i32 = call %Mod.ref.loc15(%.loc15_44.2, %.loc15_55.2) [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc15_57.1: %i32 = value_of_initializer %int.umod.loc15 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc15_57.2: %i32 = converted %int.umod.loc15, %.loc15_57.1 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %c: %i32 = bind_name c, %.loc15_57.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_div_by_zero.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Mod.type.1: type = fn_type @Mod.1 [template]
// CHECK:STDOUT:   %Mod: %Mod.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.10) [template]
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
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Mod = %Mod.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mod.decl: %Mod.type.1 = fn_decl @Mod.1 [template = constants.%Mod] {
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mod.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.umod";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mod.ref.loc12: %Mod.type.1 = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_0.loc12: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc12_18: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_18: <bound method> = bound_method %int_1, %impl.elem0.loc12_18 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_18: <specific function> = specific_function %Convert.bound.loc12_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_18: init %i32 = call %Convert.specific_fn.loc12_18(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_18.1: %i32 = value_of_initializer %int.convert_checked.loc12_18 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_18.2: %i32 = converted %int_1, %.loc12_18.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc12_21: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_21: <bound method> = bound_method %int_0.loc12, %impl.elem0.loc12_21 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_21: <specific function> = specific_function %Convert.bound.loc12_21, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc12_21: init %i32 = call %Convert.specific_fn.loc12_21(%int_0.loc12) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_21.1: %i32 = value_of_initializer %int.convert_checked.loc12_21 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_21.2: %i32 = converted %int_0.loc12, %.loc12_21.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int.umod.loc12: init %i32 = call %Mod.ref.loc12(%.loc12_18.2, %.loc12_21.2) [template = <error>]
// CHECK:STDOUT:   %.loc12_23.1: %i32 = value_of_initializer %int.umod.loc12 [template = <error>]
// CHECK:STDOUT:   %.loc12_23.2: %i32 = converted %int.umod.loc12, %.loc12_23.1 [template = <error>]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc12_23.2
// CHECK:STDOUT:   %Mod.ref.loc17: %Mod.type.1 = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %int_0.loc17_18: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_0.loc17_21: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc17_18: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc17_18: <bound method> = bound_method %int_0.loc17_18, %impl.elem0.loc17_18 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_18: <specific function> = specific_function %Convert.bound.loc17_18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc17_18: init %i32 = call %Convert.specific_fn.loc17_18(%int_0.loc17_18) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc17_18.1: %i32 = value_of_initializer %int.convert_checked.loc17_18 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc17_18.2: %i32 = converted %int_0.loc17_18, %.loc17_18.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %impl.elem0.loc17_21: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc17_21: <bound method> = bound_method %int_0.loc17_21, %impl.elem0.loc17_21 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_21: <specific function> = specific_function %Convert.bound.loc17_21, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc17_21: init %i32 = call %Convert.specific_fn.loc17_21(%int_0.loc17_21) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc17_21.1: %i32 = value_of_initializer %int.convert_checked.loc17_21 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc17_21.2: %i32 = converted %int_0.loc17_21, %.loc17_21.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int.umod.loc17: init %i32 = call %Mod.ref.loc17(%.loc17_18.2, %.loc17_21.2) [template = <error>]
// CHECK:STDOUT:   %.loc17_23.1: %i32 = value_of_initializer %int.umod.loc17 [template = <error>]
// CHECK:STDOUT:   %.loc17_23.2: %i32 = converted %int.umod.loc17, %.loc17_23.1 [template = <error>]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc17_23.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
