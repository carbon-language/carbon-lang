// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/smod.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/smod.carbon

// --- int_div.carbon

fn Mod(a: i32, b: i32) -> i32 = "int.smod";

var arr: [i32; Mod(5, 3)];
let arr_p: [i32; 2]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Mod(a, b);
}

// --- fail_overflow.carbon

package FailOverflow;

fn Mod(a: i32, b: i32) -> i32 = "int.smod";
fn Sub(a: i32, b: i32) -> i32 = "int.ssub";
fn Negate(a: i32) -> i32 = "int.snegate";

// -0x7FFF_FFFF % -1 is OK.
let a: i32 = Mod(Negate(0x7FFF_FFFF), Negate(1));

// -0x8000_0000 % 1 is OK.
let b: i32 = Mod(Sub(Negate(0x7FFF_FFFF), 1), 1);

// -0x8000_0000 / -1 overflows, so -0x8000_0000 % -1 is disallowed, even though
// its result is representable.
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:14: error: integer overflow in calculation -2147483648 % -1 [CompileTimeIntegerOverflow]
// CHECK:STDERR: let c: i32 = Mod(Sub(Negate(0x7FFF_FFFF), 1), Negate(1));
// CHECK:STDERR:              ^~~~
// CHECK:STDERR:
let c: i32 = Mod(Sub(Negate(0x7FFF_FFFF), 1), Negate(1));

// --- fail_div_by_zero.carbon

package FailDivByZero;

fn Mod(a: i32, b: i32) -> i32 = "int.smod";

// Remainder of division by zero is not defined.

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+4]]:14: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let a: i32 = Mod(1, 0);
// CHECK:STDERR:              ^~~~
// CHECK:STDERR:
let a: i32 = Mod(1, 0);

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+3]]:14: error: division by zero [CompileTimeDivisionByZero]
// CHECK:STDERR: let b: i32 = Mod(0, 0);
// CHECK:STDERR:              ^~~~
let b: i32 = Mod(0, 0);

// CHECK:STDOUT: --- int_div.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Mod.type: type = fn_type @Mod [template]
// CHECK:STDOUT:   %Mod: %Mod.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 3 [template]
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
// CHECK:STDOUT:   %.33: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%.1) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.35: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.34, %Convert.15 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.4(%.1) [template]
// CHECK:STDOUT:   %.38: Core.IntLiteral = int_value 2 [template]
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
// CHECK:STDOUT:     .Mod = %Mod.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mod.decl: %Mod.type = fn_decl @Mod [template = constants.%Mod] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc2_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_11: init type = call constants.%Int(%.loc2_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_11.2: type = value_of_initializer %int.make_type_signed.loc2_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_11.3: type = converted %int.make_type_signed.loc2_11, %.loc2_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_19: init type = call constants.%Int(%.loc2_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.2: type = value_of_initializer %int.make_type_signed.loc2_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.3: type = converted %int.make_type_signed.loc2_19, %.loc2_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_27: init type = call constants.%Int(%.loc2_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.2: type = value_of_initializer %int.make_type_signed.loc2_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.3: type = converted %int.make_type_signed.loc2_27, %.loc2_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%.loc4_11.1) [template = constants.%i32]
// CHECK:STDOUT:   %Mod.ref: %Mod.type = name_ref Mod, %Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %.loc4_20.1: Core.IntLiteral = int_value 5 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_23.1: Core.IntLiteral = int_value 3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_20.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_20.3: <bound method> = bound_method %.loc4_20.1, %.loc4_20.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc4_20.4: <specific function> = specific_function %.loc4_20.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc4_20: init %i32 = call %.loc4_20.4(%.loc4_20.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_20.5: %i32 = value_of_initializer %int.convert_checked.loc4_20 [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_20.6: %i32 = converted %.loc4_20.1, %.loc4_20.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_23.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_23.3: <bound method> = bound_method %.loc4_23.1, %.loc4_23.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc4_23.4: <specific function> = specific_function %.loc4_23.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc4_23: init %i32 = call %.loc4_23.4(%.loc4_23.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_23.5: %i32 = value_of_initializer %int.convert_checked.loc4_23 [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_23.6: %i32 = converted %.loc4_23.1, %.loc4_23.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.smod: init %i32 = call %Mod.ref(%.loc4_20.6, %.loc4_23.6) [template = constants.%.34]
// CHECK:STDOUT:   %.loc4_11.2: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_11.3: type = converted %int.make_type_signed.loc4, %.loc4_11.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_19.1: %Convert.type.6 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc4_19.2: <bound method> = bound_method %int.smod, %.loc4_19.1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc4_19.3: <specific function> = specific_function %.loc4_19.2, @Convert.4(constants.%.1) [template = constants.%.37]
// CHECK:STDOUT:   %.loc4_19.4: %i32 = value_of_initializer %int.smod [template = constants.%.34]
// CHECK:STDOUT:   %.loc4_19.5: %i32 = converted %int.smod, %.loc4_19.4 [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc4_19: init Core.IntLiteral = call %.loc4_19.3(%.loc4_19.5) [template = constants.%.38]
// CHECK:STDOUT:   %.loc4_19.6: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4_19 [template = constants.%.38]
// CHECK:STDOUT:   %.loc4_19.7: Core.IntLiteral = converted %int.smod, %.loc4_19.6 [template = constants.%.38]
// CHECK:STDOUT:   %.loc4_25: type = array_type %.loc4_19.7, %i32 [template = constants.%.39]
// CHECK:STDOUT:   %arr.var: ref %.39 = var arr
// CHECK:STDOUT:   %arr: ref %.39 = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc5_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%.loc5_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_18: Core.IntLiteral = int_value 2 [template = constants.%.38]
// CHECK:STDOUT:   %.loc5_13.2: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_13.3: type = converted %int.make_type_signed.loc5, %.loc5_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_19: type = array_type %.loc5_18, %i32 [template = constants.%.39]
// CHECK:STDOUT:   %.loc5_20: type = ptr_type %.39 [template = constants.%.40]
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
// CHECK:STDOUT: fn @Mod(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.smod";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mod.ref: %Mod.type = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.smod: init %i32 = call %Mod.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19.1: %i32 = value_of_initializer %int.smod
// CHECK:STDOUT:   %.loc8_19.2: %i32 = converted %int.smod, %.loc8_19.1
// CHECK:STDOUT:   return %.loc8_19.2
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
// CHECK:STDOUT: --- fail_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Mod.type: type = fn_type @Mod [template]
// CHECK:STDOUT:   %Mod: %Mod.type = struct_value () [template]
// CHECK:STDOUT:   %Sub.type: type = fn_type @Sub [template]
// CHECK:STDOUT:   %Sub: %Sub.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %.30: %i32 = int_value -2147483647 [template]
// CHECK:STDOUT:   %.31: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.31, %Convert.14 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.35: %i32 = int_value -1 [template]
// CHECK:STDOUT:   %.36: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %.37: %i32 = int_value -2147483648 [template]
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
// CHECK:STDOUT:     .Mod = %Mod.decl
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mod.decl: %Mod.type = fn_decl @Mod [template = constants.%Mod] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_11: init type = call constants.%Int(%.loc4_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.2: type = value_of_initializer %int.make_type_signed.loc4_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.3: type = converted %int.make_type_signed.loc4_11, %.loc4_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_19: init type = call constants.%Int(%.loc4_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.2: type = value_of_initializer %int.make_type_signed.loc4_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.3: type = converted %int.make_type_signed.loc4_19, %.loc4_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_27: init type = call constants.%Int(%.loc4_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.2: type = value_of_initializer %int.make_type_signed.loc4_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.3: type = converted %int.make_type_signed.loc4_27, %.loc4_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Sub.decl: %Sub.type = fn_decl @Sub [template = constants.%Sub] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc5_11: init type = call constants.%Int(%.loc5_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_11.2: type = value_of_initializer %int.make_type_signed.loc5_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_11.3: type = converted %int.make_type_signed.loc5_11, %.loc5_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc5_19: init type = call constants.%Int(%.loc5_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_19.2: type = value_of_initializer %int.make_type_signed.loc5_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_19.3: type = converted %int.make_type_signed.loc5_19, %.loc5_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc5_27: init type = call constants.%Int(%.loc5_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_27.2: type = value_of_initializer %int.make_type_signed.loc5_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_27.3: type = converted %int.make_type_signed.loc5_27, %.loc5_27.2 [template = constants.%i32]
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
// CHECK:STDOUT:     %.loc6_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc6_14: init type = call constants.%Int(%.loc6_14.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_14.2: type = value_of_initializer %int.make_type_signed.loc6_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_14.3: type = converted %int.make_type_signed.loc6_14, %.loc6_14.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc6_22: init type = call constants.%Int(%.loc6_22.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_22.2: type = value_of_initializer %int.make_type_signed.loc6_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_22.3: type = converted %int.make_type_signed.loc6_22, %.loc6_22.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc9: init type = call constants.%Int(%.loc9_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_8.2: type = value_of_initializer %int.make_type_signed.loc9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_8.3: type = converted %int.make_type_signed.loc9, %.loc9_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.3: type = converted %int.make_type_signed.loc12, %.loc12_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc20: init type = call constants.%Int(%.loc20_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_8.2: type = value_of_initializer %int.make_type_signed.loc20 [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_8.3: type = converted %int.make_type_signed.loc20, %.loc20_8.2 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mod(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.smod";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.ssub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mod.ref.loc9: %Mod.type = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %Negate.ref.loc9_18: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc9_25.1: Core.IntLiteral = int_value 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_25.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc9_25.3: <bound method> = bound_method %.loc9_25.1, %.loc9_25.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc9_25.4: <specific function> = specific_function %.loc9_25.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc9_25: init %i32 = call %.loc9_25.4(%.loc9_25.1) [template = constants.%.29]
// CHECK:STDOUT:   %.loc9_25.5: %i32 = value_of_initializer %int.convert_checked.loc9_25 [template = constants.%.29]
// CHECK:STDOUT:   %.loc9_25.6: %i32 = converted %.loc9_25.1, %.loc9_25.5 [template = constants.%.29]
// CHECK:STDOUT:   %int.snegate.loc9_24: init %i32 = call %Negate.ref.loc9_18(%.loc9_25.6) [template = constants.%.30]
// CHECK:STDOUT:   %Negate.ref.loc9_39: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc9_46.1: Core.IntLiteral = int_value 1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc9_46.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc9_46.3: <bound method> = bound_method %.loc9_46.1, %.loc9_46.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc9_46.4: <specific function> = specific_function %.loc9_46.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc9_46: init %i32 = call %.loc9_46.4(%.loc9_46.1) [template = constants.%.34]
// CHECK:STDOUT:   %.loc9_46.5: %i32 = value_of_initializer %int.convert_checked.loc9_46 [template = constants.%.34]
// CHECK:STDOUT:   %.loc9_46.6: %i32 = converted %.loc9_46.1, %.loc9_46.5 [template = constants.%.34]
// CHECK:STDOUT:   %int.snegate.loc9_45: init %i32 = call %Negate.ref.loc9_39(%.loc9_46.6) [template = constants.%.35]
// CHECK:STDOUT:   %.loc9_24.1: %i32 = value_of_initializer %int.snegate.loc9_24 [template = constants.%.30]
// CHECK:STDOUT:   %.loc9_24.2: %i32 = converted %int.snegate.loc9_24, %.loc9_24.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc9_45.1: %i32 = value_of_initializer %int.snegate.loc9_45 [template = constants.%.35]
// CHECK:STDOUT:   %.loc9_45.2: %i32 = converted %int.snegate.loc9_45, %.loc9_45.1 [template = constants.%.35]
// CHECK:STDOUT:   %int.smod.loc9: init %i32 = call %Mod.ref.loc9(%.loc9_24.2, %.loc9_45.2) [template = constants.%.36]
// CHECK:STDOUT:   %.loc9_49.1: %i32 = value_of_initializer %int.smod.loc9 [template = constants.%.36]
// CHECK:STDOUT:   %.loc9_49.2: %i32 = converted %int.smod.loc9, %.loc9_49.1 [template = constants.%.36]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc9_49.2
// CHECK:STDOUT:   %Mod.ref.loc12: %Mod.type = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %Sub.ref.loc12: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Negate.ref.loc12: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc12_29.1: Core.IntLiteral = int_value 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_29.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_29.3: <bound method> = bound_method %.loc12_29.1, %.loc12_29.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc12_29.4: <specific function> = specific_function %.loc12_29.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc12_29: init %i32 = call %.loc12_29.4(%.loc12_29.1) [template = constants.%.29]
// CHECK:STDOUT:   %.loc12_29.5: %i32 = value_of_initializer %int.convert_checked.loc12_29 [template = constants.%.29]
// CHECK:STDOUT:   %.loc12_29.6: %i32 = converted %.loc12_29.1, %.loc12_29.5 [template = constants.%.29]
// CHECK:STDOUT:   %int.snegate.loc12: init %i32 = call %Negate.ref.loc12(%.loc12_29.6) [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_43.1: Core.IntLiteral = int_value 1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc12_28.1: %i32 = value_of_initializer %int.snegate.loc12 [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_28.2: %i32 = converted %int.snegate.loc12, %.loc12_28.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_43.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_43.3: <bound method> = bound_method %.loc12_43.1, %.loc12_43.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc12_43.4: <specific function> = specific_function %.loc12_43.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc12_43: init %i32 = call %.loc12_43.4(%.loc12_43.1) [template = constants.%.34]
// CHECK:STDOUT:   %.loc12_43.5: %i32 = value_of_initializer %int.convert_checked.loc12_43 [template = constants.%.34]
// CHECK:STDOUT:   %.loc12_43.6: %i32 = converted %.loc12_43.1, %.loc12_43.5 [template = constants.%.34]
// CHECK:STDOUT:   %int.ssub.loc12: init %i32 = call %Sub.ref.loc12(%.loc12_28.2, %.loc12_43.6) [template = constants.%.37]
// CHECK:STDOUT:   %.loc12_47.1: Core.IntLiteral = int_value 1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc12_21.1: %i32 = value_of_initializer %int.ssub.loc12 [template = constants.%.37]
// CHECK:STDOUT:   %.loc12_21.2: %i32 = converted %int.ssub.loc12, %.loc12_21.1 [template = constants.%.37]
// CHECK:STDOUT:   %.loc12_47.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_47.3: <bound method> = bound_method %.loc12_47.1, %.loc12_47.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc12_47.4: <specific function> = specific_function %.loc12_47.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc12_47: init %i32 = call %.loc12_47.4(%.loc12_47.1) [template = constants.%.34]
// CHECK:STDOUT:   %.loc12_47.5: %i32 = value_of_initializer %int.convert_checked.loc12_47 [template = constants.%.34]
// CHECK:STDOUT:   %.loc12_47.6: %i32 = converted %.loc12_47.1, %.loc12_47.5 [template = constants.%.34]
// CHECK:STDOUT:   %int.smod.loc12: init %i32 = call %Mod.ref.loc12(%.loc12_21.2, %.loc12_47.6) [template = constants.%.36]
// CHECK:STDOUT:   %.loc12_49.1: %i32 = value_of_initializer %int.smod.loc12 [template = constants.%.36]
// CHECK:STDOUT:   %.loc12_49.2: %i32 = converted %int.smod.loc12, %.loc12_49.1 [template = constants.%.36]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc12_49.2
// CHECK:STDOUT:   %Mod.ref.loc20: %Mod.type = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %Sub.ref.loc20: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Negate.ref.loc20_22: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc20_29.1: Core.IntLiteral = int_value 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %.loc20_29.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc20_29.3: <bound method> = bound_method %.loc20_29.1, %.loc20_29.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc20_29.4: <specific function> = specific_function %.loc20_29.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc20_29: init %i32 = call %.loc20_29.4(%.loc20_29.1) [template = constants.%.29]
// CHECK:STDOUT:   %.loc20_29.5: %i32 = value_of_initializer %int.convert_checked.loc20_29 [template = constants.%.29]
// CHECK:STDOUT:   %.loc20_29.6: %i32 = converted %.loc20_29.1, %.loc20_29.5 [template = constants.%.29]
// CHECK:STDOUT:   %int.snegate.loc20_28: init %i32 = call %Negate.ref.loc20_22(%.loc20_29.6) [template = constants.%.30]
// CHECK:STDOUT:   %.loc20_43.1: Core.IntLiteral = int_value 1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc20_28.1: %i32 = value_of_initializer %int.snegate.loc20_28 [template = constants.%.30]
// CHECK:STDOUT:   %.loc20_28.2: %i32 = converted %int.snegate.loc20_28, %.loc20_28.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc20_43.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc20_43.3: <bound method> = bound_method %.loc20_43.1, %.loc20_43.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc20_43.4: <specific function> = specific_function %.loc20_43.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc20_43: init %i32 = call %.loc20_43.4(%.loc20_43.1) [template = constants.%.34]
// CHECK:STDOUT:   %.loc20_43.5: %i32 = value_of_initializer %int.convert_checked.loc20_43 [template = constants.%.34]
// CHECK:STDOUT:   %.loc20_43.6: %i32 = converted %.loc20_43.1, %.loc20_43.5 [template = constants.%.34]
// CHECK:STDOUT:   %int.ssub.loc20: init %i32 = call %Sub.ref.loc20(%.loc20_28.2, %.loc20_43.6) [template = constants.%.37]
// CHECK:STDOUT:   %Negate.ref.loc20_47: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc20_54.1: Core.IntLiteral = int_value 1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc20_54.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc20_54.3: <bound method> = bound_method %.loc20_54.1, %.loc20_54.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc20_54.4: <specific function> = specific_function %.loc20_54.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc20_54: init %i32 = call %.loc20_54.4(%.loc20_54.1) [template = constants.%.34]
// CHECK:STDOUT:   %.loc20_54.5: %i32 = value_of_initializer %int.convert_checked.loc20_54 [template = constants.%.34]
// CHECK:STDOUT:   %.loc20_54.6: %i32 = converted %.loc20_54.1, %.loc20_54.5 [template = constants.%.34]
// CHECK:STDOUT:   %int.snegate.loc20_53: init %i32 = call %Negate.ref.loc20_47(%.loc20_54.6) [template = constants.%.35]
// CHECK:STDOUT:   %.loc20_21.1: %i32 = value_of_initializer %int.ssub.loc20 [template = constants.%.37]
// CHECK:STDOUT:   %.loc20_21.2: %i32 = converted %int.ssub.loc20, %.loc20_21.1 [template = constants.%.37]
// CHECK:STDOUT:   %.loc20_53.1: %i32 = value_of_initializer %int.snegate.loc20_53 [template = constants.%.35]
// CHECK:STDOUT:   %.loc20_53.2: %i32 = converted %int.snegate.loc20_53, %.loc20_53.1 [template = constants.%.35]
// CHECK:STDOUT:   %int.smod.loc20: init %i32 = call %Mod.ref.loc20(%.loc20_21.2, %.loc20_53.2) [template = constants.%.36]
// CHECK:STDOUT:   %.loc20_57.1: %i32 = value_of_initializer %int.smod.loc20 [template = constants.%.36]
// CHECK:STDOUT:   %.loc20_57.2: %i32 = converted %int.smod.loc20, %.loc20_57.1 [template = constants.%.36]
// CHECK:STDOUT:   %c: %i32 = bind_name c, %.loc20_57.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_div_by_zero.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Mod.type: type = fn_type @Mod [template]
// CHECK:STDOUT:   %Mod: %Mod.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.30: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 0 [template]
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
// CHECK:STDOUT:     .Mod = %Mod.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mod.decl: %Mod.type = fn_decl @Mod [template = constants.%Mod] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_11: init type = call constants.%Int(%.loc4_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.2: type = value_of_initializer %int.make_type_signed.loc4_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.3: type = converted %int.make_type_signed.loc4_11, %.loc4_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_19: init type = call constants.%Int(%.loc4_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.2: type = value_of_initializer %int.make_type_signed.loc4_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.3: type = converted %int.make_type_signed.loc4_19, %.loc4_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_27: init type = call constants.%Int(%.loc4_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.2: type = value_of_initializer %int.make_type_signed.loc4_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.3: type = converted %int.make_type_signed.loc4_27, %.loc4_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.3: type = converted %int.make_type_signed.loc12, %.loc12_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc17: init type = call constants.%Int(%.loc17_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_8.2: type = value_of_initializer %int.make_type_signed.loc17 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_8.3: type = converted %int.make_type_signed.loc17, %.loc17_8.2 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mod(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.smod";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mod.ref.loc12: %Mod.type = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %.loc12_18.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_21.1: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_18.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_18.3: <bound method> = bound_method %.loc12_18.1, %.loc12_18.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc12_18.4: <specific function> = specific_function %.loc12_18.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc12_18: init %i32 = call %.loc12_18.4(%.loc12_18.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_18.5: %i32 = value_of_initializer %int.convert_checked.loc12_18 [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_18.6: %i32 = converted %.loc12_18.1, %.loc12_18.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_21.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_21.3: <bound method> = bound_method %.loc12_21.1, %.loc12_21.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc12_21.4: <specific function> = specific_function %.loc12_21.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc12_21: init %i32 = call %.loc12_21.4(%.loc12_21.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc12_21.5: %i32 = value_of_initializer %int.convert_checked.loc12_21 [template = constants.%.33]
// CHECK:STDOUT:   %.loc12_21.6: %i32 = converted %.loc12_21.1, %.loc12_21.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.smod.loc12: init %i32 = call %Mod.ref.loc12(%.loc12_18.6, %.loc12_21.6) [template = <error>]
// CHECK:STDOUT:   %.loc12_23.1: %i32 = value_of_initializer %int.smod.loc12 [template = <error>]
// CHECK:STDOUT:   %.loc12_23.2: %i32 = converted %int.smod.loc12, %.loc12_23.1 [template = <error>]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc12_23.2
// CHECK:STDOUT:   %Mod.ref.loc17: %Mod.type = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %.loc17_18.1: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc17_21.1: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc17_18.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc17_18.3: <bound method> = bound_method %.loc17_18.1, %.loc17_18.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc17_18.4: <specific function> = specific_function %.loc17_18.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc17_18: init %i32 = call %.loc17_18.4(%.loc17_18.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc17_18.5: %i32 = value_of_initializer %int.convert_checked.loc17_18 [template = constants.%.33]
// CHECK:STDOUT:   %.loc17_18.6: %i32 = converted %.loc17_18.1, %.loc17_18.5 [template = constants.%.33]
// CHECK:STDOUT:   %.loc17_21.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc17_21.3: <bound method> = bound_method %.loc17_21.1, %.loc17_21.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc17_21.4: <specific function> = specific_function %.loc17_21.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc17_21: init %i32 = call %.loc17_21.4(%.loc17_21.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc17_21.5: %i32 = value_of_initializer %int.convert_checked.loc17_21 [template = constants.%.33]
// CHECK:STDOUT:   %.loc17_21.6: %i32 = converted %.loc17_21.1, %.loc17_21.5 [template = constants.%.33]
// CHECK:STDOUT:   %int.smod.loc17: init %i32 = call %Mod.ref.loc17(%.loc17_18.6, %.loc17_21.6) [template = <error>]
// CHECK:STDOUT:   %.loc17_23.1: %i32 = value_of_initializer %int.smod.loc17 [template = <error>]
// CHECK:STDOUT:   %.loc17_23.2: %i32 = converted %int.smod.loc17, %.loc17_23.1 [template = <error>]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc17_23.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
