// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/unegate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/unegate.carbon

// --- int_negate.carbon

fn Negate(a: i32) -> i32 = "int.unegate";

var arr: [i32; Negate(Negate(123))];
let arr_p: [i32; 123]* = &arr;

let n: i32 = Negate(1);

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Negate(a);
}

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.unegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew() -> i32 = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew() -> i32 = "int.unegate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.unegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: i32, b: i32) -> i32 = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: i32, b: i32) -> i32 = "int.unegate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.unegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: i32) -> bool = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: i32) -> bool = "int.unegate";
fn JustRight(a: i32) -> i32 = "int.unegate";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:20: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_few: [i32; TooFew()];
// CHECK:STDERR:                    ^~~~~~~
// CHECK:STDERR:
var too_few: [i32; TooFew()];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:21: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_many: [i32; TooMany(1, 2)];
// CHECK:STDERR:                     ^~~~~~~~
// CHECK:STDERR:
var too_many: [i32; TooMany(1, 2)];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:28: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var bad_return_type: [i32; BadReturnType(1)];
// CHECK:STDERR:                            ^~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_return_type: [i32; BadReturnType(1)];

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:21: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
// CHECK:STDERR: var bad_call: [i32; JustRight(1, 2)];
// CHECK:STDERR:                     ^~~~~~~~~~
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-21]]:1: note: calling function declared here [InCallToEntity]
// CHECK:STDERR: fn JustRight(a: i32) -> i32 = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_call: [i32; JustRight(1, 2)];

fn RuntimeCallTooFew(a: i32) -> i32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return TooFew(a);
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-42]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn TooFew() -> i32 = "int.unegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooFew(a);
}

fn RuntimeCallTooMany(a: i32, b: i32, c: i32) -> i32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 3 arguments passed to function expecting 2 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return TooMany(a, b, c);
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-48]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn TooMany(a: i32, b: i32) -> i32 = "int.unegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooMany(a, b, c);
}

fn RuntimeCallBadReturnType(a: i32, b: i32) -> bool {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+6]]:10: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
  // CHECK:STDERR:   return BadReturnType(a, b);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-54]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn BadReturnType(a: i32) -> bool = "int.unegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return BadReturnType(a, b);
}

// --- overflow.carbon

package Overflow;

fn Negate(a: i32) -> i32 = "int.unegate";
fn Sub(a: i32, b: i32) -> i32 = "int.usub";

// -(-INT_MAX) is INT_MAX.
let a: i32 = Negate(Negate(0x7FFFFFFF));

// -(-INT_MAX - 1) wraps around to INT_MIN.
let b: i32 = Negate(Sub(Negate(0x7FFFFFFF), 1));

// CHECK:STDOUT: --- int_negate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 123 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 123 [template]
// CHECK:STDOUT:   %.30: %i32 = int_value -123 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%.1) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.31: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.29, %Convert.15 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.4(%.1) [template]
// CHECK:STDOUT:   %.34: type = array_type %.2, %i32 [template]
// CHECK:STDOUT:   %.35: type = ptr_type %.34 [template]
// CHECK:STDOUT:   %.36: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.37: <bound method> = bound_method %.36, %Convert.14 [template]
// CHECK:STDOUT:   %.38: <specific function> = specific_function %.37, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.39: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.40: %i32 = int_value -1 [template]
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
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .n = @__global_init.%n
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc2_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_14: init type = call constants.%Int(%.loc2_14.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_14.2: type = value_of_initializer %int.make_type_signed.loc2_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_14.3: type = converted %int.make_type_signed.loc2_14, %.loc2_14.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc2_22: init type = call constants.%Int(%.loc2_22.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_22.2: type = value_of_initializer %int.make_type_signed.loc2_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_22.3: type = converted %int.make_type_signed.loc2_22, %.loc2_22.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%.loc4_11.1) [template = constants.%i32]
// CHECK:STDOUT:   %Negate.ref.loc4_16: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %Negate.ref.loc4_23: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc4_30.1: Core.IntLiteral = int_value 123 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_30.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_30.3: <bound method> = bound_method %.loc4_30.1, %.loc4_30.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc4_30.4: <specific function> = specific_function %.loc4_30.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc4_30: init %i32 = call %.loc4_30.4(%.loc4_30.1) [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_30.5: %i32 = value_of_initializer %int.convert_checked.loc4_30 [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_30.6: %i32 = converted %.loc4_30.1, %.loc4_30.5 [template = constants.%.29]
// CHECK:STDOUT:   %int.unegate.loc4_29: init %i32 = call %Negate.ref.loc4_23(%.loc4_30.6) [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_29.1: %i32 = value_of_initializer %int.unegate.loc4_29 [template = constants.%.30]
// CHECK:STDOUT:   %.loc4_29.2: %i32 = converted %int.unegate.loc4_29, %.loc4_29.1 [template = constants.%.30]
// CHECK:STDOUT:   %int.unegate.loc4_22: init %i32 = call %Negate.ref.loc4_16(%.loc4_29.2) [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_11.2: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_11.3: type = converted %int.make_type_signed.loc4, %.loc4_11.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_22.1: %Convert.type.6 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc4_22.2: <bound method> = bound_method %int.unegate.loc4_22, %.loc4_22.1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc4_22.3: <specific function> = specific_function %.loc4_22.2, @Convert.4(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc4_22.4: %i32 = value_of_initializer %int.unegate.loc4_22 [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_22.5: %i32 = converted %int.unegate.loc4_22, %.loc4_22.4 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc4_22: init Core.IntLiteral = call %.loc4_22.3(%.loc4_22.5) [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_22.6: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4_22 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_22.7: Core.IntLiteral = converted %int.unegate.loc4_22, %.loc4_22.6 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_35: type = array_type %.loc4_22.7, %i32 [template = constants.%.34]
// CHECK:STDOUT:   %arr.var: ref %.34 = var arr
// CHECK:STDOUT:   %arr: ref %.34 = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc5_13.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%.loc5_13.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_18: Core.IntLiteral = int_value 123 [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_13.2: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_13.3: type = converted %int.make_type_signed.loc5, %.loc5_13.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_21: type = array_type %.loc5_18, %i32 [template = constants.%.34]
// CHECK:STDOUT:   %.loc5_22: type = ptr_type %.34 [template = constants.%.35]
// CHECK:STDOUT:   %.loc7_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%.loc7_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.2: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.3: type = converted %int.make_type_signed.loc7, %.loc7_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc9_19: init type = call constants.%Int(%.loc9_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc9_19.2: type = value_of_initializer %int.make_type_signed.loc9_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc9_19.3: type = converted %int.make_type_signed.loc9_19, %.loc9_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc9_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc9_27: init type = call constants.%Int(%.loc9_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc9_27.2: type = value_of_initializer %int.make_type_signed.loc9_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc9_27.3: type = converted %int.make_type_signed.loc9_27, %.loc9_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc9_35.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc9_35: init type = call constants.%Int(%.loc9_35.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc9_35.2: type = value_of_initializer %int.make_type_signed.loc9_35 [template = constants.%i32]
// CHECK:STDOUT:     %.loc9_35.3: type = converted %int.make_type_signed.loc9_35, %.loc9_35.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: %i32) -> %i32 = "int.unegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %int.unegate: init %i32 = call %Negate.ref(%a.ref)
// CHECK:STDOUT:   %.loc10_19.1: %i32 = value_of_initializer %int.unegate
// CHECK:STDOUT:   %.loc10_19.2: %i32 = converted %int.unegate, %.loc10_19.1
// CHECK:STDOUT:   return %.loc10_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %.34 = name_ref arr, file.%arr
// CHECK:STDOUT:   %.loc5: %.35 = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %.35 = bind_name arr_p, %.loc5
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc7_21.1: Core.IntLiteral = int_value 1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc7_21.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc7_21.3: <bound method> = bound_method %.loc7_21.1, %.loc7_21.2 [template = constants.%.37]
// CHECK:STDOUT:   %.loc7_21.4: <specific function> = specific_function %.loc7_21.3, @Convert.2(constants.%.1) [template = constants.%.38]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc7_21.4(%.loc7_21.1) [template = constants.%.39]
// CHECK:STDOUT:   %.loc7_21.5: %i32 = value_of_initializer %int.convert_checked [template = constants.%.39]
// CHECK:STDOUT:   %.loc7_21.6: %i32 = converted %.loc7_21.1, %.loc7_21.5 [template = constants.%.39]
// CHECK:STDOUT:   %int.unegate: init %i32 = call %Negate.ref(%.loc7_21.6) [template = constants.%.40]
// CHECK:STDOUT:   %.loc7_23.1: %i32 = value_of_initializer %int.unegate [template = constants.%.40]
// CHECK:STDOUT:   %.loc7_23.2: %i32 = converted %int.unegate, %.loc7_23.1 [template = constants.%.40]
// CHECK:STDOUT:   %n: %i32 = bind_name n, %.loc7_23.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %TooFew.type: type = fn_type @TooFew [template]
// CHECK:STDOUT:   %TooFew: %TooFew.type = struct_value () [template]
// CHECK:STDOUT:   %TooMany.type: type = fn_type @TooMany [template]
// CHECK:STDOUT:   %TooMany: %TooMany.type = struct_value () [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %BadReturnType.type: type = fn_type @BadReturnType [template]
// CHECK:STDOUT:   %BadReturnType: %BadReturnType.type = struct_value () [template]
// CHECK:STDOUT:   %JustRight.type: type = fn_type @JustRight [template]
// CHECK:STDOUT:   %JustRight: %JustRight.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.30: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %RuntimeCallTooFew.type: type = fn_type @RuntimeCallTooFew [template]
// CHECK:STDOUT:   %RuntimeCallTooFew: %RuntimeCallTooFew.type = struct_value () [template]
// CHECK:STDOUT:   %RuntimeCallTooMany.type: type = fn_type @RuntimeCallTooMany [template]
// CHECK:STDOUT:   %RuntimeCallTooMany: %RuntimeCallTooMany.type = struct_value () [template]
// CHECK:STDOUT:   %RuntimeCallBadReturnType.type: type = fn_type @RuntimeCallBadReturnType [template]
// CHECK:STDOUT:   %RuntimeCallBadReturnType: %RuntimeCallBadReturnType.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .TooFew = %TooFew.decl
// CHECK:STDOUT:     .TooMany = %TooMany.decl
// CHECK:STDOUT:     .BadReturnType = %BadReturnType.decl
// CHECK:STDOUT:     .JustRight = %JustRight.decl
// CHECK:STDOUT:     .too_few = %too_few
// CHECK:STDOUT:     .too_many = %too_many
// CHECK:STDOUT:     .bad_return_type = %bad_return_type
// CHECK:STDOUT:     .bad_call = %bad_call
// CHECK:STDOUT:     .RuntimeCallTooFew = %RuntimeCallTooFew.decl
// CHECK:STDOUT:     .RuntimeCallTooMany = %RuntimeCallTooMany.decl
// CHECK:STDOUT:     .RuntimeCallBadReturnType = %RuntimeCallBadReturnType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %TooFew.decl: %TooFew.type = fn_decl @TooFew [template = constants.%TooFew] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_16.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc8_16.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_16.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_16.3: type = converted %int.make_type_signed, %.loc8_16.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany.decl: %TooMany.type = fn_decl @TooMany [template = constants.%TooMany] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_15.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc13_15: init type = call constants.%Int(%.loc13_15.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_15.2: type = value_of_initializer %int.make_type_signed.loc13_15 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_15.3: type = converted %int.make_type_signed.loc13_15, %.loc13_15.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_23.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc13_23: init type = call constants.%Int(%.loc13_23.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_23.2: type = value_of_initializer %int.make_type_signed.loc13_23 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_23.3: type = converted %int.make_type_signed.loc13_23, %.loc13_23.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_31.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc13_31: init type = call constants.%Int(%.loc13_31.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_31.2: type = value_of_initializer %int.make_type_signed.loc13_31 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_31.3: type = converted %int.make_type_signed.loc13_31, %.loc13_31.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BadReturnType.decl: %BadReturnType.type = fn_decl @BadReturnType [template = constants.%BadReturnType] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_21.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc18_21.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_21.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_21.3: type = converted %int.make_type_signed, %.loc18_21.2 [template = constants.%i32]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc18_29.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc18_29.2: type = converted %bool.make_type, %.loc18_29.1 [template = bool]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %JustRight.decl: %JustRight.type = fn_decl @JustRight [template = constants.%JustRight] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc19_17.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc19_17: init type = call constants.%Int(%.loc19_17.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_17.2: type = value_of_initializer %int.make_type_signed.loc19_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_17.3: type = converted %int.make_type_signed.loc19_17, %.loc19_17.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_25.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc19_25: init type = call constants.%Int(%.loc19_25.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_25.2: type = value_of_initializer %int.make_type_signed.loc19_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_25.3: type = converted %int.make_type_signed.loc19_25, %.loc19_25.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc25_15.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc25: init type = call constants.%Int(%.loc25_15.1) [template = constants.%i32]
// CHECK:STDOUT:   %TooFew.ref: %TooFew.type = name_ref TooFew, %TooFew.decl [template = constants.%TooFew]
// CHECK:STDOUT:   %TooFew.call: init %i32 = call %TooFew.ref()
// CHECK:STDOUT:   %.loc25_15.2: type = value_of_initializer %int.make_type_signed.loc25 [template = constants.%i32]
// CHECK:STDOUT:   %.loc25_15.3: type = converted %int.make_type_signed.loc25, %.loc25_15.2 [template = constants.%i32]
// CHECK:STDOUT:   %too_few.var: ref <error> = var too_few
// CHECK:STDOUT:   %too_few: ref <error> = bind_name too_few, %too_few.var
// CHECK:STDOUT:   %.loc30_16.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc30: init type = call constants.%Int(%.loc30_16.1) [template = constants.%i32]
// CHECK:STDOUT:   %TooMany.ref: %TooMany.type = name_ref TooMany, %TooMany.decl [template = constants.%TooMany]
// CHECK:STDOUT:   %.loc30_29.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc30_32.1: Core.IntLiteral = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc30_29.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc30_29.3: <bound method> = bound_method %.loc30_29.1, %.loc30_29.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc30_29.4: <specific function> = specific_function %.loc30_29.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc30_29: init %i32 = call %.loc30_29.4(%.loc30_29.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc30_29.5: %i32 = value_of_initializer %int.convert_checked.loc30_29 [template = constants.%.30]
// CHECK:STDOUT:   %.loc30_29.6: %i32 = converted %.loc30_29.1, %.loc30_29.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc30_32.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc30_32.3: <bound method> = bound_method %.loc30_32.1, %.loc30_32.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc30_32.4: <specific function> = specific_function %.loc30_32.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc30_32: init %i32 = call %.loc30_32.4(%.loc30_32.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc30_32.5: %i32 = value_of_initializer %int.convert_checked.loc30_32 [template = constants.%.33]
// CHECK:STDOUT:   %.loc30_32.6: %i32 = converted %.loc30_32.1, %.loc30_32.5 [template = constants.%.33]
// CHECK:STDOUT:   %TooMany.call: init %i32 = call %TooMany.ref(%.loc30_29.6, %.loc30_32.6)
// CHECK:STDOUT:   %.loc30_16.2: type = value_of_initializer %int.make_type_signed.loc30 [template = constants.%i32]
// CHECK:STDOUT:   %.loc30_16.3: type = converted %int.make_type_signed.loc30, %.loc30_16.2 [template = constants.%i32]
// CHECK:STDOUT:   %too_many.var: ref <error> = var too_many
// CHECK:STDOUT:   %too_many: ref <error> = bind_name too_many, %too_many.var
// CHECK:STDOUT:   %.loc35_23.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc35: init type = call constants.%Int(%.loc35_23.1) [template = constants.%i32]
// CHECK:STDOUT:   %BadReturnType.ref: %BadReturnType.type = name_ref BadReturnType, %BadReturnType.decl [template = constants.%BadReturnType]
// CHECK:STDOUT:   %.loc35_42.1: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc35_42.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc35_42.3: <bound method> = bound_method %.loc35_42.1, %.loc35_42.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc35_42.4: <specific function> = specific_function %.loc35_42.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc35: init %i32 = call %.loc35_42.4(%.loc35_42.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc35_42.5: %i32 = value_of_initializer %int.convert_checked.loc35 [template = constants.%.30]
// CHECK:STDOUT:   %.loc35_42.6: %i32 = converted %.loc35_42.1, %.loc35_42.5 [template = constants.%.30]
// CHECK:STDOUT:   %BadReturnType.call: init bool = call %BadReturnType.ref(%.loc35_42.6)
// CHECK:STDOUT:   %.loc35_23.2: type = value_of_initializer %int.make_type_signed.loc35 [template = constants.%i32]
// CHECK:STDOUT:   %.loc35_23.3: type = converted %int.make_type_signed.loc35, %.loc35_23.2 [template = constants.%i32]
// CHECK:STDOUT:   %bad_return_type.var: ref <error> = var bad_return_type
// CHECK:STDOUT:   %bad_return_type: ref <error> = bind_name bad_return_type, %bad_return_type.var
// CHECK:STDOUT:   %.loc44_16.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc44: init type = call constants.%Int(%.loc44_16.1) [template = constants.%i32]
// CHECK:STDOUT:   %JustRight.ref: %JustRight.type = name_ref JustRight, %JustRight.decl [template = constants.%JustRight]
// CHECK:STDOUT:   %.loc44_31: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc44_34: Core.IntLiteral = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc44_16.2: type = value_of_initializer %int.make_type_signed.loc44 [template = constants.%i32]
// CHECK:STDOUT:   %.loc44_16.3: type = converted %int.make_type_signed.loc44, %.loc44_16.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc44_36: type = array_type <error>, %i32 [template = <error>]
// CHECK:STDOUT:   %bad_call.var: ref <error> = var bad_call
// CHECK:STDOUT:   %bad_call: ref <error> = bind_name bad_call, %bad_call.var
// CHECK:STDOUT:   %RuntimeCallTooFew.decl: %RuntimeCallTooFew.type = fn_decl @RuntimeCallTooFew [template = constants.%RuntimeCallTooFew] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc46_25.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc46_25: init type = call constants.%Int(%.loc46_25.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc46_25.2: type = value_of_initializer %int.make_type_signed.loc46_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc46_25.3: type = converted %int.make_type_signed.loc46_25, %.loc46_25.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc46_33.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc46_33: init type = call constants.%Int(%.loc46_33.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc46_33.2: type = value_of_initializer %int.make_type_signed.loc46_33 [template = constants.%i32]
// CHECK:STDOUT:     %.loc46_33.3: type = converted %int.make_type_signed.loc46_33, %.loc46_33.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooMany.decl: %RuntimeCallTooMany.type = fn_decl @RuntimeCallTooMany [template = constants.%RuntimeCallTooMany] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %c.patt: %i32 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %i32 = value_param_pattern %c.patt, runtime_param2
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param3
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc57_26.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc57_26: init type = call constants.%Int(%.loc57_26.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_26.2: type = value_of_initializer %int.make_type_signed.loc57_26 [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_26.3: type = converted %int.make_type_signed.loc57_26, %.loc57_26.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_34.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc57_34: init type = call constants.%Int(%.loc57_34.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_34.2: type = value_of_initializer %int.make_type_signed.loc57_34 [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_34.3: type = converted %int.make_type_signed.loc57_34, %.loc57_34.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_42.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc57_42: init type = call constants.%Int(%.loc57_42.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_42.2: type = value_of_initializer %int.make_type_signed.loc57_42 [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_42.3: type = converted %int.make_type_signed.loc57_42, %.loc57_42.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_50.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc57_50: init type = call constants.%Int(%.loc57_50.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_50.2: type = value_of_initializer %int.make_type_signed.loc57_50 [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_50.3: type = converted %int.make_type_signed.loc57_50, %.loc57_50.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: %i32 = value_param runtime_param2
// CHECK:STDOUT:     %c: %i32 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param3
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallBadReturnType.decl: %RuntimeCallBadReturnType.type = fn_decl @RuntimeCallBadReturnType [template = constants.%RuntimeCallBadReturnType] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc68_32.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc68_32: init type = call constants.%Int(%.loc68_32.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc68_32.2: type = value_of_initializer %int.make_type_signed.loc68_32 [template = constants.%i32]
// CHECK:STDOUT:     %.loc68_32.3: type = converted %int.make_type_signed.loc68_32, %.loc68_32.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc68_40.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc68_40: init type = call constants.%Int(%.loc68_40.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc68_40.2: type = value_of_initializer %int.make_type_signed.loc68_40 [template = constants.%i32]
// CHECK:STDOUT:     %.loc68_40.3: type = converted %int.make_type_signed.loc68_40, %.loc68_40.2 [template = constants.%i32]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc68_48.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc68_48.2: type = converted %bool.make_type, %.loc68_48.1 [template = bool]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew() -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BadReturnType(%a.param_patt: %i32) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @JustRight(%a.param_patt: %i32) -> %i32 = "int.unegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooFew(%a.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooFew.ref: %TooFew.type = name_ref TooFew, file.%TooFew.decl [template = constants.%TooFew]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooMany(%a.param_patt: %i32, %b.param_patt: %i32, %c.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooMany.ref: %TooMany.type = name_ref TooMany, file.%TooMany.decl [template = constants.%TooMany]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: %i32 = name_ref c, %c
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallBadReturnType(%a.param_patt: %i32, %b.param_patt: %i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BadReturnType.ref: %BadReturnType.type = name_ref BadReturnType, file.%BadReturnType.decl [template = constants.%BadReturnType]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %Sub.type: type = fn_type @Sub [template]
// CHECK:STDOUT:   %Sub: %Sub.type = struct_value () [template]
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
// CHECK:STDOUT:   %.35: %i32 = int_value -2147483648 [template]
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
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_14: init type = call constants.%Int(%.loc4_14.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_14.2: type = value_of_initializer %int.make_type_signed.loc4_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_14.3: type = converted %int.make_type_signed.loc4_14, %.loc4_14.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_22.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc4_22: init type = call constants.%Int(%.loc4_22.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_22.2: type = value_of_initializer %int.make_type_signed.loc4_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_22.3: type = converted %int.make_type_signed.loc4_22, %.loc4_22.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
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
// CHECK:STDOUT:   %.loc8_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%.loc8_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8.2: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8.3: type = converted %int.make_type_signed.loc8, %.loc8_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%.loc11_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_8.2: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_8.3: type = converted %int.make_type_signed.loc11, %.loc11_8.2 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: %i32) -> %i32 = "int.unegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.usub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref.loc8_14: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %Negate.ref.loc8_21: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc8_28.1: Core.IntLiteral = int_value 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_28.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc8_28.3: <bound method> = bound_method %.loc8_28.1, %.loc8_28.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc8_28.4: <specific function> = specific_function %.loc8_28.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %i32 = call %.loc8_28.4(%.loc8_28.1) [template = constants.%.29]
// CHECK:STDOUT:   %.loc8_28.5: %i32 = value_of_initializer %int.convert_checked.loc8 [template = constants.%.29]
// CHECK:STDOUT:   %.loc8_28.6: %i32 = converted %.loc8_28.1, %.loc8_28.5 [template = constants.%.29]
// CHECK:STDOUT:   %int.unegate.loc8_27: init %i32 = call %Negate.ref.loc8_21(%.loc8_28.6) [template = constants.%.30]
// CHECK:STDOUT:   %.loc8_27.1: %i32 = value_of_initializer %int.unegate.loc8_27 [template = constants.%.30]
// CHECK:STDOUT:   %.loc8_27.2: %i32 = converted %int.unegate.loc8_27, %.loc8_27.1 [template = constants.%.30]
// CHECK:STDOUT:   %int.unegate.loc8_20: init %i32 = call %Negate.ref.loc8_14(%.loc8_27.2) [template = constants.%.29]
// CHECK:STDOUT:   %.loc8_40.1: %i32 = value_of_initializer %int.unegate.loc8_20 [template = constants.%.29]
// CHECK:STDOUT:   %.loc8_40.2: %i32 = converted %int.unegate.loc8_20, %.loc8_40.1 [template = constants.%.29]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc8_40.2
// CHECK:STDOUT:   %Negate.ref.loc11_14: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %Sub.ref: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Negate.ref.loc11_25: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc11_32.1: Core.IntLiteral = int_value 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_32.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_32.3: <bound method> = bound_method %.loc11_32.1, %.loc11_32.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc11_32.4: <specific function> = specific_function %.loc11_32.3, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc11_32: init %i32 = call %.loc11_32.4(%.loc11_32.1) [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_32.5: %i32 = value_of_initializer %int.convert_checked.loc11_32 [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_32.6: %i32 = converted %.loc11_32.1, %.loc11_32.5 [template = constants.%.29]
// CHECK:STDOUT:   %int.unegate.loc11_31: init %i32 = call %Negate.ref.loc11_25(%.loc11_32.6) [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_45.1: Core.IntLiteral = int_value 1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_31.1: %i32 = value_of_initializer %int.unegate.loc11_31 [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_31.2: %i32 = converted %int.unegate.loc11_31, %.loc11_31.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_45.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.3: <bound method> = bound_method %.loc11_45.1, %.loc11_45.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_45.4: <specific function> = specific_function %.loc11_45.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc11_45: init %i32 = call %.loc11_45.4(%.loc11_45.1) [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_45.5: %i32 = value_of_initializer %int.convert_checked.loc11_45 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_45.6: %i32 = converted %.loc11_45.1, %.loc11_45.5 [template = constants.%.34]
// CHECK:STDOUT:   %int.usub: init %i32 = call %Sub.ref(%.loc11_31.2, %.loc11_45.6) [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_24.1: %i32 = value_of_initializer %int.usub [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_24.2: %i32 = converted %int.usub, %.loc11_24.1 [template = constants.%.35]
// CHECK:STDOUT:   %int.unegate.loc11_20: init %i32 = call %Negate.ref.loc11_14(%.loc11_24.2) [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_48.1: %i32 = value_of_initializer %int.unegate.loc11_20 [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_48.2: %i32 = converted %int.unegate.loc11_20, %.loc11_48.1 [template = constants.%.35]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc11_48.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
