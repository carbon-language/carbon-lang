// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/snegate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/snegate.carbon

// --- int_negate.carbon

fn Negate(a: i32) -> i32 = "int.snegate";

var arr: [i32; Negate(Negate(123))];
let arr_p: [i32; 123]* = &arr;

let n: i32 = Negate(1);

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Negate(a);
}

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.snegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew() -> i32 = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew() -> i32 = "int.snegate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.snegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: i32, b: i32) -> i32 = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: i32, b: i32) -> i32 = "int.snegate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.snegate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: i32) -> bool = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: i32) -> bool = "int.snegate";
fn JustRight(a: i32) -> i32 = "int.snegate";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:20: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_few: [i32; TooFew()];
// CHECK:STDERR:                    ^~~~~~~~
// CHECK:STDERR:
var too_few: [i32; TooFew()];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:21: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var too_many: [i32; TooMany(1, 2)];
// CHECK:STDERR:                     ^~~~~~~~~~~~~
// CHECK:STDERR:
var too_many: [i32; TooMany(1, 2)];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:28: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var bad_return_type: [i32; BadReturnType(1)];
// CHECK:STDERR:                            ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_return_type: [i32; BadReturnType(1)];

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:21: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
// CHECK:STDERR: var bad_call: [i32; JustRight(1, 2)];
// CHECK:STDERR:                     ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-21]]:1: note: calling function declared here [InCallToEntity]
// CHECK:STDERR: fn JustRight(a: i32) -> i32 = "int.snegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_call: [i32; JustRight(1, 2)];

fn RuntimeCallTooFew(a: i32) -> i32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return TooFew(a);
  // CHECK:STDERR:          ^~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-42]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn TooFew() -> i32 = "int.snegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooFew(a);
}

fn RuntimeCallTooMany(a: i32, b: i32, c: i32) -> i32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 3 arguments passed to function expecting 2 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return TooMany(a, b, c);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-48]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn TooMany(a: i32, b: i32) -> i32 = "int.snegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooMany(a, b, c);
}

fn RuntimeCallBadReturnType(a: i32, b: i32) -> bool {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
  // CHECK:STDERR:   return BadReturnType(a, b);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-54]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn BadReturnType(a: i32) -> bool = "int.snegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return BadReturnType(a, b);
}

// --- fail_overflow.carbon

package FailOverflow;

fn Negate(a: i32) -> i32 = "int.snegate";
fn Sub(a: i32, b: i32) -> i32 = "int.ssub";

// -(-INT_MAX) is INT_MAX.
let a: i32 = Negate(Negate(0x7FFFFFFF));

// -(-INT_MAX - 1) is too large for i32.
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+3]]:14: error: integer overflow in negation of -2147483648 [CompileTimeIntegerNegateOverflow]
// CHECK:STDERR: let b: i32 = Negate(Sub(Negate(0x7FFFFFFF), 1));
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
let b: i32 = Negate(Sub(Negate(0x7FFFFFFF), 1));

// CHECK:STDOUT: --- int_negate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %int_123.1: Core.IntLiteral = int_value 123 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_123.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_123.2: %i32 = int_value 123 [template]
// CHECK:STDOUT:   %int_-123: %i32 = int_value -123 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %interface.10: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_123.2, %Convert.15 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.4(%int_32) [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_123.1, %i32 [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_-1: %i32 = int_value -1 [template]
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
// CHECK:STDOUT:     %int_32.loc2_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_14: init type = call constants.%Int(%int_32.loc2_14) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_14.1: type = value_of_initializer %int.make_type_signed.loc2_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_14.2: type = converted %int.make_type_signed.loc2_14, %.loc2_14.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc2_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_22: init type = call constants.%Int(%int_32.loc2_22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_22.1: type = value_of_initializer %int.make_type_signed.loc2_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_22.2: type = converted %int.make_type_signed.loc2_22, %.loc2_22.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc4: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%int_32.loc4) [template = constants.%i32]
// CHECK:STDOUT:   %Negate.ref.loc4_16: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %Negate.ref.loc4_23: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_123.loc4: Core.IntLiteral = int_value 123 [template = constants.%int_123.1]
// CHECK:STDOUT:   %impl.elem0.loc4_30: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc4_30: <bound method> = bound_method %int_123.loc4, %impl.elem0.loc4_30 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc4_30: <specific function> = specific_function %Convert.bound.loc4_30, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc4_30: init %i32 = call %Convert.specific_fn.loc4_30(%int_123.loc4) [template = constants.%int_123.2]
// CHECK:STDOUT:   %.loc4_30.1: %i32 = value_of_initializer %int.convert_checked.loc4_30 [template = constants.%int_123.2]
// CHECK:STDOUT:   %.loc4_30.2: %i32 = converted %int_123.loc4, %.loc4_30.1 [template = constants.%int_123.2]
// CHECK:STDOUT:   %int.snegate.loc4_33: init %i32 = call %Negate.ref.loc4_23(%.loc4_30.2) [template = constants.%int_-123]
// CHECK:STDOUT:   %.loc4_33.1: %i32 = value_of_initializer %int.snegate.loc4_33 [template = constants.%int_-123]
// CHECK:STDOUT:   %.loc4_33.2: %i32 = converted %int.snegate.loc4_33, %.loc4_33.1 [template = constants.%int_-123]
// CHECK:STDOUT:   %int.snegate.loc4_34: init %i32 = call %Negate.ref.loc4_16(%.loc4_33.2) [template = constants.%int_123.2]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_signed.loc4, %.loc4_11.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc4_34: %Convert.type.6 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %Convert.bound.loc4_34: <bound method> = bound_method %int.snegate.loc4_34, %impl.elem0.loc4_34 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc4_34: <specific function> = specific_function %Convert.bound.loc4_34, @Convert.4(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %.loc4_34.1: %i32 = value_of_initializer %int.snegate.loc4_34 [template = constants.%int_123.2]
// CHECK:STDOUT:   %.loc4_34.2: %i32 = converted %int.snegate.loc4_34, %.loc4_34.1 [template = constants.%int_123.2]
// CHECK:STDOUT:   %int.convert_checked.loc4_34: init Core.IntLiteral = call %Convert.specific_fn.loc4_34(%.loc4_34.2) [template = constants.%int_123.1]
// CHECK:STDOUT:   %.loc4_34.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4_34 [template = constants.%int_123.1]
// CHECK:STDOUT:   %.loc4_34.4: Core.IntLiteral = converted %int.snegate.loc4_34, %.loc4_34.3 [template = constants.%int_123.1]
// CHECK:STDOUT:   %array_type.loc4: type = array_type %.loc4_34.4, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %arr.var: ref %array_type = var arr
// CHECK:STDOUT:   %arr: ref %array_type = bind_name arr, %arr.var
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%int_32.loc5) [template = constants.%i32]
// CHECK:STDOUT:   %int_123.loc5: Core.IntLiteral = int_value 123 [template = constants.%int_123.1]
// CHECK:STDOUT:   %.loc5_13.1: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_13.2: type = converted %int.make_type_signed.loc5, %.loc5_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type.loc5: type = array_type %int_123.loc5, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template = constants.%ptr]
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%int_32.loc7) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.1: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.2: type = converted %int.make_type_signed.loc7, %.loc7_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc9_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc9_19: init type = call constants.%Int(%int_32.loc9_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc9_19.1: type = value_of_initializer %int.make_type_signed.loc9_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc9_19.2: type = converted %int.make_type_signed.loc9_19, %.loc9_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc9_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc9_27: init type = call constants.%Int(%int_32.loc9_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc9_27.1: type = value_of_initializer %int.make_type_signed.loc9_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc9_27.2: type = converted %int.make_type_signed.loc9_27, %.loc9_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc9_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc9_35: init type = call constants.%Int(%int_32.loc9_35) [template = constants.%i32]
// CHECK:STDOUT:     %.loc9_35.1: type = value_of_initializer %int.make_type_signed.loc9_35 [template = constants.%i32]
// CHECK:STDOUT:     %.loc9_35.2: type = converted %int.make_type_signed.loc9_35, %.loc9_35.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %int.snegate: init %i32 = call %Negate.ref(%a.ref)
// CHECK:STDOUT:   %.loc10_19.1: %i32 = value_of_initializer %int.snegate
// CHECK:STDOUT:   %.loc10_19.2: %i32 = converted %int.snegate, %.loc10_19.1
// CHECK:STDOUT:   return %.loc10_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %array_type = name_ref arr, file.%arr
// CHECK:STDOUT:   %addr: %ptr = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %ptr = bind_name arr_p, %addr
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1, %impl.elem0 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc7_21.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc7_21.2: %i32 = converted %int_1, %.loc7_21.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.snegate: init %i32 = call %Negate.ref(%.loc7_21.2) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc7_23.1: %i32 = value_of_initializer %int.snegate [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc7_23.2: %i32 = converted %int.snegate, %.loc7_23.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %n: %i32 = bind_name n, %.loc7_23.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
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
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
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
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_16.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_16.2: type = converted %int.make_type_signed, %.loc8_16.1 [template = constants.%i32]
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
// CHECK:STDOUT:     %int_32.loc13_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc13_15: init type = call constants.%Int(%int_32.loc13_15) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_15.1: type = value_of_initializer %int.make_type_signed.loc13_15 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_15.2: type = converted %int.make_type_signed.loc13_15, %.loc13_15.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc13_23: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc13_23: init type = call constants.%Int(%int_32.loc13_23) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_23.1: type = value_of_initializer %int.make_type_signed.loc13_23 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_23.2: type = converted %int.make_type_signed.loc13_23, %.loc13_23.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc13_31: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc13_31: init type = call constants.%Int(%int_32.loc13_31) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_31.1: type = value_of_initializer %int.make_type_signed.loc13_31 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_31.2: type = converted %int.make_type_signed.loc13_31, %.loc13_31.1 [template = constants.%i32]
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
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_21.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_21.2: type = converted %int.make_type_signed, %.loc18_21.1 [template = constants.%i32]
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
// CHECK:STDOUT:     %int_32.loc19_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc19_17: init type = call constants.%Int(%int_32.loc19_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_17.1: type = value_of_initializer %int.make_type_signed.loc19_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_17.2: type = converted %int.make_type_signed.loc19_17, %.loc19_17.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc19_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc19_25: init type = call constants.%Int(%int_32.loc19_25) [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_25.1: type = value_of_initializer %int.make_type_signed.loc19_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_25.2: type = converted %int.make_type_signed.loc19_25, %.loc19_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc25: init type = call constants.%Int(%int_32.loc25) [template = constants.%i32]
// CHECK:STDOUT:   %TooFew.ref: %TooFew.type = name_ref TooFew, %TooFew.decl [template = constants.%TooFew]
// CHECK:STDOUT:   %TooFew.call: init %i32 = call %TooFew.ref()
// CHECK:STDOUT:   %.loc25_15.1: type = value_of_initializer %int.make_type_signed.loc25 [template = constants.%i32]
// CHECK:STDOUT:   %.loc25_15.2: type = converted %int.make_type_signed.loc25, %.loc25_15.1 [template = constants.%i32]
// CHECK:STDOUT:   %too_few.var: ref <error> = var too_few
// CHECK:STDOUT:   %too_few: ref <error> = bind_name too_few, %too_few.var
// CHECK:STDOUT:   %int_32.loc30: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc30: init type = call constants.%Int(%int_32.loc30) [template = constants.%i32]
// CHECK:STDOUT:   %TooMany.ref: %TooMany.type = name_ref TooMany, %TooMany.decl [template = constants.%TooMany]
// CHECK:STDOUT:   %int_1.loc30: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc30: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc30_29: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc30_29: <bound method> = bound_method %int_1.loc30, %impl.elem0.loc30_29 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc30_29: <specific function> = specific_function %Convert.bound.loc30_29, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc30_29: init %i32 = call %Convert.specific_fn.loc30_29(%int_1.loc30) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc30_29.1: %i32 = value_of_initializer %int.convert_checked.loc30_29 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc30_29.2: %i32 = converted %int_1.loc30, %.loc30_29.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc30_32: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc30_32: <bound method> = bound_method %int_2.loc30, %impl.elem0.loc30_32 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc30_32: <specific function> = specific_function %Convert.bound.loc30_32, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc30_32: init %i32 = call %Convert.specific_fn.loc30_32(%int_2.loc30) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc30_32.1: %i32 = value_of_initializer %int.convert_checked.loc30_32 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc30_32.2: %i32 = converted %int_2.loc30, %.loc30_32.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %TooMany.call: init %i32 = call %TooMany.ref(%.loc30_29.2, %.loc30_32.2)
// CHECK:STDOUT:   %.loc30_16.1: type = value_of_initializer %int.make_type_signed.loc30 [template = constants.%i32]
// CHECK:STDOUT:   %.loc30_16.2: type = converted %int.make_type_signed.loc30, %.loc30_16.1 [template = constants.%i32]
// CHECK:STDOUT:   %too_many.var: ref <error> = var too_many
// CHECK:STDOUT:   %too_many: ref <error> = bind_name too_many, %too_many.var
// CHECK:STDOUT:   %int_32.loc35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc35: init type = call constants.%Int(%int_32.loc35) [template = constants.%i32]
// CHECK:STDOUT:   %BadReturnType.ref: %BadReturnType.type = name_ref BadReturnType, %BadReturnType.decl [template = constants.%BadReturnType]
// CHECK:STDOUT:   %int_1.loc35: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc35: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc35: <bound method> = bound_method %int_1.loc35, %impl.elem0.loc35 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc35: <specific function> = specific_function %Convert.bound.loc35, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc35: init %i32 = call %Convert.specific_fn.loc35(%int_1.loc35) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc35_42.1: %i32 = value_of_initializer %int.convert_checked.loc35 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc35_42.2: %i32 = converted %int_1.loc35, %.loc35_42.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %BadReturnType.call: init bool = call %BadReturnType.ref(%.loc35_42.2)
// CHECK:STDOUT:   %.loc35_23.1: type = value_of_initializer %int.make_type_signed.loc35 [template = constants.%i32]
// CHECK:STDOUT:   %.loc35_23.2: type = converted %int.make_type_signed.loc35, %.loc35_23.1 [template = constants.%i32]
// CHECK:STDOUT:   %bad_return_type.var: ref <error> = var bad_return_type
// CHECK:STDOUT:   %bad_return_type: ref <error> = bind_name bad_return_type, %bad_return_type.var
// CHECK:STDOUT:   %int_32.loc44: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc44: init type = call constants.%Int(%int_32.loc44) [template = constants.%i32]
// CHECK:STDOUT:   %JustRight.ref: %JustRight.type = name_ref JustRight, %JustRight.decl [template = constants.%JustRight]
// CHECK:STDOUT:   %int_1.loc44: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc44: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc44_16.1: type = value_of_initializer %int.make_type_signed.loc44 [template = constants.%i32]
// CHECK:STDOUT:   %.loc44_16.2: type = converted %int.make_type_signed.loc44, %.loc44_16.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type: type = array_type <error>, %i32 [template = <error>]
// CHECK:STDOUT:   %bad_call.var: ref <error> = var bad_call
// CHECK:STDOUT:   %bad_call: ref <error> = bind_name bad_call, %bad_call.var
// CHECK:STDOUT:   %RuntimeCallTooFew.decl: %RuntimeCallTooFew.type = fn_decl @RuntimeCallTooFew [template = constants.%RuntimeCallTooFew] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc46_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc46_25: init type = call constants.%Int(%int_32.loc46_25) [template = constants.%i32]
// CHECK:STDOUT:     %.loc46_25.1: type = value_of_initializer %int.make_type_signed.loc46_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc46_25.2: type = converted %int.make_type_signed.loc46_25, %.loc46_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc46_33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc46_33: init type = call constants.%Int(%int_32.loc46_33) [template = constants.%i32]
// CHECK:STDOUT:     %.loc46_33.1: type = value_of_initializer %int.make_type_signed.loc46_33 [template = constants.%i32]
// CHECK:STDOUT:     %.loc46_33.2: type = converted %int.make_type_signed.loc46_33, %.loc46_33.1 [template = constants.%i32]
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
// CHECK:STDOUT:     %int_32.loc57_26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc57_26: init type = call constants.%Int(%int_32.loc57_26) [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_26.1: type = value_of_initializer %int.make_type_signed.loc57_26 [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_26.2: type = converted %int.make_type_signed.loc57_26, %.loc57_26.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc57_34: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc57_34: init type = call constants.%Int(%int_32.loc57_34) [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_34.1: type = value_of_initializer %int.make_type_signed.loc57_34 [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_34.2: type = converted %int.make_type_signed.loc57_34, %.loc57_34.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc57_42: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc57_42: init type = call constants.%Int(%int_32.loc57_42) [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_42.1: type = value_of_initializer %int.make_type_signed.loc57_42 [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_42.2: type = converted %int.make_type_signed.loc57_42, %.loc57_42.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc57_50: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc57_50: init type = call constants.%Int(%int_32.loc57_50) [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_50.1: type = value_of_initializer %int.make_type_signed.loc57_50 [template = constants.%i32]
// CHECK:STDOUT:     %.loc57_50.2: type = converted %int.make_type_signed.loc57_50, %.loc57_50.1 [template = constants.%i32]
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
// CHECK:STDOUT:     %int_32.loc68_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc68_32: init type = call constants.%Int(%int_32.loc68_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc68_32.1: type = value_of_initializer %int.make_type_signed.loc68_32 [template = constants.%i32]
// CHECK:STDOUT:     %.loc68_32.2: type = converted %int.make_type_signed.loc68_32, %.loc68_32.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc68_40: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc68_40: init type = call constants.%Int(%int_32.loc68_40) [template = constants.%i32]
// CHECK:STDOUT:     %.loc68_40.1: type = value_of_initializer %int.make_type_signed.loc68_40 [template = constants.%i32]
// CHECK:STDOUT:     %.loc68_40.2: type = converted %int.make_type_signed.loc68_40, %.loc68_40.1 [template = constants.%i32]
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
// CHECK:STDOUT: fn @JustRight(%a.param_patt: %i32) -> %i32 = "int.snegate";
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
// CHECK:STDOUT: --- fail_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %Sub.type: type = fn_type @Sub [template]
// CHECK:STDOUT:   %Sub: %Sub.type = struct_value () [template]
// CHECK:STDOUT:   %int_2147483647.1: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_2147483647.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2147483647.2: %i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %int_-2147483647: %i32 = int_value -2147483647 [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_-2147483648: %i32 = int_value -2147483648 [template]
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
// CHECK:STDOUT:     %int_32.loc4_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_14: init type = call constants.%Int(%int_32.loc4_14) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_14.1: type = value_of_initializer %int.make_type_signed.loc4_14 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_14.2: type = converted %int.make_type_signed.loc4_14, %.loc4_14.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_22: init type = call constants.%Int(%int_32.loc4_22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_22.1: type = value_of_initializer %int.make_type_signed.loc4_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_22.2: type = converted %int.make_type_signed.loc4_22, %.loc4_22.1 [template = constants.%i32]
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
// CHECK:STDOUT:     %int_32.loc5_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc5_11: init type = call constants.%Int(%int_32.loc5_11) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_11.1: type = value_of_initializer %int.make_type_signed.loc5_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_11.2: type = converted %int.make_type_signed.loc5_11, %.loc5_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc5_19: init type = call constants.%Int(%int_32.loc5_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_19.1: type = value_of_initializer %int.make_type_signed.loc5_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_19.2: type = converted %int.make_type_signed.loc5_19, %.loc5_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc5_27: init type = call constants.%Int(%int_32.loc5_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_27.1: type = value_of_initializer %int.make_type_signed.loc5_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_27.2: type = converted %int.make_type_signed.loc5_27, %.loc5_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%int_32.loc8) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8.1: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8.2: type = converted %int.make_type_signed.loc8, %.loc8_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14: init type = call constants.%Int(%int_32.loc14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_8.1: type = value_of_initializer %int.make_type_signed.loc14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_8.2: type = converted %int.make_type_signed.loc14, %.loc14_8.1 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.ssub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref.loc8_14: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %Negate.ref.loc8_21: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_2147483647.loc8: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %impl.elem0.loc8: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc8: <bound method> = bound_method %int_2147483647.loc8, %impl.elem0.loc8 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc8: <specific function> = specific_function %Convert.bound.loc8, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %i32 = call %Convert.specific_fn.loc8(%int_2147483647.loc8) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc8_28.1: %i32 = value_of_initializer %int.convert_checked.loc8 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc8_28.2: %i32 = converted %int_2147483647.loc8, %.loc8_28.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %int.snegate.loc8_38: init %i32 = call %Negate.ref.loc8_21(%.loc8_28.2) [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %.loc8_38.1: %i32 = value_of_initializer %int.snegate.loc8_38 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %.loc8_38.2: %i32 = converted %int.snegate.loc8_38, %.loc8_38.1 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %int.snegate.loc8_39: init %i32 = call %Negate.ref.loc8_14(%.loc8_38.2) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc8_40.1: %i32 = value_of_initializer %int.snegate.loc8_39 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc8_40.2: %i32 = converted %int.snegate.loc8_39, %.loc8_40.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc8_40.2
// CHECK:STDOUT:   %Negate.ref.loc14_14: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %Sub.ref: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Negate.ref.loc14_25: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_2147483647.loc14: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %impl.elem0.loc14_32: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc14_32: <bound method> = bound_method %int_2147483647.loc14, %impl.elem0.loc14_32 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_32: <specific function> = specific_function %Convert.bound.loc14_32, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc14_32: init %i32 = call %Convert.specific_fn.loc14_32(%int_2147483647.loc14) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc14_32.1: %i32 = value_of_initializer %int.convert_checked.loc14_32 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc14_32.2: %i32 = converted %int_2147483647.loc14, %.loc14_32.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %int.snegate.loc14_42: init %i32 = call %Negate.ref.loc14_25(%.loc14_32.2) [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc14_42.1: %i32 = value_of_initializer %int.snegate.loc14_42 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %.loc14_42.2: %i32 = converted %int.snegate.loc14_42, %.loc14_42.1 [template = constants.%int_-2147483647]
// CHECK:STDOUT:   %impl.elem0.loc14_45: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc14_45: <bound method> = bound_method %int_1, %impl.elem0.loc14_45 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_45: <specific function> = specific_function %Convert.bound.loc14_45, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc14_45: init %i32 = call %Convert.specific_fn.loc14_45(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_45.1: %i32 = value_of_initializer %int.convert_checked.loc14_45 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_45.2: %i32 = converted %int_1, %.loc14_45.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.ssub: init %i32 = call %Sub.ref(%.loc14_42.2, %.loc14_45.2) [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc14_46.1: %i32 = value_of_initializer %int.ssub [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc14_46.2: %i32 = converted %int.ssub, %.loc14_46.1 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %int.snegate.loc14_47: init %i32 = call %Negate.ref.loc14_14(%.loc14_46.2) [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc14_48.1: %i32 = value_of_initializer %int.snegate.loc14_47 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %.loc14_48.2: %i32 = converted %int.snegate.loc14_47, %.loc14_48.1 [template = constants.%int_-2147483648]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc14_48.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
