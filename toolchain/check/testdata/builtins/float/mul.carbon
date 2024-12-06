// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/mul.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/mul.carbon

// --- mul_sub.carbon

fn Mul(a: f64, b: f64) -> f64 = "float.mul";

fn RuntimeCall(a: f64, b: f64) -> f64 {
  return Mul(a, b);
}

var x: f64 = Mul(2.0, 0.5);

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.mul" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew(a: f64) -> f64 = "float.mul";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew(a: f64) -> f64 = "float.mul";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.mul" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: f64, b: f64, c: f64) -> f64 = "float.mul";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: f64, b: f64, c: f64) -> f64 = "float.mul";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+3]]:1: error: invalid signature for builtin function "float.mul" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: f64, b: f64) -> bool = "float.mul";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn BadReturnType(a: f64, b: f64) -> bool = "float.mul";
fn JustRight(a: f64, b: f64) -> f64 = "float.mul";

fn RuntimeCallTooFew(a: f64) -> f64 {
  return TooFew(a);
}

fn RuntimeCallTooMany(a: f64, b: f64, c: f64) -> f64 {
  return TooMany(a, b, c);
}

fn RuntimeCallBadReturnType(a: f64, b: f64) -> bool {
  return BadReturnType(a, b);
}

// CHECK:STDOUT: --- mul_sub.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %Mul.type: type = fn_type @Mul [template]
// CHECK:STDOUT:   %Mul: %Mul.type = struct_value () [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT:   %float.1: f64 = float_literal 2 [template]
// CHECK:STDOUT:   %float.2: f64 = float_literal 0.5 [template]
// CHECK:STDOUT:   %float.3: f64 = float_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Float = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Mul = %Mul.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mul.decl: %Mul.type = fn_decl @Mul [template = constants.%Mul] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc2_11: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc2_11: init type = call constants.%Float(%int_64.loc2_11) [template = f64]
// CHECK:STDOUT:     %.loc2_11.1: type = value_of_initializer %float.make_type.loc2_11 [template = f64]
// CHECK:STDOUT:     %.loc2_11.2: type = converted %float.make_type.loc2_11, %.loc2_11.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc2_19: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc2_19: init type = call constants.%Float(%int_64.loc2_19) [template = f64]
// CHECK:STDOUT:     %.loc2_19.1: type = value_of_initializer %float.make_type.loc2_19 [template = f64]
// CHECK:STDOUT:     %.loc2_19.2: type = converted %float.make_type.loc2_19, %.loc2_19.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc2_27: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc2_27: init type = call constants.%Float(%int_64.loc2_27) [template = f64]
// CHECK:STDOUT:     %.loc2_27.1: type = value_of_initializer %float.make_type.loc2_27 [template = f64]
// CHECK:STDOUT:     %.loc2_27.2: type = converted %float.make_type.loc2_27, %.loc2_27.1 [template = f64]
// CHECK:STDOUT:     %a.param: f64 = value_param runtime_param0
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param runtime_param1
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref f64 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc4_19: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc4_19: init type = call constants.%Float(%int_64.loc4_19) [template = f64]
// CHECK:STDOUT:     %.loc4_19.1: type = value_of_initializer %float.make_type.loc4_19 [template = f64]
// CHECK:STDOUT:     %.loc4_19.2: type = converted %float.make_type.loc4_19, %.loc4_19.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc4_27: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc4_27: init type = call constants.%Float(%int_64.loc4_27) [template = f64]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %float.make_type.loc4_27 [template = f64]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %float.make_type.loc4_27, %.loc4_27.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc4_35: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc4_35: init type = call constants.%Float(%int_64.loc4_35) [template = f64]
// CHECK:STDOUT:     %.loc4_35.1: type = value_of_initializer %float.make_type.loc4_35 [template = f64]
// CHECK:STDOUT:     %.loc4_35.2: type = converted %float.make_type.loc4_35, %.loc4_35.1 [template = f64]
// CHECK:STDOUT:     %a.param: f64 = value_param runtime_param0
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param runtime_param1
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref f64 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%int_64) [template = f64]
// CHECK:STDOUT:   %.loc8_8.1: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc8_8.2: type = converted %float.make_type, %.loc8_8.1 [template = f64]
// CHECK:STDOUT:   %x.var: ref f64 = var x
// CHECK:STDOUT:   %x: ref f64 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mul(%a.param_patt: f64, %b.param_patt: f64) -> f64 = "float.mul";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: f64, %b.param_patt: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mul.ref: %Mul.type = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %float.mul: init f64 = call %Mul.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc5_19.1: f64 = value_of_initializer %float.mul
// CHECK:STDOUT:   %.loc5_19.2: f64 = converted %float.mul, %.loc5_19.1
// CHECK:STDOUT:   return %.loc5_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mul.ref: %Mul.type = name_ref Mul, file.%Mul.decl [template = constants.%Mul]
// CHECK:STDOUT:   %float.loc8_18: f64 = float_literal 2 [template = constants.%float.1]
// CHECK:STDOUT:   %float.loc8_23: f64 = float_literal 0.5 [template = constants.%float.2]
// CHECK:STDOUT:   %float.mul: init f64 = call %Mul.ref(%float.loc8_18, %float.loc8_23) [template = constants.%float.3]
// CHECK:STDOUT:   assign file.%x.var, %float.mul
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
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
// CHECK:STDOUT:     .Float = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.2
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
// CHECK:STDOUT:     .RuntimeCallTooFew = %RuntimeCallTooFew.decl
// CHECK:STDOUT:     .RuntimeCallTooMany = %RuntimeCallTooMany.decl
// CHECK:STDOUT:     .RuntimeCallBadReturnType = %RuntimeCallBadReturnType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %TooFew.decl: %TooFew.type = fn_decl @TooFew [template = constants.%TooFew] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc8_14: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc8_14: init type = call constants.%Float(%int_64.loc8_14) [template = f64]
// CHECK:STDOUT:     %.loc8_14.1: type = value_of_initializer %float.make_type.loc8_14 [template = f64]
// CHECK:STDOUT:     %.loc8_14.2: type = converted %float.make_type.loc8_14, %.loc8_14.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc8_22: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc8_22: init type = call constants.%Float(%int_64.loc8_22) [template = f64]
// CHECK:STDOUT:     %.loc8_22.1: type = value_of_initializer %float.make_type.loc8_22 [template = f64]
// CHECK:STDOUT:     %.loc8_22.2: type = converted %float.make_type.loc8_22, %.loc8_22.1 [template = f64]
// CHECK:STDOUT:     %a.param: f64 = value_param runtime_param0
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref f64 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany.decl: %TooMany.type = fn_decl @TooMany [template = constants.%TooMany] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %c.patt: f64 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: f64 = value_param_pattern %c.patt, runtime_param2
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, runtime_param3
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc13_15: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc13_15: init type = call constants.%Float(%int_64.loc13_15) [template = f64]
// CHECK:STDOUT:     %.loc13_15.1: type = value_of_initializer %float.make_type.loc13_15 [template = f64]
// CHECK:STDOUT:     %.loc13_15.2: type = converted %float.make_type.loc13_15, %.loc13_15.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc13_23: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc13_23: init type = call constants.%Float(%int_64.loc13_23) [template = f64]
// CHECK:STDOUT:     %.loc13_23.1: type = value_of_initializer %float.make_type.loc13_23 [template = f64]
// CHECK:STDOUT:     %.loc13_23.2: type = converted %float.make_type.loc13_23, %.loc13_23.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc13_31: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc13_31: init type = call constants.%Float(%int_64.loc13_31) [template = f64]
// CHECK:STDOUT:     %.loc13_31.1: type = value_of_initializer %float.make_type.loc13_31 [template = f64]
// CHECK:STDOUT:     %.loc13_31.2: type = converted %float.make_type.loc13_31, %.loc13_31.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc13_39: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc13_39: init type = call constants.%Float(%int_64.loc13_39) [template = f64]
// CHECK:STDOUT:     %.loc13_39.1: type = value_of_initializer %float.make_type.loc13_39 [template = f64]
// CHECK:STDOUT:     %.loc13_39.2: type = converted %float.make_type.loc13_39, %.loc13_39.1 [template = f64]
// CHECK:STDOUT:     %a.param: f64 = value_param runtime_param0
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param runtime_param1
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: f64 = value_param runtime_param2
// CHECK:STDOUT:     %c: f64 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref f64 = out_param runtime_param3
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BadReturnType.decl: %BadReturnType.type = fn_decl @BadReturnType [template = constants.%BadReturnType] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc17_21: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc17_21: init type = call constants.%Float(%int_64.loc17_21) [template = f64]
// CHECK:STDOUT:     %.loc17_21.1: type = value_of_initializer %float.make_type.loc17_21 [template = f64]
// CHECK:STDOUT:     %.loc17_21.2: type = converted %float.make_type.loc17_21, %.loc17_21.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc17_29: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc17_29: init type = call constants.%Float(%int_64.loc17_29) [template = f64]
// CHECK:STDOUT:     %.loc17_29.1: type = value_of_initializer %float.make_type.loc17_29 [template = f64]
// CHECK:STDOUT:     %.loc17_29.2: type = converted %float.make_type.loc17_29, %.loc17_29.1 [template = f64]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc17_37.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc17_37.2: type = converted %bool.make_type, %.loc17_37.1 [template = bool]
// CHECK:STDOUT:     %a.param: f64 = value_param runtime_param0
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param runtime_param1
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %JustRight.decl: %JustRight.type = fn_decl @JustRight [template = constants.%JustRight] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc18_17: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc18_17: init type = call constants.%Float(%int_64.loc18_17) [template = f64]
// CHECK:STDOUT:     %.loc18_17.1: type = value_of_initializer %float.make_type.loc18_17 [template = f64]
// CHECK:STDOUT:     %.loc18_17.2: type = converted %float.make_type.loc18_17, %.loc18_17.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc18_25: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc18_25: init type = call constants.%Float(%int_64.loc18_25) [template = f64]
// CHECK:STDOUT:     %.loc18_25.1: type = value_of_initializer %float.make_type.loc18_25 [template = f64]
// CHECK:STDOUT:     %.loc18_25.2: type = converted %float.make_type.loc18_25, %.loc18_25.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc18_33: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc18_33: init type = call constants.%Float(%int_64.loc18_33) [template = f64]
// CHECK:STDOUT:     %.loc18_33.1: type = value_of_initializer %float.make_type.loc18_33 [template = f64]
// CHECK:STDOUT:     %.loc18_33.2: type = converted %float.make_type.loc18_33, %.loc18_33.1 [template = f64]
// CHECK:STDOUT:     %a.param: f64 = value_param runtime_param0
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param runtime_param1
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref f64 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooFew.decl: %RuntimeCallTooFew.type = fn_decl @RuntimeCallTooFew [template = constants.%RuntimeCallTooFew] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc20_25: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc20_25: init type = call constants.%Float(%int_64.loc20_25) [template = f64]
// CHECK:STDOUT:     %.loc20_25.1: type = value_of_initializer %float.make_type.loc20_25 [template = f64]
// CHECK:STDOUT:     %.loc20_25.2: type = converted %float.make_type.loc20_25, %.loc20_25.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc20_33: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc20_33: init type = call constants.%Float(%int_64.loc20_33) [template = f64]
// CHECK:STDOUT:     %.loc20_33.1: type = value_of_initializer %float.make_type.loc20_33 [template = f64]
// CHECK:STDOUT:     %.loc20_33.2: type = converted %float.make_type.loc20_33, %.loc20_33.1 [template = f64]
// CHECK:STDOUT:     %a.param: f64 = value_param runtime_param0
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref f64 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooMany.decl: %RuntimeCallTooMany.type = fn_decl @RuntimeCallTooMany [template = constants.%RuntimeCallTooMany] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %c.patt: f64 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: f64 = value_param_pattern %c.patt, runtime_param2
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, runtime_param3
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc24_26: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc24_26: init type = call constants.%Float(%int_64.loc24_26) [template = f64]
// CHECK:STDOUT:     %.loc24_26.1: type = value_of_initializer %float.make_type.loc24_26 [template = f64]
// CHECK:STDOUT:     %.loc24_26.2: type = converted %float.make_type.loc24_26, %.loc24_26.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc24_34: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc24_34: init type = call constants.%Float(%int_64.loc24_34) [template = f64]
// CHECK:STDOUT:     %.loc24_34.1: type = value_of_initializer %float.make_type.loc24_34 [template = f64]
// CHECK:STDOUT:     %.loc24_34.2: type = converted %float.make_type.loc24_34, %.loc24_34.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc24_42: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc24_42: init type = call constants.%Float(%int_64.loc24_42) [template = f64]
// CHECK:STDOUT:     %.loc24_42.1: type = value_of_initializer %float.make_type.loc24_42 [template = f64]
// CHECK:STDOUT:     %.loc24_42.2: type = converted %float.make_type.loc24_42, %.loc24_42.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc24_50: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc24_50: init type = call constants.%Float(%int_64.loc24_50) [template = f64]
// CHECK:STDOUT:     %.loc24_50.1: type = value_of_initializer %float.make_type.loc24_50 [template = f64]
// CHECK:STDOUT:     %.loc24_50.2: type = converted %float.make_type.loc24_50, %.loc24_50.1 [template = f64]
// CHECK:STDOUT:     %a.param: f64 = value_param runtime_param0
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param runtime_param1
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: f64 = value_param runtime_param2
// CHECK:STDOUT:     %c: f64 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref f64 = out_param runtime_param3
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallBadReturnType.decl: %RuntimeCallBadReturnType.type = fn_decl @RuntimeCallBadReturnType [template = constants.%RuntimeCallBadReturnType] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc28_32: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc28_32: init type = call constants.%Float(%int_64.loc28_32) [template = f64]
// CHECK:STDOUT:     %.loc28_32.1: type = value_of_initializer %float.make_type.loc28_32 [template = f64]
// CHECK:STDOUT:     %.loc28_32.2: type = converted %float.make_type.loc28_32, %.loc28_32.1 [template = f64]
// CHECK:STDOUT:     %int_64.loc28_40: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc28_40: init type = call constants.%Float(%int_64.loc28_40) [template = f64]
// CHECK:STDOUT:     %.loc28_40.1: type = value_of_initializer %float.make_type.loc28_40 [template = f64]
// CHECK:STDOUT:     %.loc28_40.2: type = converted %float.make_type.loc28_40, %.loc28_40.1 [template = f64]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc28_48.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc28_48.2: type = converted %bool.make_type, %.loc28_48.1 [template = bool]
// CHECK:STDOUT:     %a.param: f64 = value_param runtime_param0
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param runtime_param1
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew(%a.param_patt: f64) -> f64;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany(%a.param_patt: f64, %b.param_patt: f64, %c.param_patt: f64) -> f64;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BadReturnType(%a.param_patt: f64, %b.param_patt: f64) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @JustRight(%a.param_patt: f64, %b.param_patt: f64) -> f64 = "float.mul";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooFew(%a.param_patt: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooFew.ref: %TooFew.type = name_ref TooFew, file.%TooFew.decl [template = constants.%TooFew]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %TooFew.call: init f64 = call %TooFew.ref(%a.ref)
// CHECK:STDOUT:   %.loc21_19.1: f64 = value_of_initializer %TooFew.call
// CHECK:STDOUT:   %.loc21_19.2: f64 = converted %TooFew.call, %.loc21_19.1
// CHECK:STDOUT:   return %.loc21_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooMany(%a.param_patt: f64, %b.param_patt: f64, %c.param_patt: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooMany.ref: %TooMany.type = name_ref TooMany, file.%TooMany.decl [template = constants.%TooMany]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: f64 = name_ref c, %c
// CHECK:STDOUT:   %TooMany.call: init f64 = call %TooMany.ref(%a.ref, %b.ref, %c.ref)
// CHECK:STDOUT:   %.loc25_26.1: f64 = value_of_initializer %TooMany.call
// CHECK:STDOUT:   %.loc25_26.2: f64 = converted %TooMany.call, %.loc25_26.1
// CHECK:STDOUT:   return %.loc25_26.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallBadReturnType(%a.param_patt: f64, %b.param_patt: f64) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BadReturnType.ref: %BadReturnType.type = name_ref BadReturnType, file.%BadReturnType.decl [template = constants.%BadReturnType]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %BadReturnType.call: init bool = call %BadReturnType.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc29_29.1: bool = value_of_initializer %BadReturnType.call
// CHECK:STDOUT:   %.loc29_29.2: bool = converted %BadReturnType.call, %.loc29_29.1
// CHECK:STDOUT:   return %.loc29_29.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
