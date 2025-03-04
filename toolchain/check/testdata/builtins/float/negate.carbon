// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/negate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/negate.carbon

// --- float_negate.carbon

fn Negate(a: f64) -> f64 = "float.negate";

fn RuntimeCallIsValid(a: f64, b: f64) -> f64 {
  return Negate(a);
}

let a: f64 = Negate(1.5);

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.negate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooFew() -> f64 = "float.negate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew() -> f64 = "float.negate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.negate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn TooMany(a: f64, b: f64) -> f64 = "float.negate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: f64, b: f64) -> f64 = "float.negate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.negate" [InvalidBuiltinSignature]
// CHECK:STDERR: fn BadReturnType(a: f64) -> bool = "float.negate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: f64) -> bool = "float.negate";
fn JustRight(a: f64) -> f64 = "float.negate";

fn RuntimeCallIsValidTooFew(a: f64) -> f64 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return TooFew(a);
  // CHECK:STDERR:          ^~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-17]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn TooFew() -> f64 = "float.negate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooFew(a);
}

fn RuntimeCallIsValidTooMany(a: f64, b: f64, c: f64) -> f64 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 3 arguments passed to function expecting 2 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return TooMany(a, b, c);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-23]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn TooMany(a: f64, b: f64) -> f64 = "float.negate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooMany(a, b, c);
}

fn RuntimeCallIsValidBadReturnType(a: f64, b: f64) -> bool {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
  // CHECK:STDERR:   return BadReturnType(a, b);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-29]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn BadReturnType(a: f64) -> bool = "float.negate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return BadReturnType(a, b);
}

// CHECK:STDOUT: --- float_negate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [concrete]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [concrete]
// CHECK:STDOUT:   %RuntimeCallIsValid.type: type = fn_type @RuntimeCallIsValid [concrete]
// CHECK:STDOUT:   %RuntimeCallIsValid: %RuntimeCallIsValid.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.e93: f64 = float_literal 1.5 [concrete]
// CHECK:STDOUT:   %float.928: f64 = float_literal -1.5 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .RuntimeCallIsValid = %RuntimeCallIsValid.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [concrete = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc2_22: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc2_22: init type = call constants.%Float(%int_64.loc2_22) [concrete = f64]
// CHECK:STDOUT:     %.loc2_22.1: type = value_of_initializer %float.make_type.loc2_22 [concrete = f64]
// CHECK:STDOUT:     %.loc2_22.2: type = converted %float.make_type.loc2_22, %.loc2_22.1 [concrete = f64]
// CHECK:STDOUT:     %a.param: f64 = value_param call_param0
// CHECK:STDOUT:     %.loc2_14.1: type = splice_block %.loc2_14.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc2_14: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc2_14: init type = call constants.%Float(%int_64.loc2_14) [concrete = f64]
// CHECK:STDOUT:       %.loc2_14.2: type = value_of_initializer %float.make_type.loc2_14 [concrete = f64]
// CHECK:STDOUT:       %.loc2_14.3: type = converted %float.make_type.loc2_14, %.loc2_14.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref f64 = out_param call_param1
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallIsValid.decl: %RuntimeCallIsValid.type = fn_decl @RuntimeCallIsValid [concrete = constants.%RuntimeCallIsValid] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc4_42: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc4_42: init type = call constants.%Float(%int_64.loc4_42) [concrete = f64]
// CHECK:STDOUT:     %.loc4_42.1: type = value_of_initializer %float.make_type.loc4_42 [concrete = f64]
// CHECK:STDOUT:     %.loc4_42.2: type = converted %float.make_type.loc4_42, %.loc4_42.1 [concrete = f64]
// CHECK:STDOUT:     %a.param: f64 = value_param call_param0
// CHECK:STDOUT:     %.loc4_26.1: type = splice_block %.loc4_26.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc4_26: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc4_26: init type = call constants.%Float(%int_64.loc4_26) [concrete = f64]
// CHECK:STDOUT:       %.loc4_26.2: type = value_of_initializer %float.make_type.loc4_26 [concrete = f64]
// CHECK:STDOUT:       %.loc4_26.3: type = converted %float.make_type.loc4_26, %.loc4_26.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param call_param1
// CHECK:STDOUT:     %.loc4_34.1: type = splice_block %.loc4_34.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc4_34: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc4_34: init type = call constants.%Float(%int_64.loc4_34) [concrete = f64]
// CHECK:STDOUT:       %.loc4_34.2: type = value_of_initializer %float.make_type.loc4_34 [concrete = f64]
// CHECK:STDOUT:       %.loc4_34.3: type = converted %float.make_type.loc4_34, %.loc4_34.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref f64 = out_param call_param2
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_8.1: type = splice_block %.loc8_8.3 [concrete = f64] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type: init type = call constants.%Float(%int_64) [concrete = f64]
// CHECK:STDOUT:     %.loc8_8.2: type = value_of_initializer %float.make_type [concrete = f64]
// CHECK:STDOUT:     %.loc8_8.3: type = converted %float.make_type, %.loc8_8.2 [concrete = f64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_24.1: ref f64 = temporary_storage
// CHECK:STDOUT:   %.loc8_24.2: ref f64 = temporary %.loc8_24.1, @__global_init.%float.negate
// CHECK:STDOUT:   %a: ref f64 = bind_name a, %.loc8_24.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a.param_patt: f64) -> f64 = "float.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param_patt: f64, %b.param_patt: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %float.negate: init f64 = call %Negate.ref(%a.ref)
// CHECK:STDOUT:   %.loc5_19.1: f64 = value_of_initializer %float.negate
// CHECK:STDOUT:   %.loc5_19.2: f64 = converted %float.negate, %.loc5_19.1
// CHECK:STDOUT:   return %.loc5_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [concrete = constants.%Negate]
// CHECK:STDOUT:   %float: f64 = float_literal 1.5 [concrete = constants.%float.e93]
// CHECK:STDOUT:   %float.negate: init f64 = call %Negate.ref(%float) [concrete = constants.%float.928]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %TooFew.type: type = fn_type @TooFew [concrete]
// CHECK:STDOUT:   %TooFew: %TooFew.type = struct_value () [concrete]
// CHECK:STDOUT:   %TooMany.type: type = fn_type @TooMany [concrete]
// CHECK:STDOUT:   %TooMany: %TooMany.type = struct_value () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %BadReturnType.type: type = fn_type @BadReturnType [concrete]
// CHECK:STDOUT:   %BadReturnType: %BadReturnType.type = struct_value () [concrete]
// CHECK:STDOUT:   %JustRight.type: type = fn_type @JustRight [concrete]
// CHECK:STDOUT:   %JustRight: %JustRight.type = struct_value () [concrete]
// CHECK:STDOUT:   %RuntimeCallIsValidTooFew.type: type = fn_type @RuntimeCallIsValidTooFew [concrete]
// CHECK:STDOUT:   %RuntimeCallIsValidTooFew: %RuntimeCallIsValidTooFew.type = struct_value () [concrete]
// CHECK:STDOUT:   %RuntimeCallIsValidTooMany.type: type = fn_type @RuntimeCallIsValidTooMany [concrete]
// CHECK:STDOUT:   %RuntimeCallIsValidTooMany: %RuntimeCallIsValidTooMany.type = struct_value () [concrete]
// CHECK:STDOUT:   %RuntimeCallIsValidBadReturnType.type: type = fn_type @RuntimeCallIsValidBadReturnType [concrete]
// CHECK:STDOUT:   %RuntimeCallIsValidBadReturnType: %RuntimeCallIsValidBadReturnType.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .TooFew = %TooFew.decl
// CHECK:STDOUT:     .TooMany = %TooMany.decl
// CHECK:STDOUT:     .BadReturnType = %BadReturnType.decl
// CHECK:STDOUT:     .JustRight = %JustRight.decl
// CHECK:STDOUT:     .RuntimeCallIsValidTooFew = %RuntimeCallIsValidTooFew.decl
// CHECK:STDOUT:     .RuntimeCallIsValidTooMany = %RuntimeCallIsValidTooMany.decl
// CHECK:STDOUT:     .RuntimeCallIsValidBadReturnType = %RuntimeCallIsValidBadReturnType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %TooFew.decl: %TooFew.type = fn_decl @TooFew [concrete = constants.%TooFew] {
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type: init type = call constants.%Float(%int_64) [concrete = f64]
// CHECK:STDOUT:     %.loc8_16.1: type = value_of_initializer %float.make_type [concrete = f64]
// CHECK:STDOUT:     %.loc8_16.2: type = converted %float.make_type, %.loc8_16.1 [concrete = f64]
// CHECK:STDOUT:     %return.param: ref f64 = out_param call_param0
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany.decl: %TooMany.type = fn_decl @TooMany [concrete = constants.%TooMany] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc13_31: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc13_31: init type = call constants.%Float(%int_64.loc13_31) [concrete = f64]
// CHECK:STDOUT:     %.loc13_31.1: type = value_of_initializer %float.make_type.loc13_31 [concrete = f64]
// CHECK:STDOUT:     %.loc13_31.2: type = converted %float.make_type.loc13_31, %.loc13_31.1 [concrete = f64]
// CHECK:STDOUT:     %a.param: f64 = value_param call_param0
// CHECK:STDOUT:     %.loc13_15.1: type = splice_block %.loc13_15.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc13_15: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc13_15: init type = call constants.%Float(%int_64.loc13_15) [concrete = f64]
// CHECK:STDOUT:       %.loc13_15.2: type = value_of_initializer %float.make_type.loc13_15 [concrete = f64]
// CHECK:STDOUT:       %.loc13_15.3: type = converted %float.make_type.loc13_15, %.loc13_15.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param call_param1
// CHECK:STDOUT:     %.loc13_23.1: type = splice_block %.loc13_23.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc13_23: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc13_23: init type = call constants.%Float(%int_64.loc13_23) [concrete = f64]
// CHECK:STDOUT:       %.loc13_23.2: type = value_of_initializer %float.make_type.loc13_23 [concrete = f64]
// CHECK:STDOUT:       %.loc13_23.3: type = converted %float.make_type.loc13_23, %.loc13_23.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref f64 = out_param call_param2
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BadReturnType.decl: %BadReturnType.type = fn_decl @BadReturnType [concrete = constants.%BadReturnType] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc18_29.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc18_29.2: type = converted %bool.make_type, %.loc18_29.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: f64 = value_param call_param0
// CHECK:STDOUT:     %.loc18_21.1: type = splice_block %.loc18_21.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type: init type = call constants.%Float(%int_64) [concrete = f64]
// CHECK:STDOUT:       %.loc18_21.2: type = value_of_initializer %float.make_type [concrete = f64]
// CHECK:STDOUT:       %.loc18_21.3: type = converted %float.make_type, %.loc18_21.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %JustRight.decl: %JustRight.type = fn_decl @JustRight [concrete = constants.%JustRight] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc19_25: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc19_25: init type = call constants.%Float(%int_64.loc19_25) [concrete = f64]
// CHECK:STDOUT:     %.loc19_25.1: type = value_of_initializer %float.make_type.loc19_25 [concrete = f64]
// CHECK:STDOUT:     %.loc19_25.2: type = converted %float.make_type.loc19_25, %.loc19_25.1 [concrete = f64]
// CHECK:STDOUT:     %a.param: f64 = value_param call_param0
// CHECK:STDOUT:     %.loc19_17.1: type = splice_block %.loc19_17.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc19_17: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc19_17: init type = call constants.%Float(%int_64.loc19_17) [concrete = f64]
// CHECK:STDOUT:       %.loc19_17.2: type = value_of_initializer %float.make_type.loc19_17 [concrete = f64]
// CHECK:STDOUT:       %.loc19_17.3: type = converted %float.make_type.loc19_17, %.loc19_17.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref f64 = out_param call_param1
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallIsValidTooFew.decl: %RuntimeCallIsValidTooFew.type = fn_decl @RuntimeCallIsValidTooFew [concrete = constants.%RuntimeCallIsValidTooFew] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc21_40: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc21_40: init type = call constants.%Float(%int_64.loc21_40) [concrete = f64]
// CHECK:STDOUT:     %.loc21_40.1: type = value_of_initializer %float.make_type.loc21_40 [concrete = f64]
// CHECK:STDOUT:     %.loc21_40.2: type = converted %float.make_type.loc21_40, %.loc21_40.1 [concrete = f64]
// CHECK:STDOUT:     %a.param: f64 = value_param call_param0
// CHECK:STDOUT:     %.loc21_32.1: type = splice_block %.loc21_32.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc21_32: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc21_32: init type = call constants.%Float(%int_64.loc21_32) [concrete = f64]
// CHECK:STDOUT:       %.loc21_32.2: type = value_of_initializer %float.make_type.loc21_32 [concrete = f64]
// CHECK:STDOUT:       %.loc21_32.3: type = converted %float.make_type.loc21_32, %.loc21_32.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref f64 = out_param call_param1
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallIsValidTooMany.decl: %RuntimeCallIsValidTooMany.type = fn_decl @RuntimeCallIsValidTooMany [concrete = constants.%RuntimeCallIsValidTooMany] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %c.patt: f64 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: f64 = value_param_pattern %c.patt, call_param2
// CHECK:STDOUT:     %return.patt: f64 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: f64 = out_param_pattern %return.patt, call_param3
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.loc32_57: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type.loc32_57: init type = call constants.%Float(%int_64.loc32_57) [concrete = f64]
// CHECK:STDOUT:     %.loc32_57.1: type = value_of_initializer %float.make_type.loc32_57 [concrete = f64]
// CHECK:STDOUT:     %.loc32_57.2: type = converted %float.make_type.loc32_57, %.loc32_57.1 [concrete = f64]
// CHECK:STDOUT:     %a.param: f64 = value_param call_param0
// CHECK:STDOUT:     %.loc32_33.1: type = splice_block %.loc32_33.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc32_33: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc32_33: init type = call constants.%Float(%int_64.loc32_33) [concrete = f64]
// CHECK:STDOUT:       %.loc32_33.2: type = value_of_initializer %float.make_type.loc32_33 [concrete = f64]
// CHECK:STDOUT:       %.loc32_33.3: type = converted %float.make_type.loc32_33, %.loc32_33.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param call_param1
// CHECK:STDOUT:     %.loc32_41.1: type = splice_block %.loc32_41.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc32_41: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc32_41: init type = call constants.%Float(%int_64.loc32_41) [concrete = f64]
// CHECK:STDOUT:       %.loc32_41.2: type = value_of_initializer %float.make_type.loc32_41 [concrete = f64]
// CHECK:STDOUT:       %.loc32_41.3: type = converted %float.make_type.loc32_41, %.loc32_41.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: f64 = value_param call_param2
// CHECK:STDOUT:     %.loc32_49.1: type = splice_block %.loc32_49.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc32_49: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc32_49: init type = call constants.%Float(%int_64.loc32_49) [concrete = f64]
// CHECK:STDOUT:       %.loc32_49.2: type = value_of_initializer %float.make_type.loc32_49 [concrete = f64]
// CHECK:STDOUT:       %.loc32_49.3: type = converted %float.make_type.loc32_49, %.loc32_49.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: f64 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref f64 = out_param call_param3
// CHECK:STDOUT:     %return: ref f64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallIsValidBadReturnType.decl: %RuntimeCallIsValidBadReturnType.type = fn_decl @RuntimeCallIsValidBadReturnType [concrete = constants.%RuntimeCallIsValidBadReturnType] {
// CHECK:STDOUT:     %a.patt: f64 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: f64 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: f64 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: f64 = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc43_55.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc43_55.2: type = converted %bool.make_type, %.loc43_55.1 [concrete = bool]
// CHECK:STDOUT:     %a.param: f64 = value_param call_param0
// CHECK:STDOUT:     %.loc43_39.1: type = splice_block %.loc43_39.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc43_39: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc43_39: init type = call constants.%Float(%int_64.loc43_39) [concrete = f64]
// CHECK:STDOUT:       %.loc43_39.2: type = value_of_initializer %float.make_type.loc43_39 [concrete = f64]
// CHECK:STDOUT:       %.loc43_39.3: type = converted %float.make_type.loc43_39, %.loc43_39.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: f64 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: f64 = value_param call_param1
// CHECK:STDOUT:     %.loc43_47.1: type = splice_block %.loc43_47.3 [concrete = f64] {
// CHECK:STDOUT:       %int_64.loc43_47: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %float.make_type.loc43_47: init type = call constants.%Float(%int_64.loc43_47) [concrete = f64]
// CHECK:STDOUT:       %.loc43_47.2: type = value_of_initializer %float.make_type.loc43_47 [concrete = f64]
// CHECK:STDOUT:       %.loc43_47.3: type = converted %float.make_type.loc43_47, %.loc43_47.2 [concrete = f64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: f64 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew() -> f64;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany(%a.param_patt: f64, %b.param_patt: f64) -> f64;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BadReturnType(%a.param_patt: f64) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @JustRight(%a.param_patt: f64) -> f64 = "float.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValidTooFew(%a.param_patt: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooFew.ref: %TooFew.type = name_ref TooFew, file.%TooFew.decl [concrete = constants.%TooFew]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValidTooMany(%a.param_patt: f64, %b.param_patt: f64, %c.param_patt: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooMany.ref: %TooMany.type = name_ref TooMany, file.%TooMany.decl [concrete = constants.%TooMany]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: f64 = name_ref c, %c
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValidBadReturnType(%a.param_patt: f64, %b.param_patt: f64) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BadReturnType.ref: %BadReturnType.type = name_ref BadReturnType, file.%BadReturnType.decl [concrete = constants.%BadReturnType]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
