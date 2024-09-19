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

fn RuntimeCall(a: f64, b: f64) -> f64 {
  return Negate(a);
}

let a: f64 = Negate(1.5);

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.negate"
// CHECK:STDERR: fn TooFew() -> f64 = "float.negate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew() -> f64 = "float.negate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.negate"
// CHECK:STDERR: fn TooMany(a: f64, b: f64) -> f64 = "float.negate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: f64, b: f64) -> f64 = "float.negate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "float.negate"
// CHECK:STDERR: fn BadReturnType(a: f64) -> bool = "float.negate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: f64) -> bool = "float.negate";
fn JustRight(a: f64) -> f64 = "float.negate";

fn RuntimeCallTooFew(a: f64) -> f64 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 1 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   return TooFew(a);
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-17]]:1: calling function declared here
  // CHECK:STDERR: fn TooFew() -> f64 = "float.negate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooFew(a);
}

fn RuntimeCallTooMany(a: f64, b: f64, c: f64) -> f64 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 3 argument(s) passed to function expecting 2 argument(s).
  // CHECK:STDERR:   return TooMany(a, b, c);
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-23]]:1: calling function declared here
  // CHECK:STDERR: fn TooMany(a: f64, b: f64) -> f64 = "float.negate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooMany(a, b, c);
}

fn RuntimeCallBadReturnType(a: f64, b: f64) -> bool {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+6]]:10: error: 2 argument(s) passed to function expecting 1 argument(s).
  // CHECK:STDERR:   return BadReturnType(a, b);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-29]]:1: calling function declared here
  // CHECK:STDERR: fn BadReturnType(a: f64) -> bool = "float.negate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return BadReturnType(a, b);
}

// CHECK:STDOUT: --- float_negate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT:   %.3: f64 = float_literal 1.5 [template]
// CHECK:STDOUT:   %.4: f64 = float_literal -1.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Float = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Float.type = import_ref Core//prelude/types, inst+32, loaded [template = constants.%Float]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %.loc2_14.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc2_14: init type = call constants.%Float(%.loc2_14.1) [template = f64]
// CHECK:STDOUT:     %.loc2_14.2: type = value_of_initializer %float.make_type.loc2_14 [template = f64]
// CHECK:STDOUT:     %.loc2_14.3: type = converted %float.make_type.loc2_14, %.loc2_14.2 [template = f64]
// CHECK:STDOUT:     %a.loc2_11.1: f64 = param a, runtime_param0
// CHECK:STDOUT:     @Negate.%a: f64 = bind_name a, %a.loc2_11.1
// CHECK:STDOUT:     %.loc2_22.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc2_22: init type = call constants.%Float(%.loc2_22.1) [template = f64]
// CHECK:STDOUT:     %.loc2_22.2: type = value_of_initializer %float.make_type.loc2_22 [template = f64]
// CHECK:STDOUT:     %.loc2_22.3: type = converted %float.make_type.loc2_22, %.loc2_22.2 [template = f64]
// CHECK:STDOUT:     @Negate.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %.loc4_19.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc4_19: init type = call constants.%Float(%.loc4_19.1) [template = f64]
// CHECK:STDOUT:     %.loc4_19.2: type = value_of_initializer %float.make_type.loc4_19 [template = f64]
// CHECK:STDOUT:     %.loc4_19.3: type = converted %float.make_type.loc4_19, %.loc4_19.2 [template = f64]
// CHECK:STDOUT:     %a.loc4_16.1: f64 = param a, runtime_param0
// CHECK:STDOUT:     @RuntimeCall.%a: f64 = bind_name a, %a.loc4_16.1
// CHECK:STDOUT:     %.loc4_27.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc4_27: init type = call constants.%Float(%.loc4_27.1) [template = f64]
// CHECK:STDOUT:     %.loc4_27.2: type = value_of_initializer %float.make_type.loc4_27 [template = f64]
// CHECK:STDOUT:     %.loc4_27.3: type = converted %float.make_type.loc4_27, %.loc4_27.2 [template = f64]
// CHECK:STDOUT:     %b.loc4_24.1: f64 = param b, runtime_param1
// CHECK:STDOUT:     @RuntimeCall.%b: f64 = bind_name b, %b.loc4_24.1
// CHECK:STDOUT:     %.loc4_35.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc4_35: init type = call constants.%Float(%.loc4_35.1) [template = f64]
// CHECK:STDOUT:     %.loc4_35.2: type = value_of_initializer %float.make_type.loc4_35 [template = f64]
// CHECK:STDOUT:     %.loc4_35.3: type = converted %float.make_type.loc4_35, %.loc4_35.2 [template = f64]
// CHECK:STDOUT:     @RuntimeCall.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_8.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:   %float.make_type.loc8: init type = call constants.%Float(%.loc8_8.1) [template = f64]
// CHECK:STDOUT:   %.loc8_8.2: type = value_of_initializer %float.make_type.loc8 [template = f64]
// CHECK:STDOUT:   %.loc8_8.3: type = converted %float.make_type.loc8, %.loc8_8.2 [template = f64]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: f64) -> f64 = "float.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: f64, %b: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %float.negate: init f64 = call %Negate.ref(%a.ref)
// CHECK:STDOUT:   %.loc5_19.1: f64 = value_of_initializer %float.negate
// CHECK:STDOUT:   %.loc5_19.2: f64 = converted %float.negate, %.loc5_19.1
// CHECK:STDOUT:   return %.loc5_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc8_21: f64 = float_literal 1.5 [template = constants.%.3]
// CHECK:STDOUT:   %float.negate: init f64 = call %Negate.ref(%.loc8_21) [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_25.1: f64 = value_of_initializer %float.negate [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_25.2: f64 = converted %float.negate, %.loc8_25.1 [template = constants.%.4]
// CHECK:STDOUT:   %a: f64 = bind_name a, %.loc8_25.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
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
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Float.type = import_ref Core//prelude/types, inst+32, loaded [template = constants.%Float]
// CHECK:STDOUT:   %import_ref.2: %Bool.type = import_ref Core//prelude/types/bool, inst+2, loaded [template = constants.%Bool]
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
// CHECK:STDOUT:     %.loc8_16.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc8: init type = call constants.%Float(%.loc8_16.1) [template = f64]
// CHECK:STDOUT:     %.loc8_16.2: type = value_of_initializer %float.make_type.loc8 [template = f64]
// CHECK:STDOUT:     %.loc8_16.3: type = converted %float.make_type.loc8, %.loc8_16.2 [template = f64]
// CHECK:STDOUT:     @TooFew.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany.decl: %TooMany.type = fn_decl @TooMany [template = constants.%TooMany] {
// CHECK:STDOUT:     %.loc13_15.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc13_15: init type = call constants.%Float(%.loc13_15.1) [template = f64]
// CHECK:STDOUT:     %.loc13_15.2: type = value_of_initializer %float.make_type.loc13_15 [template = f64]
// CHECK:STDOUT:     %.loc13_15.3: type = converted %float.make_type.loc13_15, %.loc13_15.2 [template = f64]
// CHECK:STDOUT:     %a.loc13_12.1: f64 = param a, runtime_param0
// CHECK:STDOUT:     @TooMany.%a: f64 = bind_name a, %a.loc13_12.1
// CHECK:STDOUT:     %.loc13_23.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc13_23: init type = call constants.%Float(%.loc13_23.1) [template = f64]
// CHECK:STDOUT:     %.loc13_23.2: type = value_of_initializer %float.make_type.loc13_23 [template = f64]
// CHECK:STDOUT:     %.loc13_23.3: type = converted %float.make_type.loc13_23, %.loc13_23.2 [template = f64]
// CHECK:STDOUT:     %b.loc13_20.1: f64 = param b, runtime_param1
// CHECK:STDOUT:     @TooMany.%b: f64 = bind_name b, %b.loc13_20.1
// CHECK:STDOUT:     %.loc13_31.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc13_31: init type = call constants.%Float(%.loc13_31.1) [template = f64]
// CHECK:STDOUT:     %.loc13_31.2: type = value_of_initializer %float.make_type.loc13_31 [template = f64]
// CHECK:STDOUT:     %.loc13_31.3: type = converted %float.make_type.loc13_31, %.loc13_31.2 [template = f64]
// CHECK:STDOUT:     @TooMany.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BadReturnType.decl: %BadReturnType.type = fn_decl @BadReturnType [template = constants.%BadReturnType] {
// CHECK:STDOUT:     %.loc18_21.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc18: init type = call constants.%Float(%.loc18_21.1) [template = f64]
// CHECK:STDOUT:     %.loc18_21.2: type = value_of_initializer %float.make_type.loc18 [template = f64]
// CHECK:STDOUT:     %.loc18_21.3: type = converted %float.make_type.loc18, %.loc18_21.2 [template = f64]
// CHECK:STDOUT:     %a.loc18_18.1: f64 = param a, runtime_param0
// CHECK:STDOUT:     @BadReturnType.%a: f64 = bind_name a, %a.loc18_18.1
// CHECK:STDOUT:     %bool.make_type.loc18: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc18_29.1: type = value_of_initializer %bool.make_type.loc18 [template = bool]
// CHECK:STDOUT:     %.loc18_29.2: type = converted %bool.make_type.loc18, %.loc18_29.1 [template = bool]
// CHECK:STDOUT:     @BadReturnType.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %JustRight.decl: %JustRight.type = fn_decl @JustRight [template = constants.%JustRight] {
// CHECK:STDOUT:     %.loc19_17.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc19_17: init type = call constants.%Float(%.loc19_17.1) [template = f64]
// CHECK:STDOUT:     %.loc19_17.2: type = value_of_initializer %float.make_type.loc19_17 [template = f64]
// CHECK:STDOUT:     %.loc19_17.3: type = converted %float.make_type.loc19_17, %.loc19_17.2 [template = f64]
// CHECK:STDOUT:     %a.loc19_14.1: f64 = param a, runtime_param0
// CHECK:STDOUT:     @JustRight.%a: f64 = bind_name a, %a.loc19_14.1
// CHECK:STDOUT:     %.loc19_25.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc19_25: init type = call constants.%Float(%.loc19_25.1) [template = f64]
// CHECK:STDOUT:     %.loc19_25.2: type = value_of_initializer %float.make_type.loc19_25 [template = f64]
// CHECK:STDOUT:     %.loc19_25.3: type = converted %float.make_type.loc19_25, %.loc19_25.2 [template = f64]
// CHECK:STDOUT:     @JustRight.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooFew.decl: %RuntimeCallTooFew.type = fn_decl @RuntimeCallTooFew [template = constants.%RuntimeCallTooFew] {
// CHECK:STDOUT:     %.loc21_25.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc21_25: init type = call constants.%Float(%.loc21_25.1) [template = f64]
// CHECK:STDOUT:     %.loc21_25.2: type = value_of_initializer %float.make_type.loc21_25 [template = f64]
// CHECK:STDOUT:     %.loc21_25.3: type = converted %float.make_type.loc21_25, %.loc21_25.2 [template = f64]
// CHECK:STDOUT:     %a.loc21_22.1: f64 = param a, runtime_param0
// CHECK:STDOUT:     @RuntimeCallTooFew.%a: f64 = bind_name a, %a.loc21_22.1
// CHECK:STDOUT:     %.loc21_33.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc21_33: init type = call constants.%Float(%.loc21_33.1) [template = f64]
// CHECK:STDOUT:     %.loc21_33.2: type = value_of_initializer %float.make_type.loc21_33 [template = f64]
// CHECK:STDOUT:     %.loc21_33.3: type = converted %float.make_type.loc21_33, %.loc21_33.2 [template = f64]
// CHECK:STDOUT:     @RuntimeCallTooFew.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooMany.decl: %RuntimeCallTooMany.type = fn_decl @RuntimeCallTooMany [template = constants.%RuntimeCallTooMany] {
// CHECK:STDOUT:     %.loc32_26.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc32_26: init type = call constants.%Float(%.loc32_26.1) [template = f64]
// CHECK:STDOUT:     %.loc32_26.2: type = value_of_initializer %float.make_type.loc32_26 [template = f64]
// CHECK:STDOUT:     %.loc32_26.3: type = converted %float.make_type.loc32_26, %.loc32_26.2 [template = f64]
// CHECK:STDOUT:     %a.loc32_23.1: f64 = param a, runtime_param0
// CHECK:STDOUT:     @RuntimeCallTooMany.%a: f64 = bind_name a, %a.loc32_23.1
// CHECK:STDOUT:     %.loc32_34.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc32_34: init type = call constants.%Float(%.loc32_34.1) [template = f64]
// CHECK:STDOUT:     %.loc32_34.2: type = value_of_initializer %float.make_type.loc32_34 [template = f64]
// CHECK:STDOUT:     %.loc32_34.3: type = converted %float.make_type.loc32_34, %.loc32_34.2 [template = f64]
// CHECK:STDOUT:     %b.loc32_31.1: f64 = param b, runtime_param1
// CHECK:STDOUT:     @RuntimeCallTooMany.%b: f64 = bind_name b, %b.loc32_31.1
// CHECK:STDOUT:     %.loc32_42.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc32_42: init type = call constants.%Float(%.loc32_42.1) [template = f64]
// CHECK:STDOUT:     %.loc32_42.2: type = value_of_initializer %float.make_type.loc32_42 [template = f64]
// CHECK:STDOUT:     %.loc32_42.3: type = converted %float.make_type.loc32_42, %.loc32_42.2 [template = f64]
// CHECK:STDOUT:     %c.loc32_39.1: f64 = param c, runtime_param2
// CHECK:STDOUT:     @RuntimeCallTooMany.%c: f64 = bind_name c, %c.loc32_39.1
// CHECK:STDOUT:     %.loc32_50.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc32_50: init type = call constants.%Float(%.loc32_50.1) [template = f64]
// CHECK:STDOUT:     %.loc32_50.2: type = value_of_initializer %float.make_type.loc32_50 [template = f64]
// CHECK:STDOUT:     %.loc32_50.3: type = converted %float.make_type.loc32_50, %.loc32_50.2 [template = f64]
// CHECK:STDOUT:     @RuntimeCallTooMany.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallBadReturnType.decl: %RuntimeCallBadReturnType.type = fn_decl @RuntimeCallBadReturnType [template = constants.%RuntimeCallBadReturnType] {
// CHECK:STDOUT:     %.loc43_32.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc43_32: init type = call constants.%Float(%.loc43_32.1) [template = f64]
// CHECK:STDOUT:     %.loc43_32.2: type = value_of_initializer %float.make_type.loc43_32 [template = f64]
// CHECK:STDOUT:     %.loc43_32.3: type = converted %float.make_type.loc43_32, %.loc43_32.2 [template = f64]
// CHECK:STDOUT:     %a.loc43_29.1: f64 = param a, runtime_param0
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%a: f64 = bind_name a, %a.loc43_29.1
// CHECK:STDOUT:     %.loc43_40.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc43_40: init type = call constants.%Float(%.loc43_40.1) [template = f64]
// CHECK:STDOUT:     %.loc43_40.2: type = value_of_initializer %float.make_type.loc43_40 [template = f64]
// CHECK:STDOUT:     %.loc43_40.3: type = converted %float.make_type.loc43_40, %.loc43_40.2 [template = f64]
// CHECK:STDOUT:     %b.loc43_37.1: f64 = param b, runtime_param1
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%b: f64 = bind_name b, %b.loc43_37.1
// CHECK:STDOUT:     %bool.make_type.loc43: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc43_48.1: type = value_of_initializer %bool.make_type.loc43 [template = bool]
// CHECK:STDOUT:     %.loc43_48.2: type = converted %bool.make_type.loc43, %.loc43_48.1 [template = bool]
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew() -> f64;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany(%a: f64, %b: f64) -> f64;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BadReturnType(%a: f64) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @JustRight(%a: f64) -> f64 = "float.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooFew(%a: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooFew.ref: %TooFew.type = name_ref TooFew, file.%TooFew.decl [template = constants.%TooFew]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooMany(%a: f64, %b: f64, %c: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooMany.ref: %TooMany.type = name_ref TooMany, file.%TooMany.decl [template = constants.%TooMany]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: f64 = name_ref c, %c
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallBadReturnType(%a: f64, %b: f64) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BadReturnType.ref: %BadReturnType.type = name_ref BadReturnType, file.%BadReturnType.decl [template = constants.%BadReturnType]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
