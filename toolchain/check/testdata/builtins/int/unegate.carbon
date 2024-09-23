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

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.unegate"
// CHECK:STDERR: fn TooFew() -> i32 = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew() -> i32 = "int.unegate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.unegate"
// CHECK:STDERR: fn TooMany(a: i32, b: i32) -> i32 = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: i32, b: i32) -> i32 = "int.unegate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.unegate"
// CHECK:STDERR: fn BadReturnType(a: i32) -> bool = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: i32) -> bool = "int.unegate";
fn JustRight(a: i32) -> i32 = "int.unegate";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:20: error: array bound is not a constant
// CHECK:STDERR: var too_few: [i32; TooFew()];
// CHECK:STDERR:                    ^~~~~~~
// CHECK:STDERR:
var too_few: [i32; TooFew()];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:21: error: array bound is not a constant
// CHECK:STDERR: var too_many: [i32; TooMany(1, 2)];
// CHECK:STDERR:                     ^~~~~~~~
// CHECK:STDERR:
var too_many: [i32; TooMany(1, 2)];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:28: error: array bound is not a constant
// CHECK:STDERR: var bad_return_type: [i32; BadReturnType(1)];
// CHECK:STDERR:                            ^~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_return_type: [i32; BadReturnType(1)];

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:21: error: 2 argument(s) passed to function expecting 1 argument(s).
// CHECK:STDERR: var bad_call: [i32; JustRight(1, 2)];
// CHECK:STDERR:                     ^~~~~~~~~~
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-21]]:1: note: calling function declared here
// CHECK:STDERR: fn JustRight(a: i32) -> i32 = "int.unegate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_call: [i32; JustRight(1, 2)];

fn RuntimeCallTooFew(a: i32) -> i32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 1 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   return TooFew(a);
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-42]]:1: note: calling function declared here
  // CHECK:STDERR: fn TooFew() -> i32 = "int.unegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooFew(a);
}

fn RuntimeCallTooMany(a: i32, b: i32, c: i32) -> i32 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: error: 3 argument(s) passed to function expecting 2 argument(s).
  // CHECK:STDERR:   return TooMany(a, b, c);
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-48]]:1: note: calling function declared here
  // CHECK:STDERR: fn TooMany(a: i32, b: i32) -> i32 = "int.unegate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooMany(a, b, c);
}

fn RuntimeCallBadReturnType(a: i32, b: i32) -> bool {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+6]]:10: error: 2 argument(s) passed to function expecting 1 argument(s).
  // CHECK:STDERR:   return BadReturnType(a, b);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-54]]:1: note: calling function declared here
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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 123 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal -123 [template]
// CHECK:STDOUT:   %.4: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal -1 [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %int.make_type_32.loc2_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_14.1: type = value_of_initializer %int.make_type_32.loc2_14 [template = i32]
// CHECK:STDOUT:     %.loc2_14.2: type = converted %int.make_type_32.loc2_14, %.loc2_14.1 [template = i32]
// CHECK:STDOUT:     %a.loc2_11.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @Negate.%a: i32 = bind_name a, %a.loc2_11.1
// CHECK:STDOUT:     %int.make_type_32.loc2_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_22.1: type = value_of_initializer %int.make_type_32.loc2_22 [template = i32]
// CHECK:STDOUT:     %.loc2_22.2: type = converted %int.make_type_32.loc2_22, %.loc2_22.1 [template = i32]
// CHECK:STDOUT:     @Negate.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %Negate.ref.loc4_16: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %Negate.ref.loc4_23: %Negate.type = name_ref Negate, %Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc4_30: i32 = int_literal 123 [template = constants.%.2]
// CHECK:STDOUT:   %int.unegate.loc4_29: init i32 = call %Negate.ref.loc4_23(%.loc4_30) [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_22.1: i32 = value_of_initializer %int.unegate.loc4_29 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_22.2: i32 = converted %int.unegate.loc4_29, %.loc4_22.1 [template = constants.%.3]
// CHECK:STDOUT:   %int.unegate.loc4_22: init i32 = call %Negate.ref.loc4_16(%.loc4_22.2) [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_32.loc4, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:   %.loc4_35: type = array_type %int.unegate.loc4_22, i32 [template = constants.%.4]
// CHECK:STDOUT:   %arr.var: ref %.4 = var arr
// CHECK:STDOUT:   %arr: ref %.4 = bind_name arr, %arr.var
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_18: i32 = int_literal 123 [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_13.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_13.2: type = converted %int.make_type_32.loc5, %.loc5_13.1 [template = i32]
// CHECK:STDOUT:   %.loc5_21: type = array_type %.loc5_18, i32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_22: type = ptr_type %.4 [template = constants.%.5]
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc7_8.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_8.2: type = converted %int.make_type_32.loc7, %.loc7_8.1 [template = i32]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %int.make_type_32.loc9_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc9_19.1: type = value_of_initializer %int.make_type_32.loc9_19 [template = i32]
// CHECK:STDOUT:     %.loc9_19.2: type = converted %int.make_type_32.loc9_19, %.loc9_19.1 [template = i32]
// CHECK:STDOUT:     %a.loc9_16.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc9_16.1
// CHECK:STDOUT:     %int.make_type_32.loc9_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc9_27.1: type = value_of_initializer %int.make_type_32.loc9_27 [template = i32]
// CHECK:STDOUT:     %.loc9_27.2: type = converted %int.make_type_32.loc9_27, %.loc9_27.1 [template = i32]
// CHECK:STDOUT:     %b.loc9_24.1: i32 = param b, runtime_param1
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc9_24.1
// CHECK:STDOUT:     %int.make_type_32.loc9_35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc9_35.1: type = value_of_initializer %int.make_type_32.loc9_35 [template = i32]
// CHECK:STDOUT:     %.loc9_35.2: type = converted %int.make_type_32.loc9_35, %.loc9_35.1 [template = i32]
// CHECK:STDOUT:     @RuntimeCall.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: i32) -> i32 = "int.unegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %int.unegate: init i32 = call %Negate.ref(%a.ref)
// CHECK:STDOUT:   %.loc10_19.1: i32 = value_of_initializer %int.unegate
// CHECK:STDOUT:   %.loc10_19.2: i32 = converted %int.unegate, %.loc10_19.1
// CHECK:STDOUT:   return %.loc10_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %.4 = name_ref arr, file.%arr
// CHECK:STDOUT:   %.loc5: %.5 = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %.5 = bind_name arr_p, %.loc5
// CHECK:STDOUT:   %Negate.ref: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc7_21: i32 = int_literal 1 [template = constants.%.6]
// CHECK:STDOUT:   %int.unegate: init i32 = call %Negate.ref(%.loc7_21) [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_23.1: i32 = value_of_initializer %int.unegate [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_23.2: i32 = converted %int.unegate, %.loc7_23.1 [template = constants.%.7]
// CHECK:STDOUT:   %n: i32 = bind_name n, %.loc7_23.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
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
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
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
// CHECK:STDOUT:     .Int32 = %import_ref.1
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
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %int.make_type_32.loc8: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_16.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:     %.loc8_16.2: type = converted %int.make_type_32.loc8, %.loc8_16.1 [template = i32]
// CHECK:STDOUT:     @TooFew.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany.decl: %TooMany.type = fn_decl @TooMany [template = constants.%TooMany] {
// CHECK:STDOUT:     %int.make_type_32.loc13_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_15.1: type = value_of_initializer %int.make_type_32.loc13_15 [template = i32]
// CHECK:STDOUT:     %.loc13_15.2: type = converted %int.make_type_32.loc13_15, %.loc13_15.1 [template = i32]
// CHECK:STDOUT:     %a.loc13_12.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @TooMany.%a: i32 = bind_name a, %a.loc13_12.1
// CHECK:STDOUT:     %int.make_type_32.loc13_23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_23.1: type = value_of_initializer %int.make_type_32.loc13_23 [template = i32]
// CHECK:STDOUT:     %.loc13_23.2: type = converted %int.make_type_32.loc13_23, %.loc13_23.1 [template = i32]
// CHECK:STDOUT:     %b.loc13_20.1: i32 = param b, runtime_param1
// CHECK:STDOUT:     @TooMany.%b: i32 = bind_name b, %b.loc13_20.1
// CHECK:STDOUT:     %int.make_type_32.loc13_31: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_31.1: type = value_of_initializer %int.make_type_32.loc13_31 [template = i32]
// CHECK:STDOUT:     %.loc13_31.2: type = converted %int.make_type_32.loc13_31, %.loc13_31.1 [template = i32]
// CHECK:STDOUT:     @TooMany.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BadReturnType.decl: %BadReturnType.type = fn_decl @BadReturnType [template = constants.%BadReturnType] {
// CHECK:STDOUT:     %int.make_type_32.loc18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_21.1: type = value_of_initializer %int.make_type_32.loc18 [template = i32]
// CHECK:STDOUT:     %.loc18_21.2: type = converted %int.make_type_32.loc18, %.loc18_21.1 [template = i32]
// CHECK:STDOUT:     %a.loc18_18.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @BadReturnType.%a: i32 = bind_name a, %a.loc18_18.1
// CHECK:STDOUT:     %bool.make_type.loc18: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc18_29.1: type = value_of_initializer %bool.make_type.loc18 [template = bool]
// CHECK:STDOUT:     %.loc18_29.2: type = converted %bool.make_type.loc18, %.loc18_29.1 [template = bool]
// CHECK:STDOUT:     @BadReturnType.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %JustRight.decl: %JustRight.type = fn_decl @JustRight [template = constants.%JustRight] {
// CHECK:STDOUT:     %int.make_type_32.loc19_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc19_17.1: type = value_of_initializer %int.make_type_32.loc19_17 [template = i32]
// CHECK:STDOUT:     %.loc19_17.2: type = converted %int.make_type_32.loc19_17, %.loc19_17.1 [template = i32]
// CHECK:STDOUT:     %a.loc19_14.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @JustRight.%a: i32 = bind_name a, %a.loc19_14.1
// CHECK:STDOUT:     %int.make_type_32.loc19_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc19_25.1: type = value_of_initializer %int.make_type_32.loc19_25 [template = i32]
// CHECK:STDOUT:     %.loc19_25.2: type = converted %int.make_type_32.loc19_25, %.loc19_25.1 [template = i32]
// CHECK:STDOUT:     @JustRight.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %TooFew.ref: %TooFew.type = name_ref TooFew, %TooFew.decl [template = constants.%TooFew]
// CHECK:STDOUT:   %TooFew.call: init i32 = call %TooFew.ref()
// CHECK:STDOUT:   %too_few.var: ref <error> = var too_few
// CHECK:STDOUT:   %too_few: ref <error> = bind_name too_few, %too_few.var
// CHECK:STDOUT:   %int.make_type_32.loc30: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %TooMany.ref: %TooMany.type = name_ref TooMany, %TooMany.decl [template = constants.%TooMany]
// CHECK:STDOUT:   %.loc30_29: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc30_32: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %TooMany.call: init i32 = call %TooMany.ref(%.loc30_29, %.loc30_32)
// CHECK:STDOUT:   %too_many.var: ref <error> = var too_many
// CHECK:STDOUT:   %too_many: ref <error> = bind_name too_many, %too_many.var
// CHECK:STDOUT:   %int.make_type_32.loc35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %BadReturnType.ref: %BadReturnType.type = name_ref BadReturnType, %BadReturnType.decl [template = constants.%BadReturnType]
// CHECK:STDOUT:   %.loc35: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %BadReturnType.call: init bool = call %BadReturnType.ref(%.loc35)
// CHECK:STDOUT:   %bad_return_type.var: ref <error> = var bad_return_type
// CHECK:STDOUT:   %bad_return_type: ref <error> = bind_name bad_return_type, %bad_return_type.var
// CHECK:STDOUT:   %int.make_type_32.loc44: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %JustRight.ref: %JustRight.type = name_ref JustRight, %JustRight.decl [template = constants.%JustRight]
// CHECK:STDOUT:   %.loc44_31: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc44_34: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc44_16.1: type = value_of_initializer %int.make_type_32.loc44 [template = i32]
// CHECK:STDOUT:   %.loc44_16.2: type = converted %int.make_type_32.loc44, %.loc44_16.1 [template = i32]
// CHECK:STDOUT:   %.loc44_36: type = array_type <error>, i32 [template = <error>]
// CHECK:STDOUT:   %bad_call.var: ref <error> = var bad_call
// CHECK:STDOUT:   %bad_call: ref <error> = bind_name bad_call, %bad_call.var
// CHECK:STDOUT:   %RuntimeCallTooFew.decl: %RuntimeCallTooFew.type = fn_decl @RuntimeCallTooFew [template = constants.%RuntimeCallTooFew] {
// CHECK:STDOUT:     %int.make_type_32.loc46_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc46_25.1: type = value_of_initializer %int.make_type_32.loc46_25 [template = i32]
// CHECK:STDOUT:     %.loc46_25.2: type = converted %int.make_type_32.loc46_25, %.loc46_25.1 [template = i32]
// CHECK:STDOUT:     %a.loc46_22.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @RuntimeCallTooFew.%a: i32 = bind_name a, %a.loc46_22.1
// CHECK:STDOUT:     %int.make_type_32.loc46_33: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc46_33.1: type = value_of_initializer %int.make_type_32.loc46_33 [template = i32]
// CHECK:STDOUT:     %.loc46_33.2: type = converted %int.make_type_32.loc46_33, %.loc46_33.1 [template = i32]
// CHECK:STDOUT:     @RuntimeCallTooFew.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooMany.decl: %RuntimeCallTooMany.type = fn_decl @RuntimeCallTooMany [template = constants.%RuntimeCallTooMany] {
// CHECK:STDOUT:     %int.make_type_32.loc57_26: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc57_26.1: type = value_of_initializer %int.make_type_32.loc57_26 [template = i32]
// CHECK:STDOUT:     %.loc57_26.2: type = converted %int.make_type_32.loc57_26, %.loc57_26.1 [template = i32]
// CHECK:STDOUT:     %a.loc57_23.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @RuntimeCallTooMany.%a: i32 = bind_name a, %a.loc57_23.1
// CHECK:STDOUT:     %int.make_type_32.loc57_34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc57_34.1: type = value_of_initializer %int.make_type_32.loc57_34 [template = i32]
// CHECK:STDOUT:     %.loc57_34.2: type = converted %int.make_type_32.loc57_34, %.loc57_34.1 [template = i32]
// CHECK:STDOUT:     %b.loc57_31.1: i32 = param b, runtime_param1
// CHECK:STDOUT:     @RuntimeCallTooMany.%b: i32 = bind_name b, %b.loc57_31.1
// CHECK:STDOUT:     %int.make_type_32.loc57_42: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc57_42.1: type = value_of_initializer %int.make_type_32.loc57_42 [template = i32]
// CHECK:STDOUT:     %.loc57_42.2: type = converted %int.make_type_32.loc57_42, %.loc57_42.1 [template = i32]
// CHECK:STDOUT:     %c.loc57_39.1: i32 = param c, runtime_param2
// CHECK:STDOUT:     @RuntimeCallTooMany.%c: i32 = bind_name c, %c.loc57_39.1
// CHECK:STDOUT:     %int.make_type_32.loc57_50: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc57_50.1: type = value_of_initializer %int.make_type_32.loc57_50 [template = i32]
// CHECK:STDOUT:     %.loc57_50.2: type = converted %int.make_type_32.loc57_50, %.loc57_50.1 [template = i32]
// CHECK:STDOUT:     @RuntimeCallTooMany.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallBadReturnType.decl: %RuntimeCallBadReturnType.type = fn_decl @RuntimeCallBadReturnType [template = constants.%RuntimeCallBadReturnType] {
// CHECK:STDOUT:     %int.make_type_32.loc68_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc68_32.1: type = value_of_initializer %int.make_type_32.loc68_32 [template = i32]
// CHECK:STDOUT:     %.loc68_32.2: type = converted %int.make_type_32.loc68_32, %.loc68_32.1 [template = i32]
// CHECK:STDOUT:     %a.loc68_29.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%a: i32 = bind_name a, %a.loc68_29.1
// CHECK:STDOUT:     %int.make_type_32.loc68_40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc68_40.1: type = value_of_initializer %int.make_type_32.loc68_40 [template = i32]
// CHECK:STDOUT:     %.loc68_40.2: type = converted %int.make_type_32.loc68_40, %.loc68_40.1 [template = i32]
// CHECK:STDOUT:     %b.loc68_37.1: i32 = param b, runtime_param1
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%b: i32 = bind_name b, %b.loc68_37.1
// CHECK:STDOUT:     %bool.make_type.loc68: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc68_48.1: type = value_of_initializer %bool.make_type.loc68 [template = bool]
// CHECK:STDOUT:     %.loc68_48.2: type = converted %bool.make_type.loc68, %.loc68_48.1 [template = bool]
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany(%a: i32, %b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BadReturnType(%a: i32) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @JustRight(%a: i32) -> i32 = "int.unegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooFew(%a: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooFew.ref: %TooFew.type = name_ref TooFew, file.%TooFew.decl [template = constants.%TooFew]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooMany(%a: i32, %b: i32, %c: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooMany.ref: %TooMany.type = name_ref TooMany, file.%TooMany.decl [template = constants.%TooMany]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: i32 = name_ref c, %c
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallBadReturnType(%a: i32, %b: i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BadReturnType.ref: %BadReturnType.type = name_ref BadReturnType, file.%BadReturnType.decl [template = constants.%BadReturnType]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %Sub.type: type = fn_type @Sub [template]
// CHECK:STDOUT:   %Sub: %Sub.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2147483647 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal -2147483647 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal -2147483648 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %int.make_type_32.loc4_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_14.1: type = value_of_initializer %int.make_type_32.loc4_14 [template = i32]
// CHECK:STDOUT:     %.loc4_14.2: type = converted %int.make_type_32.loc4_14, %.loc4_14.1 [template = i32]
// CHECK:STDOUT:     %a.loc4_11.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @Negate.%a: i32 = bind_name a, %a.loc4_11.1
// CHECK:STDOUT:     %int.make_type_32.loc4_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_22.1: type = value_of_initializer %int.make_type_32.loc4_22 [template = i32]
// CHECK:STDOUT:     %.loc4_22.2: type = converted %int.make_type_32.loc4_22, %.loc4_22.1 [template = i32]
// CHECK:STDOUT:     @Negate.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Sub.decl: %Sub.type = fn_decl @Sub [template = constants.%Sub] {
// CHECK:STDOUT:     %int.make_type_32.loc5_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_11.1: type = value_of_initializer %int.make_type_32.loc5_11 [template = i32]
// CHECK:STDOUT:     %.loc5_11.2: type = converted %int.make_type_32.loc5_11, %.loc5_11.1 [template = i32]
// CHECK:STDOUT:     %a.loc5_8.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @Sub.%a: i32 = bind_name a, %a.loc5_8.1
// CHECK:STDOUT:     %int.make_type_32.loc5_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_19.1: type = value_of_initializer %int.make_type_32.loc5_19 [template = i32]
// CHECK:STDOUT:     %.loc5_19.2: type = converted %int.make_type_32.loc5_19, %.loc5_19.1 [template = i32]
// CHECK:STDOUT:     %b.loc5_16.1: i32 = param b, runtime_param1
// CHECK:STDOUT:     @Sub.%b: i32 = bind_name b, %b.loc5_16.1
// CHECK:STDOUT:     %int.make_type_32.loc5_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_27.1: type = value_of_initializer %int.make_type_32.loc5_27 [template = i32]
// CHECK:STDOUT:     %.loc5_27.2: type = converted %int.make_type_32.loc5_27, %.loc5_27.1 [template = i32]
// CHECK:STDOUT:     @Sub.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc8: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_8.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:   %.loc8_8.2: type = converted %int.make_type_32.loc8, %.loc8_8.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_8.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_8.2: type = converted %int.make_type_32.loc11, %.loc11_8.1 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: i32) -> i32 = "int.unegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a: i32, %b: i32) -> i32 = "int.usub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref.loc8_14: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %Negate.ref.loc8_21: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc8_28: i32 = int_literal 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %int.unegate.loc8_27: init i32 = call %Negate.ref.loc8_21(%.loc8_28) [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_20.1: i32 = value_of_initializer %int.unegate.loc8_27 [template = constants.%.3]
// CHECK:STDOUT:   %.loc8_20.2: i32 = converted %int.unegate.loc8_27, %.loc8_20.1 [template = constants.%.3]
// CHECK:STDOUT:   %int.unegate.loc8_20: init i32 = call %Negate.ref.loc8_14(%.loc8_20.2) [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_40.1: i32 = value_of_initializer %int.unegate.loc8_20 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_40.2: i32 = converted %int.unegate.loc8_20, %.loc8_40.1 [template = constants.%.2]
// CHECK:STDOUT:   %a: i32 = bind_name a, %.loc8_40.2
// CHECK:STDOUT:   %Negate.ref.loc11_14: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %Sub.ref: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Negate.ref.loc11_25: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc11_32: i32 = int_literal 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %int.unegate.loc11_31: init i32 = call %Negate.ref.loc11_25(%.loc11_32) [template = constants.%.3]
// CHECK:STDOUT:   %.loc11_45: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_24.1: i32 = value_of_initializer %int.unegate.loc11_31 [template = constants.%.3]
// CHECK:STDOUT:   %.loc11_24.2: i32 = converted %int.unegate.loc11_31, %.loc11_24.1 [template = constants.%.3]
// CHECK:STDOUT:   %int.usub: init i32 = call %Sub.ref(%.loc11_24.2, %.loc11_45) [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_20.1: i32 = value_of_initializer %int.usub [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_20.2: i32 = converted %int.usub, %.loc11_20.1 [template = constants.%.5]
// CHECK:STDOUT:   %int.unegate.loc11_20: init i32 = call %Negate.ref.loc11_14(%.loc11_20.2) [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_48.1: i32 = value_of_initializer %int.unegate.loc11_20 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_48.2: i32 = converted %int.unegate.loc11_20, %.loc11_48.1 [template = constants.%.5]
// CHECK:STDOUT:   %b: i32 = bind_name b, %.loc11_48.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
