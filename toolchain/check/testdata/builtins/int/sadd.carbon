// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/sadd.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/sadd.carbon

// --- int_add.carbon

fn Add(a: i32, b: i32) -> i32 = "int.sadd";

var arr: [i32; Add(1, 2)];
let arr_p: [i32; 3]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Add(a, b);
}

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd"
// CHECK:STDERR: fn TooFew(a: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew(a: i32) -> i32 = "int.sadd";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd"
// CHECK:STDERR: fn TooMany(a: i32, b: i32, c: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: i32, b: i32, c: i32) -> i32 = "int.sadd";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.sadd"
// CHECK:STDERR: fn BadReturnType(a: i32, b: i32) -> bool = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: i32, b: i32) -> bool = "int.sadd";
fn JustRight(a: i32, b: i32) -> i32 = "int.sadd";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:20: error: array bound is not a constant
// CHECK:STDERR: var too_few: [i32; TooFew(1)];
// CHECK:STDERR:                    ^~~~~~~
// CHECK:STDERR:
var too_few: [i32; TooFew(1)];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:21: error: array bound is not a constant
// CHECK:STDERR: var too_many: [i32; TooMany(1, 2, 3)];
// CHECK:STDERR:                     ^~~~~~~~
// CHECK:STDERR:
var too_many: [i32; TooMany(1, 2, 3)];
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:28: error: array bound is not a constant
// CHECK:STDERR: var bad_return_type: [i32; BadReturnType(1, 2)];
// CHECK:STDERR:                            ^~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_return_type: [i32; BadReturnType(1, 2)];

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:21: error: 3 argument(s) passed to function expecting 2 argument(s).
// CHECK:STDERR: var bad_call: [i32; JustRight(1, 2, 3)];
// CHECK:STDERR:                     ^~~~~~~~~~
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-21]]:1: note: calling function declared here
// CHECK:STDERR: fn JustRight(a: i32, b: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var bad_call: [i32; JustRight(1, 2, 3)];

fn RuntimeCallTooFew(a: i32) -> i32 {
  return TooFew(a);
}

fn RuntimeCallTooMany(a: i32, b: i32, c: i32) -> i32 {
  return TooMany(a, b, c);
}

fn RuntimeCallBadReturnType(a: i32, b: i32) -> bool {
  return BadReturnType(a, b);
}

// --- fail_overflow.carbon

package FailOverflow;

fn Add(a: i32, b: i32) -> i32 = "int.sadd";

let a: i32 = Add(0x7FFFFFFF, 0);
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+3]]:14: error: integer overflow in calculation 2147483647 + 1
// CHECK:STDERR: let b: i32 = Add(0x7FFFFFFF, 1);
// CHECK:STDERR:              ^~~~
let b: i32 = Add(0x7FFFFFFF, 1);

// CHECK:STDOUT: --- int_add.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Add.type: type = fn_type @Add [template]
// CHECK:STDOUT:   %Add: %Add.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, i32 [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
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
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Add.decl: %Add.type = fn_decl @Add [template = constants.%Add] {
// CHECK:STDOUT:     %int.make_type_32.loc2_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_11.1: type = value_of_initializer %int.make_type_32.loc2_11 [template = i32]
// CHECK:STDOUT:     %.loc2_11.2: type = converted %int.make_type_32.loc2_11, %.loc2_11.1 [template = i32]
// CHECK:STDOUT:     %a.loc2_8.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @Add.%a: i32 = bind_name a, %a.loc2_8.1
// CHECK:STDOUT:     %int.make_type_32.loc2_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_19.1: type = value_of_initializer %int.make_type_32.loc2_19 [template = i32]
// CHECK:STDOUT:     %.loc2_19.2: type = converted %int.make_type_32.loc2_19, %.loc2_19.1 [template = i32]
// CHECK:STDOUT:     %b.loc2_16.1: i32 = param b, runtime_param1
// CHECK:STDOUT:     @Add.%b: i32 = bind_name b, %b.loc2_16.1
// CHECK:STDOUT:     %int.make_type_32.loc2_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_27.1: type = value_of_initializer %int.make_type_32.loc2_27 [template = i32]
// CHECK:STDOUT:     %.loc2_27.2: type = converted %int.make_type_32.loc2_27, %.loc2_27.1 [template = i32]
// CHECK:STDOUT:     @Add.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %Add.ref: %Add.type = name_ref Add, %Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %.loc4_20: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_23: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %int.sadd: init i32 = call %Add.ref(%.loc4_20, %.loc4_23) [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_32.loc4, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:   %.loc4_25: type = array_type %int.sadd, i32 [template = constants.%.5]
// CHECK:STDOUT:   %arr.var: ref %.5 = var arr
// CHECK:STDOUT:   %arr: ref %.5 = bind_name arr, %arr.var
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_18: i32 = int_literal 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_13.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_13.2: type = converted %int.make_type_32.loc5, %.loc5_13.1 [template = i32]
// CHECK:STDOUT:   %.loc5_19: type = array_type %.loc5_18, i32 [template = constants.%.5]
// CHECK:STDOUT:   %.loc5_20: type = ptr_type %.5 [template = constants.%.6]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %int.make_type_32.loc7_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_19.1: type = value_of_initializer %int.make_type_32.loc7_19 [template = i32]
// CHECK:STDOUT:     %.loc7_19.2: type = converted %int.make_type_32.loc7_19, %.loc7_19.1 [template = i32]
// CHECK:STDOUT:     %a.loc7_16.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc7_16.1
// CHECK:STDOUT:     %int.make_type_32.loc7_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_27.1: type = value_of_initializer %int.make_type_32.loc7_27 [template = i32]
// CHECK:STDOUT:     %.loc7_27.2: type = converted %int.make_type_32.loc7_27, %.loc7_27.1 [template = i32]
// CHECK:STDOUT:     %b.loc7_24.1: i32 = param b, runtime_param1
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc7_24.1
// CHECK:STDOUT:     %int.make_type_32.loc7_35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_35.1: type = value_of_initializer %int.make_type_32.loc7_35 [template = i32]
// CHECK:STDOUT:     %.loc7_35.2: type = converted %int.make_type_32.loc7_35, %.loc7_35.1 [template = i32]
// CHECK:STDOUT:     @RuntimeCall.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a: i32, %b: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Add.ref: %Add.type = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %int.sadd: init i32 = call %Add.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19.1: i32 = value_of_initializer %int.sadd
// CHECK:STDOUT:   %.loc8_19.2: i32 = converted %int.sadd, %.loc8_19.1
// CHECK:STDOUT:   return %.loc8_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %.5 = name_ref arr, file.%arr
// CHECK:STDOUT:   %.loc5: %.6 = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %.6 = bind_name arr_p, %.loc5
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
// CHECK:STDOUT:   %.4: i32 = int_literal 3 [template]
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
// CHECK:STDOUT:     %int.make_type_32.loc8_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_14.1: type = value_of_initializer %int.make_type_32.loc8_14 [template = i32]
// CHECK:STDOUT:     %.loc8_14.2: type = converted %int.make_type_32.loc8_14, %.loc8_14.1 [template = i32]
// CHECK:STDOUT:     %a.loc8_11.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @TooFew.%a: i32 = bind_name a, %a.loc8_11.1
// CHECK:STDOUT:     %int.make_type_32.loc8_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_22.1: type = value_of_initializer %int.make_type_32.loc8_22 [template = i32]
// CHECK:STDOUT:     %.loc8_22.2: type = converted %int.make_type_32.loc8_22, %.loc8_22.1 [template = i32]
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
// CHECK:STDOUT:     %c.loc13_28.1: i32 = param c, runtime_param2
// CHECK:STDOUT:     @TooMany.%c: i32 = bind_name c, %c.loc13_28.1
// CHECK:STDOUT:     %int.make_type_32.loc13_39: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_39.1: type = value_of_initializer %int.make_type_32.loc13_39 [template = i32]
// CHECK:STDOUT:     %.loc13_39.2: type = converted %int.make_type_32.loc13_39, %.loc13_39.1 [template = i32]
// CHECK:STDOUT:     @TooMany.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BadReturnType.decl: %BadReturnType.type = fn_decl @BadReturnType [template = constants.%BadReturnType] {
// CHECK:STDOUT:     %int.make_type_32.loc18_21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_21.1: type = value_of_initializer %int.make_type_32.loc18_21 [template = i32]
// CHECK:STDOUT:     %.loc18_21.2: type = converted %int.make_type_32.loc18_21, %.loc18_21.1 [template = i32]
// CHECK:STDOUT:     %a.loc18_18.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @BadReturnType.%a: i32 = bind_name a, %a.loc18_18.1
// CHECK:STDOUT:     %int.make_type_32.loc18_29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_29.1: type = value_of_initializer %int.make_type_32.loc18_29 [template = i32]
// CHECK:STDOUT:     %.loc18_29.2: type = converted %int.make_type_32.loc18_29, %.loc18_29.1 [template = i32]
// CHECK:STDOUT:     %b.loc18_26.1: i32 = param b, runtime_param1
// CHECK:STDOUT:     @BadReturnType.%b: i32 = bind_name b, %b.loc18_26.1
// CHECK:STDOUT:     %bool.make_type.loc18: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc18_37.1: type = value_of_initializer %bool.make_type.loc18 [template = bool]
// CHECK:STDOUT:     %.loc18_37.2: type = converted %bool.make_type.loc18, %.loc18_37.1 [template = bool]
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
// CHECK:STDOUT:     %b.loc19_22.1: i32 = param b, runtime_param1
// CHECK:STDOUT:     @JustRight.%b: i32 = bind_name b, %b.loc19_22.1
// CHECK:STDOUT:     %int.make_type_32.loc19_33: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc19_33.1: type = value_of_initializer %int.make_type_32.loc19_33 [template = i32]
// CHECK:STDOUT:     %.loc19_33.2: type = converted %int.make_type_32.loc19_33, %.loc19_33.1 [template = i32]
// CHECK:STDOUT:     @JustRight.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %TooFew.ref: %TooFew.type = name_ref TooFew, %TooFew.decl [template = constants.%TooFew]
// CHECK:STDOUT:   %.loc25: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %TooFew.call: init i32 = call %TooFew.ref(%.loc25)
// CHECK:STDOUT:   %too_few.var: ref <error> = var too_few
// CHECK:STDOUT:   %too_few: ref <error> = bind_name too_few, %too_few.var
// CHECK:STDOUT:   %int.make_type_32.loc30: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %TooMany.ref: %TooMany.type = name_ref TooMany, %TooMany.decl [template = constants.%TooMany]
// CHECK:STDOUT:   %.loc30_29: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc30_32: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc30_35: i32 = int_literal 3 [template = constants.%.4]
// CHECK:STDOUT:   %TooMany.call: init i32 = call %TooMany.ref(%.loc30_29, %.loc30_32, %.loc30_35)
// CHECK:STDOUT:   %too_many.var: ref <error> = var too_many
// CHECK:STDOUT:   %too_many: ref <error> = bind_name too_many, %too_many.var
// CHECK:STDOUT:   %int.make_type_32.loc35: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %BadReturnType.ref: %BadReturnType.type = name_ref BadReturnType, %BadReturnType.decl [template = constants.%BadReturnType]
// CHECK:STDOUT:   %.loc35_42: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc35_45: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %BadReturnType.call: init bool = call %BadReturnType.ref(%.loc35_42, %.loc35_45)
// CHECK:STDOUT:   %bad_return_type.var: ref <error> = var bad_return_type
// CHECK:STDOUT:   %bad_return_type: ref <error> = bind_name bad_return_type, %bad_return_type.var
// CHECK:STDOUT:   %int.make_type_32.loc44: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %JustRight.ref: %JustRight.type = name_ref JustRight, %JustRight.decl [template = constants.%JustRight]
// CHECK:STDOUT:   %.loc44_31: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc44_34: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc44_37: i32 = int_literal 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc44_16.1: type = value_of_initializer %int.make_type_32.loc44 [template = i32]
// CHECK:STDOUT:   %.loc44_16.2: type = converted %int.make_type_32.loc44, %.loc44_16.1 [template = i32]
// CHECK:STDOUT:   %.loc44_39: type = array_type <error>, i32 [template = <error>]
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
// CHECK:STDOUT:     %int.make_type_32.loc50_26: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc50_26.1: type = value_of_initializer %int.make_type_32.loc50_26 [template = i32]
// CHECK:STDOUT:     %.loc50_26.2: type = converted %int.make_type_32.loc50_26, %.loc50_26.1 [template = i32]
// CHECK:STDOUT:     %a.loc50_23.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @RuntimeCallTooMany.%a: i32 = bind_name a, %a.loc50_23.1
// CHECK:STDOUT:     %int.make_type_32.loc50_34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc50_34.1: type = value_of_initializer %int.make_type_32.loc50_34 [template = i32]
// CHECK:STDOUT:     %.loc50_34.2: type = converted %int.make_type_32.loc50_34, %.loc50_34.1 [template = i32]
// CHECK:STDOUT:     %b.loc50_31.1: i32 = param b, runtime_param1
// CHECK:STDOUT:     @RuntimeCallTooMany.%b: i32 = bind_name b, %b.loc50_31.1
// CHECK:STDOUT:     %int.make_type_32.loc50_42: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc50_42.1: type = value_of_initializer %int.make_type_32.loc50_42 [template = i32]
// CHECK:STDOUT:     %.loc50_42.2: type = converted %int.make_type_32.loc50_42, %.loc50_42.1 [template = i32]
// CHECK:STDOUT:     %c.loc50_39.1: i32 = param c, runtime_param2
// CHECK:STDOUT:     @RuntimeCallTooMany.%c: i32 = bind_name c, %c.loc50_39.1
// CHECK:STDOUT:     %int.make_type_32.loc50_50: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc50_50.1: type = value_of_initializer %int.make_type_32.loc50_50 [template = i32]
// CHECK:STDOUT:     %.loc50_50.2: type = converted %int.make_type_32.loc50_50, %.loc50_50.1 [template = i32]
// CHECK:STDOUT:     @RuntimeCallTooMany.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallBadReturnType.decl: %RuntimeCallBadReturnType.type = fn_decl @RuntimeCallBadReturnType [template = constants.%RuntimeCallBadReturnType] {
// CHECK:STDOUT:     %int.make_type_32.loc54_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc54_32.1: type = value_of_initializer %int.make_type_32.loc54_32 [template = i32]
// CHECK:STDOUT:     %.loc54_32.2: type = converted %int.make_type_32.loc54_32, %.loc54_32.1 [template = i32]
// CHECK:STDOUT:     %a.loc54_29.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%a: i32 = bind_name a, %a.loc54_29.1
// CHECK:STDOUT:     %int.make_type_32.loc54_40: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc54_40.1: type = value_of_initializer %int.make_type_32.loc54_40 [template = i32]
// CHECK:STDOUT:     %.loc54_40.2: type = converted %int.make_type_32.loc54_40, %.loc54_40.1 [template = i32]
// CHECK:STDOUT:     %b.loc54_37.1: i32 = param b, runtime_param1
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%b: i32 = bind_name b, %b.loc54_37.1
// CHECK:STDOUT:     %bool.make_type.loc54: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc54_48.1: type = value_of_initializer %bool.make_type.loc54 [template = bool]
// CHECK:STDOUT:     %.loc54_48.2: type = converted %bool.make_type.loc54, %.loc54_48.1 [template = bool]
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew(%a: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany(%a: i32, %b: i32, %c: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BadReturnType(%a: i32, %b: i32) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @JustRight(%a: i32, %b: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooFew(%a: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooFew.ref: %TooFew.type = name_ref TooFew, file.%TooFew.decl [template = constants.%TooFew]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %TooFew.call: init i32 = call %TooFew.ref(%a.ref)
// CHECK:STDOUT:   %.loc47_19.1: i32 = value_of_initializer %TooFew.call
// CHECK:STDOUT:   %.loc47_19.2: i32 = converted %TooFew.call, %.loc47_19.1
// CHECK:STDOUT:   return %.loc47_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooMany(%a: i32, %b: i32, %c: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooMany.ref: %TooMany.type = name_ref TooMany, file.%TooMany.decl [template = constants.%TooMany]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: i32 = name_ref c, %c
// CHECK:STDOUT:   %TooMany.call: init i32 = call %TooMany.ref(%a.ref, %b.ref, %c.ref)
// CHECK:STDOUT:   %.loc51_26.1: i32 = value_of_initializer %TooMany.call
// CHECK:STDOUT:   %.loc51_26.2: i32 = converted %TooMany.call, %.loc51_26.1
// CHECK:STDOUT:   return %.loc51_26.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallBadReturnType(%a: i32, %b: i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BadReturnType.ref: %BadReturnType.type = name_ref BadReturnType, file.%BadReturnType.decl [template = constants.%BadReturnType]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %BadReturnType.call: init bool = call %BadReturnType.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc55_29.1: bool = value_of_initializer %BadReturnType.call
// CHECK:STDOUT:   %.loc55_29.2: bool = converted %BadReturnType.call, %.loc55_29.1
// CHECK:STDOUT:   return %.loc55_29.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Add.type: type = fn_type @Add [template]
// CHECK:STDOUT:   %Add: %Add.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2147483647 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
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
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Add.decl: %Add.type = fn_decl @Add [template = constants.%Add] {
// CHECK:STDOUT:     %int.make_type_32.loc4_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_11.1: type = value_of_initializer %int.make_type_32.loc4_11 [template = i32]
// CHECK:STDOUT:     %.loc4_11.2: type = converted %int.make_type_32.loc4_11, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:     %a.loc4_8.1: i32 = param a, runtime_param0
// CHECK:STDOUT:     @Add.%a: i32 = bind_name a, %a.loc4_8.1
// CHECK:STDOUT:     %int.make_type_32.loc4_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_19.1: type = value_of_initializer %int.make_type_32.loc4_19 [template = i32]
// CHECK:STDOUT:     %.loc4_19.2: type = converted %int.make_type_32.loc4_19, %.loc4_19.1 [template = i32]
// CHECK:STDOUT:     %b.loc4_16.1: i32 = param b, runtime_param1
// CHECK:STDOUT:     @Add.%b: i32 = bind_name b, %b.loc4_16.1
// CHECK:STDOUT:     %int.make_type_32.loc4_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32.loc4_27 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32.loc4_27, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     @Add.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_8.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_8.2: type = converted %int.make_type_32.loc6, %.loc6_8.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc10: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc10_8.1: type = value_of_initializer %int.make_type_32.loc10 [template = i32]
// CHECK:STDOUT:   %.loc10_8.2: type = converted %int.make_type_32.loc10, %.loc10_8.1 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a: i32, %b: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Add.ref.loc6: %Add.type = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %.loc6_18: i32 = int_literal 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_30: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %int.sadd.loc6: init i32 = call %Add.ref.loc6(%.loc6_18, %.loc6_30) [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_32.1: i32 = value_of_initializer %int.sadd.loc6 [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_32.2: i32 = converted %int.sadd.loc6, %.loc6_32.1 [template = constants.%.2]
// CHECK:STDOUT:   %a: i32 = bind_name a, %.loc6_32.2
// CHECK:STDOUT:   %Add.ref.loc10: %Add.type = name_ref Add, file.%Add.decl [template = constants.%Add]
// CHECK:STDOUT:   %.loc10_18: i32 = int_literal 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_30: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %int.sadd.loc10: init i32 = call %Add.ref.loc10(%.loc10_18, %.loc10_30) [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_32.1: i32 = value_of_initializer %int.sadd.loc10 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_32.2: i32 = converted %int.sadd.loc10, %.loc10_32.1 [template = constants.%.5]
// CHECK:STDOUT:   %b: i32 = bind_name b, %.loc10_32.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
