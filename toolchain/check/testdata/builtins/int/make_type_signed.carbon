// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- types.carbon

library "types" api;

fn Int(n: i32) -> type = "int.make_type_signed";

// --- use_types.carbon

library "uses_types" api;

import library "types";

fn F(n: Int(64)) -> Int(64) {
  return n;
}

fn G(n: Int(13)) -> Int(13) {
  return n;
}

fn Symbolic(N:! i32, x: Int(N)) -> Int(N) {
  return x;
}

// --- fail_zero_size.carbon

library "fail_zero_size" api;

import library "types";

// CHECK:STDERR: fail_zero_size.carbon:[[@LINE+4]]:8: ERROR: Integer type width of 0 is not positive.
// CHECK:STDERR: var n: Int(0);
// CHECK:STDERR:        ^~~~
// CHECK:STDERR:
var n: Int(0);

// --- fail_negative_size.carbon

library "fail_negative_size" api;

import library "types";

fn Negate(n: i32) -> i32 = "int.negate";

// CHECK:STDERR: fail_negative_size.carbon:[[@LINE+4]]:8: ERROR: Integer type width of -1 is not positive.
// CHECK:STDERR: var n: Int(Negate(1));
// CHECK:STDERR:        ^~~~
// CHECK:STDERR:
var n: Int(Negate(1));

// --- fail_oversized.carbon

library "fail_oversized" api;

import library "types";

// CHECK:STDERR: fail_oversized.carbon:[[@LINE+3]]:8: ERROR: Integer type width of 1000000000 is greater than the maximum supported width of 8388608.
// CHECK:STDERR: var m: Int(1000000000);
// CHECK:STDERR:        ^~~~
var m: Int(1000000000);

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Int = %Int
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Int: <function> = fn_decl @Int [template] {
// CHECK:STDOUT:     %n.loc4_8.1: i32 = param n
// CHECK:STDOUT:     @Int.%n: i32 = bind_name n, %n.loc4_8.1
// CHECK:STDOUT:     @Int.%return: ref type = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 13 [template]
// CHECK:STDOUT:   %.4: type = int_type signed, %.3 [template]
// CHECK:STDOUT:   %.5: type = int_type signed, @Symbolic.%N [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:     .G = %G
// CHECK:STDOUT:     .Symbolic = %Symbolic
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: <function> = import_ref ir2, inst+5, loc_13 [template = imports.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %Int.ref.loc6_9: <function> = name_ref Int, %import_ref [template = imports.%Int]
// CHECK:STDOUT:     %.loc6_13: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %.loc6_12: init type = call %Int.ref.loc6_9(%.loc6_13) [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_15.1: type = value_of_initializer %.loc6_12 [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_15.2: type = converted %.loc6_12, %.loc6_15.1 [template = constants.%.2]
// CHECK:STDOUT:     %n.loc6_6.1: i64 = param n
// CHECK:STDOUT:     @F.%n: i64 = bind_name n, %n.loc6_6.1
// CHECK:STDOUT:     %Int.ref.loc6_21: <function> = name_ref Int, %import_ref [template = imports.%Int]
// CHECK:STDOUT:     %.loc6_25: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %.loc6_24: init type = call %Int.ref.loc6_21(%.loc6_25) [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_27.1: type = value_of_initializer %.loc6_24 [template = constants.%.2]
// CHECK:STDOUT:     %.loc6_27.2: type = converted %.loc6_24, %.loc6_27.1 [template = constants.%.2]
// CHECK:STDOUT:     @F.%return: ref i64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {
// CHECK:STDOUT:     %Int.ref.loc10_9: <function> = name_ref Int, %import_ref [template = imports.%Int]
// CHECK:STDOUT:     %.loc10_13: i32 = int_literal 13 [template = constants.%.3]
// CHECK:STDOUT:     %.loc10_12: init type = call %Int.ref.loc10_9(%.loc10_13) [template = constants.%.4]
// CHECK:STDOUT:     %.loc10_15.1: type = value_of_initializer %.loc10_12 [template = constants.%.4]
// CHECK:STDOUT:     %.loc10_15.2: type = converted %.loc10_12, %.loc10_15.1 [template = constants.%.4]
// CHECK:STDOUT:     %n.loc10_6.1: i13 = param n
// CHECK:STDOUT:     @G.%n: i13 = bind_name n, %n.loc10_6.1
// CHECK:STDOUT:     %Int.ref.loc10_21: <function> = name_ref Int, %import_ref [template = imports.%Int]
// CHECK:STDOUT:     %.loc10_25: i32 = int_literal 13 [template = constants.%.3]
// CHECK:STDOUT:     %.loc10_24: init type = call %Int.ref.loc10_21(%.loc10_25) [template = constants.%.4]
// CHECK:STDOUT:     %.loc10_27.1: type = value_of_initializer %.loc10_24 [template = constants.%.4]
// CHECK:STDOUT:     %.loc10_27.2: type = converted %.loc10_24, %.loc10_27.1 [template = constants.%.4]
// CHECK:STDOUT:     @G.%return: ref i13 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Symbolic: <function> = fn_decl @Symbolic [template] {
// CHECK:STDOUT:     %N.loc14_13.1: i32 = param N
// CHECK:STDOUT:     @Symbolic.%N: i32 = bind_symbolic_name N, %N.loc14_13.1 [symbolic]
// CHECK:STDOUT:     %Int.ref.loc14_25: <function> = name_ref Int, %import_ref [template = imports.%Int]
// CHECK:STDOUT:     %N.ref.loc14_29: i32 = name_ref N, @Symbolic.%N [symbolic = @Symbolic.%N]
// CHECK:STDOUT:     %.loc14_28: init type = call %Int.ref.loc14_25(%N.ref.loc14_29) [symbolic = constants.%.5]
// CHECK:STDOUT:     %.loc14_30.1: type = value_of_initializer %.loc14_28 [symbolic = constants.%.5]
// CHECK:STDOUT:     %.loc14_30.2: type = converted %.loc14_28, %.loc14_30.1 [symbolic = constants.%.5]
// CHECK:STDOUT:     %x.loc14_22.1: Core.Int(N) = param x
// CHECK:STDOUT:     @Symbolic.%x: Core.Int(N) = bind_name x, %x.loc14_22.1
// CHECK:STDOUT:     %Int.ref.loc14_36: <function> = name_ref Int, %import_ref [template = imports.%Int]
// CHECK:STDOUT:     %N.ref.loc14_40: i32 = name_ref N, @Symbolic.%N [symbolic = @Symbolic.%N]
// CHECK:STDOUT:     %.loc14_39: init type = call %Int.ref.loc14_36(%N.ref.loc14_40) [symbolic = constants.%.5]
// CHECK:STDOUT:     %.loc14_41.1: type = value_of_initializer %.loc14_39 [symbolic = constants.%.5]
// CHECK:STDOUT:     %.loc14_41.2: type = converted %.loc14_39, %.loc14_41.1 [symbolic = constants.%.5]
// CHECK:STDOUT:     @Symbolic.%return: ref Core.Int(N) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n: i64) -> i64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i64 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n: i13) -> i13 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i13 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Symbolic(%N: i32, %x: Core.Int(N)) -> Core.Int(N) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: Core.Int(N) = name_ref x, %x
// CHECK:STDOUT:   return %x.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_zero_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: <function> = import_ref ir2, inst+5, loc_11 [template = imports.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Int.ref: <function> = name_ref Int, %import_ref [template = imports.%Int]
// CHECK:STDOUT:   %.loc10_12: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_11: init type = call %Int.ref(%.loc10_12) [template = <error>]
// CHECK:STDOUT:   %.loc10_13.1: type = value_of_initializer %.loc10_11 [template = <error>]
// CHECK:STDOUT:   %.loc10_13.2: type = converted %.loc10_11, %.loc10_13.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_negative_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Negate = %Negate
// CHECK:STDOUT:     .n = %n.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: <function> = import_ref ir2, inst+5, loc_23 [template = imports.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Negate: <function> = fn_decl @Negate [template] {
// CHECK:STDOUT:     %n.loc6_11.1: i32 = param n
// CHECK:STDOUT:     @Negate.%n: i32 = bind_name n, %n.loc6_11.1
// CHECK:STDOUT:     @Negate.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.ref: <function> = name_ref Int, %import_ref [template = imports.%Int]
// CHECK:STDOUT:   %Negate.ref: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc12_19: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc12_18: init i32 = call %Negate.ref(%.loc12_19) [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_11.1: i32 = value_of_initializer %.loc12_18 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_11.2: i32 = converted %.loc12_18, %.loc12_11.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_11.3: init type = call %Int.ref(%.loc12_11.2) [template = <error>]
// CHECK:STDOUT:   %.loc12_21.1: type = value_of_initializer %.loc12_11.3 [template = <error>]
// CHECK:STDOUT:   %.loc12_21.2: type = converted %.loc12_11.3, %.loc12_21.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n.loc12: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%n: i32) -> i32 = "int.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_oversized.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1000000000 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .m = %m
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: <function> = import_ref ir2, inst+5, loc_11 [template = imports.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Int.ref: <function> = name_ref Int, %import_ref [template = imports.%Int]
// CHECK:STDOUT:   %.loc9_12: i32 = int_literal 1000000000 [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_11: init type = call %Int.ref(%.loc9_12) [template = <error>]
// CHECK:STDOUT:   %.loc9_22.1: type = value_of_initializer %.loc9_11 [template = <error>]
// CHECK:STDOUT:   %.loc9_22.2: type = converted %.loc9_11, %.loc9_22.1 [template = <error>]
// CHECK:STDOUT:   %m.var: ref <error> = var m
// CHECK:STDOUT:   %m: ref <error> = bind_name m, %m.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
