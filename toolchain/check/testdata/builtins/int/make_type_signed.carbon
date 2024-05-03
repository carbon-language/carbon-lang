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

fn Negate(n: i32) -> i32 = "int.snegate";

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
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int: type = fn_type @Int [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Int.decl: Int = fn_decl @Int [template = constants.%struct] {
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
// CHECK:STDOUT:   %Int: type = fn_type @Int [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %.3: type = int_type signed, %.2 [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 13 [template]
// CHECK:STDOUT:   %.5: type = int_type signed, %.4 [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.3: G = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 0 [symbolic]
// CHECK:STDOUT:   %.6: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Symbolic: type = fn_type @Symbolic [template]
// CHECK:STDOUT:   %struct.4: Symbolic = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .Symbolic = %Symbolic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: Int = import_ref ir1, inst+5, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %Int.ref.loc6_9: Int = name_ref Int, %import_ref [template = constants.%struct.1]
// CHECK:STDOUT:     %.loc6_13: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed.loc6_12: init type = call %Int.ref.loc6_9(%.loc6_13) [template = constants.%.3]
// CHECK:STDOUT:     %.loc6_15.1: type = value_of_initializer %int.make_type_signed.loc6_12 [template = constants.%.3]
// CHECK:STDOUT:     %.loc6_15.2: type = converted %int.make_type_signed.loc6_12, %.loc6_15.1 [template = constants.%.3]
// CHECK:STDOUT:     %n.loc6_6.1: i64 = param n
// CHECK:STDOUT:     @F.%n: i64 = bind_name n, %n.loc6_6.1
// CHECK:STDOUT:     %Int.ref.loc6_21: Int = name_ref Int, %import_ref [template = constants.%struct.1]
// CHECK:STDOUT:     %.loc6_25: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_signed.loc6_24: init type = call %Int.ref.loc6_21(%.loc6_25) [template = constants.%.3]
// CHECK:STDOUT:     %.loc6_27.1: type = value_of_initializer %int.make_type_signed.loc6_24 [template = constants.%.3]
// CHECK:STDOUT:     %.loc6_27.2: type = converted %int.make_type_signed.loc6_24, %.loc6_27.1 [template = constants.%.3]
// CHECK:STDOUT:     @F.%return: ref i64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.3] {
// CHECK:STDOUT:     %Int.ref.loc10_9: Int = name_ref Int, %import_ref [template = constants.%struct.1]
// CHECK:STDOUT:     %.loc10_13: i32 = int_literal 13 [template = constants.%.4]
// CHECK:STDOUT:     %int.make_type_signed.loc10_12: init type = call %Int.ref.loc10_9(%.loc10_13) [template = constants.%.5]
// CHECK:STDOUT:     %.loc10_15.1: type = value_of_initializer %int.make_type_signed.loc10_12 [template = constants.%.5]
// CHECK:STDOUT:     %.loc10_15.2: type = converted %int.make_type_signed.loc10_12, %.loc10_15.1 [template = constants.%.5]
// CHECK:STDOUT:     %n.loc10_6.1: i13 = param n
// CHECK:STDOUT:     @G.%n: i13 = bind_name n, %n.loc10_6.1
// CHECK:STDOUT:     %Int.ref.loc10_21: Int = name_ref Int, %import_ref [template = constants.%struct.1]
// CHECK:STDOUT:     %.loc10_25: i32 = int_literal 13 [template = constants.%.4]
// CHECK:STDOUT:     %int.make_type_signed.loc10_24: init type = call %Int.ref.loc10_21(%.loc10_25) [template = constants.%.5]
// CHECK:STDOUT:     %.loc10_27.1: type = value_of_initializer %int.make_type_signed.loc10_24 [template = constants.%.5]
// CHECK:STDOUT:     %.loc10_27.2: type = converted %int.make_type_signed.loc10_24, %.loc10_27.1 [template = constants.%.5]
// CHECK:STDOUT:     @G.%return: ref i13 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Symbolic.decl: Symbolic = fn_decl @Symbolic [template = constants.%struct.4] {
// CHECK:STDOUT:     %N.loc14_13.1: i32 = param N
// CHECK:STDOUT:     @Symbolic.%N: i32 = bind_symbolic_name N 0, %N.loc14_13.1 [symbolic = constants.%N]
// CHECK:STDOUT:     %Int.ref.loc14_25: Int = name_ref Int, %import_ref [template = constants.%struct.1]
// CHECK:STDOUT:     %N.ref.loc14_29: i32 = name_ref N, @Symbolic.%N [symbolic = constants.%N]
// CHECK:STDOUT:     %int.make_type_signed.loc14_28: init type = call %Int.ref.loc14_25(%N.ref.loc14_29) [symbolic = constants.%.6]
// CHECK:STDOUT:     %.loc14_30.1: type = value_of_initializer %int.make_type_signed.loc14_28 [symbolic = constants.%.6]
// CHECK:STDOUT:     %.loc14_30.2: type = converted %int.make_type_signed.loc14_28, %.loc14_30.1 [symbolic = constants.%.6]
// CHECK:STDOUT:     %x.loc14_22.1: Core.Int(N) = param x
// CHECK:STDOUT:     @Symbolic.%x: Core.Int(N) = bind_name x, %x.loc14_22.1
// CHECK:STDOUT:     %Int.ref.loc14_36: Int = name_ref Int, %import_ref [template = constants.%struct.1]
// CHECK:STDOUT:     %N.ref.loc14_40: i32 = name_ref N, @Symbolic.%N [symbolic = constants.%N]
// CHECK:STDOUT:     %int.make_type_signed.loc14_39: init type = call %Int.ref.loc14_36(%N.ref.loc14_40) [symbolic = constants.%.6]
// CHECK:STDOUT:     %.loc14_41.1: type = value_of_initializer %int.make_type_signed.loc14_39 [symbolic = constants.%.6]
// CHECK:STDOUT:     %.loc14_41.2: type = converted %int.make_type_signed.loc14_39, %.loc14_41.1 [symbolic = constants.%.6]
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
// CHECK:STDOUT:   %Int: type = fn_type @Int [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: Int = import_ref ir1, inst+5, loaded [template = constants.%struct]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Int.ref: Int = name_ref Int, %import_ref [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_12: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_signed: init type = call %Int.ref(%.loc10_12) [template = <error>]
// CHECK:STDOUT:   %.loc10_13.1: type = value_of_initializer %int.make_type_signed [template = <error>]
// CHECK:STDOUT:   %.loc10_13.2: type = converted %int.make_type_signed, %.loc10_13.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_negative_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Negate: type = fn_type @Negate [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Negate = struct_value () [template]
// CHECK:STDOUT:   %Int: type = fn_type @Int [template]
// CHECK:STDOUT:   %struct.2: Int = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .n = %n.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: Int = import_ref ir1, inst+5, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Negate.decl: Negate = fn_decl @Negate [template = constants.%struct.1] {
// CHECK:STDOUT:     %n.loc6_11.1: i32 = param n
// CHECK:STDOUT:     @Negate.%n: i32 = bind_name n, %n.loc6_11.1
// CHECK:STDOUT:     @Negate.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.ref: Int = name_ref Int, %import_ref [template = constants.%struct.2]
// CHECK:STDOUT:   %Negate.ref: Negate = name_ref Negate, %Negate.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc12_19: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %int.snegate: init i32 = call %Negate.ref(%.loc12_19) [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_11.1: i32 = value_of_initializer %int.snegate [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_11.2: i32 = converted %int.snegate, %.loc12_11.1 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_signed: init type = call %Int.ref(%.loc12_11.2) [template = <error>]
// CHECK:STDOUT:   %.loc12_21.1: type = value_of_initializer %int.make_type_signed [template = <error>]
// CHECK:STDOUT:   %.loc12_21.2: type = converted %int.make_type_signed, %.loc12_21.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n.loc12: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%n: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_oversized.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int: type = fn_type @Int [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1000000000 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .m = %m
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: Int = import_ref ir1, inst+5, loaded [template = constants.%struct]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Int.ref: Int = name_ref Int, %import_ref [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_12: i32 = int_literal 1000000000 [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_signed: init type = call %Int.ref(%.loc9_12) [template = <error>]
// CHECK:STDOUT:   %.loc9_22.1: type = value_of_initializer %int.make_type_signed [template = <error>]
// CHECK:STDOUT:   %.loc9_22.2: type = converted %int.make_type_signed, %.loc9_22.1 [template = <error>]
// CHECK:STDOUT:   %m.var: ref <error> = var m
// CHECK:STDOUT:   %m: ref <error> = bind_name m, %m.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%n: i32) -> type = "int.make_type_signed";
// CHECK:STDOUT:
