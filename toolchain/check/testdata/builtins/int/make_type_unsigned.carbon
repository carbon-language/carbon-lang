// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- types.carbon

library "types" api;

fn UInt(n: i32) -> type = "int.make_type_unsigned";

// --- use_types.carbon

library "uses_types" api;

import library "types";

fn F(n: UInt(64)) -> UInt(64) {
  return n;
}

fn G(n: UInt(13)) -> UInt(13) {
  return n;
}

fn Symbolic(N:! i32, x: UInt(N)) -> UInt(N) {
  return x;
}

// --- fail_zero_size.carbon

library "fail_zero_size" api;

import library "types";

// CHECK:STDERR: fail_zero_size.carbon:[[@LINE+4]]:8: ERROR: Integer type width of 0 is not positive.
// CHECK:STDERR: var n: UInt(0);
// CHECK:STDERR:        ^~~~~
// CHECK:STDERR:
var n: UInt(0);

// --- fail_negative_size.carbon

library "fail_negative_size" api;

import library "types";

fn Negate(n: i32) -> i32 = "int.snegate";

// CHECK:STDERR: fail_negative_size.carbon:[[@LINE+4]]:8: ERROR: Integer type width of -1 is not positive.
// CHECK:STDERR: var n: UInt(Negate(1));
// CHECK:STDERR:        ^~~~~
// CHECK:STDERR:
var n: UInt(Negate(1));

// --- fail_oversized.carbon

library "fail_oversized" api;

import library "types";

// CHECK:STDERR: fail_oversized.carbon:[[@LINE+3]]:8: ERROR: Integer type width of 1000000000 is greater than the maximum supported width of 8388608.
// CHECK:STDERR: var m: UInt(1000000000);
// CHECK:STDERR:        ^~~~~
var m: UInt(1000000000);

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UInt: type = fn_type @UInt [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: UInt = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .UInt = %UInt.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %UInt.decl: UInt = fn_decl @UInt [template = constants.%struct] {
// CHECK:STDOUT:     %n.loc4_9.1: i32 = param n
// CHECK:STDOUT:     @UInt.%n: i32 = bind_name n, %n.loc4_9.1
// CHECK:STDOUT:     @UInt.%return: ref type = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n: i32) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UInt: type = fn_type @UInt [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: UInt = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %.3: type = int_type unsigned, %.2 [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 13 [template]
// CHECK:STDOUT:   %.5: type = int_type unsigned, %.4 [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.3: G = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 0 [symbolic]
// CHECK:STDOUT:   %.6: type = int_type unsigned, %N [symbolic]
// CHECK:STDOUT:   %Symbolic: type = fn_type @Symbolic [template]
// CHECK:STDOUT:   %struct.4: Symbolic = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .UInt = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .Symbolic = %Symbolic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: UInt = import_ref ir1, inst+5, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %UInt.ref.loc6_9: UInt = name_ref UInt, %import_ref [template = constants.%struct.1]
// CHECK:STDOUT:     %.loc6_14: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_13: init type = call %UInt.ref.loc6_9(%.loc6_14) [template = constants.%.3]
// CHECK:STDOUT:     %.loc6_16.1: type = value_of_initializer %int.make_type_unsigned.loc6_13 [template = constants.%.3]
// CHECK:STDOUT:     %.loc6_16.2: type = converted %int.make_type_unsigned.loc6_13, %.loc6_16.1 [template = constants.%.3]
// CHECK:STDOUT:     %n.loc6_6.1: u64 = param n
// CHECK:STDOUT:     @F.%n: u64 = bind_name n, %n.loc6_6.1
// CHECK:STDOUT:     %UInt.ref.loc6_22: UInt = name_ref UInt, %import_ref [template = constants.%struct.1]
// CHECK:STDOUT:     %.loc6_27: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:     %int.make_type_unsigned.loc6_26: init type = call %UInt.ref.loc6_22(%.loc6_27) [template = constants.%.3]
// CHECK:STDOUT:     %.loc6_29.1: type = value_of_initializer %int.make_type_unsigned.loc6_26 [template = constants.%.3]
// CHECK:STDOUT:     %.loc6_29.2: type = converted %int.make_type_unsigned.loc6_26, %.loc6_29.1 [template = constants.%.3]
// CHECK:STDOUT:     @F.%return: ref u64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.3] {
// CHECK:STDOUT:     %UInt.ref.loc10_9: UInt = name_ref UInt, %import_ref [template = constants.%struct.1]
// CHECK:STDOUT:     %.loc10_14: i32 = int_literal 13 [template = constants.%.4]
// CHECK:STDOUT:     %int.make_type_unsigned.loc10_13: init type = call %UInt.ref.loc10_9(%.loc10_14) [template = constants.%.5]
// CHECK:STDOUT:     %.loc10_16.1: type = value_of_initializer %int.make_type_unsigned.loc10_13 [template = constants.%.5]
// CHECK:STDOUT:     %.loc10_16.2: type = converted %int.make_type_unsigned.loc10_13, %.loc10_16.1 [template = constants.%.5]
// CHECK:STDOUT:     %n.loc10_6.1: u13 = param n
// CHECK:STDOUT:     @G.%n: u13 = bind_name n, %n.loc10_6.1
// CHECK:STDOUT:     %UInt.ref.loc10_22: UInt = name_ref UInt, %import_ref [template = constants.%struct.1]
// CHECK:STDOUT:     %.loc10_27: i32 = int_literal 13 [template = constants.%.4]
// CHECK:STDOUT:     %int.make_type_unsigned.loc10_26: init type = call %UInt.ref.loc10_22(%.loc10_27) [template = constants.%.5]
// CHECK:STDOUT:     %.loc10_29.1: type = value_of_initializer %int.make_type_unsigned.loc10_26 [template = constants.%.5]
// CHECK:STDOUT:     %.loc10_29.2: type = converted %int.make_type_unsigned.loc10_26, %.loc10_29.1 [template = constants.%.5]
// CHECK:STDOUT:     @G.%return: ref u13 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Symbolic.decl: Symbolic = fn_decl @Symbolic [template = constants.%struct.4] {
// CHECK:STDOUT:     %N.loc14_13.1: i32 = param N
// CHECK:STDOUT:     @Symbolic.%N: i32 = bind_symbolic_name N 0, %N.loc14_13.1 [symbolic = constants.%N]
// CHECK:STDOUT:     %UInt.ref.loc14_25: UInt = name_ref UInt, %import_ref [template = constants.%struct.1]
// CHECK:STDOUT:     %N.ref.loc14_30: i32 = name_ref N, @Symbolic.%N [symbolic = constants.%N]
// CHECK:STDOUT:     %int.make_type_unsigned.loc14_29: init type = call %UInt.ref.loc14_25(%N.ref.loc14_30) [symbolic = constants.%.6]
// CHECK:STDOUT:     %.loc14_31.1: type = value_of_initializer %int.make_type_unsigned.loc14_29 [symbolic = constants.%.6]
// CHECK:STDOUT:     %.loc14_31.2: type = converted %int.make_type_unsigned.loc14_29, %.loc14_31.1 [symbolic = constants.%.6]
// CHECK:STDOUT:     %x.loc14_22.1: Core.UInt(N) = param x
// CHECK:STDOUT:     @Symbolic.%x: Core.UInt(N) = bind_name x, %x.loc14_22.1
// CHECK:STDOUT:     %UInt.ref.loc14_37: UInt = name_ref UInt, %import_ref [template = constants.%struct.1]
// CHECK:STDOUT:     %N.ref.loc14_42: i32 = name_ref N, @Symbolic.%N [symbolic = constants.%N]
// CHECK:STDOUT:     %int.make_type_unsigned.loc14_41: init type = call %UInt.ref.loc14_37(%N.ref.loc14_42) [symbolic = constants.%.6]
// CHECK:STDOUT:     %.loc14_43.1: type = value_of_initializer %int.make_type_unsigned.loc14_41 [symbolic = constants.%.6]
// CHECK:STDOUT:     %.loc14_43.2: type = converted %int.make_type_unsigned.loc14_41, %.loc14_43.1 [symbolic = constants.%.6]
// CHECK:STDOUT:     @Symbolic.%return: ref Core.UInt(N) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n: i32) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n: u64) -> u64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: u64 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n: u13) -> u13 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: u13 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Symbolic(%N: i32, %x: Core.UInt(N)) -> Core.UInt(N) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: Core.UInt(N) = name_ref x, %x
// CHECK:STDOUT:   return %x.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_zero_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UInt: type = fn_type @UInt [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: UInt = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .UInt = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .n = %n
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: UInt = import_ref ir1, inst+5, loaded [template = constants.%struct]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %UInt.ref: UInt = name_ref UInt, %import_ref [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_13: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call %UInt.ref(%.loc10_13) [template = <error>]
// CHECK:STDOUT:   %.loc10_14.1: type = value_of_initializer %int.make_type_unsigned [template = <error>]
// CHECK:STDOUT:   %.loc10_14.2: type = converted %int.make_type_unsigned, %.loc10_14.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n: i32) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_negative_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Negate: type = fn_type @Negate [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Negate = struct_value () [template]
// CHECK:STDOUT:   %UInt: type = fn_type @UInt [template]
// CHECK:STDOUT:   %struct.2: UInt = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .UInt = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .n = %n.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: UInt = import_ref ir1, inst+5, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Negate.decl: Negate = fn_decl @Negate [template = constants.%struct.1] {
// CHECK:STDOUT:     %n.loc6_11.1: i32 = param n
// CHECK:STDOUT:     @Negate.%n: i32 = bind_name n, %n.loc6_11.1
// CHECK:STDOUT:     @Negate.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UInt.ref: UInt = name_ref UInt, %import_ref [template = constants.%struct.2]
// CHECK:STDOUT:   %Negate.ref: Negate = name_ref Negate, %Negate.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc12_20: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %int.snegate: init i32 = call %Negate.ref(%.loc12_20) [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_12.1: i32 = value_of_initializer %int.snegate [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_12.2: i32 = converted %int.snegate, %.loc12_12.1 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call %UInt.ref(%.loc12_12.2) [template = <error>]
// CHECK:STDOUT:   %.loc12_22.1: type = value_of_initializer %int.make_type_unsigned [template = <error>]
// CHECK:STDOUT:   %.loc12_22.2: type = converted %int.make_type_unsigned, %.loc12_22.1 [template = <error>]
// CHECK:STDOUT:   %n.var: ref <error> = var n
// CHECK:STDOUT:   %n.loc12: ref <error> = bind_name n, %n.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%n: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n: i32) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_oversized.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %UInt: type = fn_type @UInt [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: UInt = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1000000000 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .UInt = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .m = %m
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: UInt = import_ref ir1, inst+5, loaded [template = constants.%struct]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %UInt.ref: UInt = name_ref UInt, %import_ref [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_13: i32 = int_literal 1000000000 [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_unsigned: init type = call %UInt.ref(%.loc9_13) [template = <error>]
// CHECK:STDOUT:   %.loc9_23.1: type = value_of_initializer %int.make_type_unsigned [template = <error>]
// CHECK:STDOUT:   %.loc9_23.2: type = converted %int.make_type_unsigned, %.loc9_23.1 [template = <error>]
// CHECK:STDOUT:   %m.var: ref <error> = var m
// CHECK:STDOUT:   %m: ref <error> = bind_name m, %m.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UInt(%n: i32) -> type = "int.make_type_unsigned";
// CHECK:STDOUT:
