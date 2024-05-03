// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Negate(n: i32) -> i32 = "int.snegate";

// CHECK:STDERR: fail_bound_negative.carbon:[[@LINE+3]]:14: ERROR: Array bound of -1 is negative.
// CHECK:STDERR: var a: [i32; Negate(1)];
// CHECK:STDERR:              ^~~~~~~
var a: [i32; Negate(1)];

// CHECK:STDOUT: --- fail_bound_negative.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Negate: type = fn_type @Negate [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Negate = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Negate.decl: Negate = fn_decl @Negate [template = constants.%struct] {
// CHECK:STDOUT:     %n.loc7_11.1: i32 = param n
// CHECK:STDOUT:     @Negate.%n: i32 = bind_name n, %n.loc7_11.1
// CHECK:STDOUT:     @Negate.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.ref: Negate = name_ref Negate, %Negate.decl [template = constants.%struct]
// CHECK:STDOUT:   %.loc12_21: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %int.snegate: init i32 = call %Negate.ref(%.loc12_21) [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_23: type = array_type %int.snegate, i32 [template = <error>]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%n: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
