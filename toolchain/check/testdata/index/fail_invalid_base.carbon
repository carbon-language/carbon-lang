// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: The errors here are in bad locations. We should build a SemIR
// representation for a reference to a name so that we can track the location
// properly.

namespace N;
// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+3]]:14: ERROR: Expression cannot be used as a value.
// CHECK:STDERR: var a: i32 = N[0];
// CHECK:STDERR:              ^
var a: i32 = N[0];

fn F();
// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+3]]:14: ERROR: Expression cannot be used as a value.
// CHECK:STDERR: var b: i32 = F[1];
// CHECK:STDERR:              ^
var b: i32 = F[1];

// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+3]]:14: ERROR: Type `{.a: i32, .b: i32}` does not support indexing.
// CHECK:STDERR: var c: i32 = {.a = 1, .b = 2}[0];
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~
var c: i32 = {.a = 1, .b = 2}[0];

// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+3]]:14: ERROR: Type `type` does not support indexing.
// CHECK:STDERR: var d: i32 = {.a: i32, .b: i32}[0];
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~
var d: i32 = {.a: i32, .b: i32}[0];

// CHECK:STDOUT: --- fail_invalid_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.5: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.6: {.a: i32, .b: i32} = struct_value (%.2, %.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.N = %.loc11, .a = %a, .F = %F, .b = %b, .c = %c, .d = %d} [template]
// CHECK:STDOUT:   %.loc11: <namespace> = namespace {} [template]
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %N.ref: <namespace> = name_ref N, %.loc11 [template = %.loc11]
// CHECK:STDOUT:   %.loc15: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %a.var, <error>
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, %F [template = %F]
// CHECK:STDOUT:   %.loc21: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT:   %.loc26_20: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc26_28: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc26_29.1: {.a: i32, .b: i32} = struct_literal (%.loc26_20, %.loc26_28)
// CHECK:STDOUT:   %.loc26_31: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   %.loc26_29.2: {.a: i32, .b: i32} = struct_value (%.loc26_20, %.loc26_28) [template = constants.%.6]
// CHECK:STDOUT:   %.loc26_29.3: {.a: i32, .b: i32} = converted %.loc26_29.1, %.loc26_29.2 [template = constants.%.6]
// CHECK:STDOUT:   assign %c.var, <error>
// CHECK:STDOUT:   %d.var: ref i32 = var d
// CHECK:STDOUT:   %d: ref i32 = bind_name d, %d.var
// CHECK:STDOUT:   %.loc31_31: type = struct_type {.a: i32, .b: i32} [template = constants.%.4]
// CHECK:STDOUT:   %.loc31_33: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %d.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
