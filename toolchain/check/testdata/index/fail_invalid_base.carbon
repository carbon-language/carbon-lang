// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: The errors here are in bad locations. We should build a SemIR
// representation for a reference to a name so that we can track the location
// properly.

// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+3]]:12: Expression cannot be used as a value.
// CHECK:STDERR: namespace N;
// CHECK:STDERR:            ^
namespace N;
var a: i32 = N[0];

// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+3]]:1: Expression cannot be used as a value.
// CHECK:STDERR: fn F();
// CHECK:STDERR: ^
fn F();
var b: i32 = F[1];

// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+3]]:32: `{.a: i32, .b: i32}` does not support indexing.
// CHECK:STDERR: var c: i32 = {.a = 1, .b = 2}[0];
// CHECK:STDERR:                                ^
var c: i32 = {.a = 1, .b = 2}[0];

// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+3]]:34: `type` does not support indexing.
// CHECK:STDERR: var d: i32 = {.a: i32, .b: i32}[0];
// CHECK:STDERR:                                  ^
var d: i32 = {.a: i32, .b: i32}[0];

// CHECK:STDOUT: file "fail_invalid_base.carbon" {
// CHECK:STDOUT:   %.loc14 = namespace {}
// CHECK:STDOUT:   %a: ref i32 = var "a"
// CHECK:STDOUT:   %.loc15: i32 = int_literal 0
// CHECK:STDOUT:   assign %a, <error>
// CHECK:STDOUT:   %.loc20 = fn_decl @F
// CHECK:STDOUT:   %b: ref i32 = var "b"
// CHECK:STDOUT:   %.loc21: i32 = int_literal 1
// CHECK:STDOUT:   assign %b, <error>
// CHECK:STDOUT:   %c: ref i32 = var "c"
// CHECK:STDOUT:   %.loc26_20: i32 = int_literal 1
// CHECK:STDOUT:   %.loc26_28: i32 = int_literal 2
// CHECK:STDOUT:   %.loc26_29.1: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc26_29.2: {.a: i32, .b: i32} = struct_literal (%.loc26_20, %.loc26_28)
// CHECK:STDOUT:   %.loc26_31: i32 = int_literal 0
// CHECK:STDOUT:   %.loc26_29.3: {.a: i32, .b: i32} = struct_value %.loc26_29.2, (%.loc26_20, %.loc26_28)
// CHECK:STDOUT:   assign %c, <error>
// CHECK:STDOUT:   %d: ref i32 = var "d"
// CHECK:STDOUT:   %.loc31_31: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc31_33: i32 = int_literal 0
// CHECK:STDOUT:   assign %d, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
