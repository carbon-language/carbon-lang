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

// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+3]]:32: ERROR: `{.a: i32, .b: i32}` does not support indexing.
// CHECK:STDERR: var c: i32 = {.a = 1, .b = 2}[0];
// CHECK:STDERR:                                ^
var c: i32 = {.a = 1, .b = 2}[0];

// CHECK:STDERR: fail_invalid_base.carbon:[[@LINE+3]]:34: ERROR: `type` does not support indexing.
// CHECK:STDERR: var d: i32 = {.a: i32, .b: i32}[0];
// CHECK:STDERR:                                  ^
var d: i32 = {.a: i32, .b: i32}[0];

// CHECK:STDOUT: file "fail_invalid_base.carbon" {
// CHECK:STDOUT:   %.loc11: <namespace> = namespace {}
// CHECK:STDOUT:   %a.loc15_5.1: ref i32 = var "a"
// CHECK:STDOUT:   %a.loc15_5.2: ref i32 = bind_name "a", %a.loc15_5.1
// CHECK:STDOUT:   %N.ref: <namespace> = name_reference "N", %.loc11
// CHECK:STDOUT:   %.loc15: i32 = int_literal 0
// CHECK:STDOUT:   assign %a.loc15_5.1, <error>
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %b.loc21_5.1: ref i32 = var "b"
// CHECK:STDOUT:   %b.loc21_5.2: ref i32 = bind_name "b", %b.loc21_5.1
// CHECK:STDOUT:   %F.ref: <function> = name_reference "F", %F
// CHECK:STDOUT:   %.loc21: i32 = int_literal 1
// CHECK:STDOUT:   assign %b.loc21_5.1, <error>
// CHECK:STDOUT:   %c.loc26_5.1: ref i32 = var "c"
// CHECK:STDOUT:   %c.loc26_5.2: ref i32 = bind_name "c", %c.loc26_5.1
// CHECK:STDOUT:   %.loc26_20: i32 = int_literal 1
// CHECK:STDOUT:   %.loc26_28: i32 = int_literal 2
// CHECK:STDOUT:   %.loc26_29.1: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc26_29.2: {.a: i32, .b: i32} = struct_literal (%.loc26_20, %.loc26_28)
// CHECK:STDOUT:   %.loc26_31: i32 = int_literal 0
// CHECK:STDOUT:   %.loc26_29.3: type = ptr_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc26_29.4: {.a: i32, .b: i32} = struct_value %.loc26_29.2, (%.loc26_20, %.loc26_28)
// CHECK:STDOUT:   assign %c.loc26_5.1, <error>
// CHECK:STDOUT:   %d.loc31_5.1: ref i32 = var "d"
// CHECK:STDOUT:   %d.loc31_5.2: ref i32 = bind_name "d", %d.loc31_5.1
// CHECK:STDOUT:   %.loc31_31: type = struct_type {.a: i32, .b: i32}
// CHECK:STDOUT:   %.loc31_33: i32 = int_literal 0
// CHECK:STDOUT:   assign %d.loc31_5.1, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
