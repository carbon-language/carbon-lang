// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: This should probably work, it just doesn't right now.
let x: type = i32;
// CHECK:STDERR: fail_let_in_type.carbon:[[@LINE+3]]:13: ERROR: Cannot evaluate type expression.
// CHECK:STDERR: fn Six() -> x { return 6; }
// CHECK:STDERR:             ^
fn Six() -> x { return 6; }

// CHECK:STDOUT: file "fail_let_in_type.carbon" {
// CHECK:STDOUT:   %x: type = bind_name x, i32
// CHECK:STDOUT:   %Six: <function> = fn_decl @Six
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Six() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12: i32 = int_literal 6
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
