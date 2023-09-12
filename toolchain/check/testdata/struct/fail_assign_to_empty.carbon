// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_assign_to_empty.carbon" {

// CHECK:STDERR: fail_assign_to_empty.carbon:[[@LINE+3]]:21: Cannot implicitly convert from `{.a: i32}` to `{} as type`.
// CHECK:STDERR: var x: {} = {.a = 1};
// CHECK:STDERR:                     ^
var x: {} = {.a = 1};
// CHECK:STDOUT:   %.loc[[@LINE-1]]_9.1: type = struct_type {}
// CHECK:STDOUT:   %.loc[[@LINE-2]]_9.2: {} = struct_literal ()
// CHECK:STDOUT:   %x: ref {} = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-4]]_19: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-5]]_17: i32 = stub_reference %.loc[[@LINE-5]]_19
// CHECK:STDOUT:   %.loc[[@LINE-6]]_20.1: type = struct_type {.a: i32}
// CHECK:STDOUT:   %.loc[[@LINE-7]]_20.2: {.a: i32} = struct_literal (%.loc[[@LINE-7]]_17)
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
