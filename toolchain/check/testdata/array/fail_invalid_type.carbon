// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_invalid_type.carbon" {

// CHECK:STDERR: fail_invalid_type.carbon:[[@LINE+3]]:13: Cannot implicitly convert from `i32` to `type`.
// CHECK:STDERR: var a: [1; 1];
// CHECK:STDERR:             ^
var a: [1; 1];
// CHECK:STDOUT:   %.loc[[@LINE-1]]_9: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-2]]_12: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-3]]_13: type = array_type %.loc[[@LINE-3]]_12, <error>
// CHECK:STDOUT:   %a: ref [<error>; 1] = var "a"
// CHECK:STDOUT: }
