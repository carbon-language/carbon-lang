// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_assign_to_empty.carbon" {

// CHECK:STDERR: fail_assign_to_empty.carbon:[[@LINE+3]]:17: Cannot implicitly convert from `i32` to `() as type`.
// CHECK:STDERR: var x: () = (66);
// CHECK:STDERR:                 ^
var x: () = (66);
// CHECK:STDOUT:   %.loc[[@LINE-1]]_9.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc[[@LINE-2]]_9.2: () = tuple_literal ()
// CHECK:STDOUT:   %x: ref () = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-4]]_14: i32 = int_literal 66
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
