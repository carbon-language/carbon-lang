// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_value_as_type.carbon" {

// CHECK:STDERR: fail_value_as_type.carbon:[[@LINE+3]]:12: Cannot implicitly convert from `i32` to `type`.
// CHECK:STDERR: var x: (1, );
// CHECK:STDERR:            ^
var x: (1, );
// CHECK:STDOUT:   %.loc[[@LINE-1]]_9.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-2]]_9.2: i32 = stub_reference %.loc[[@LINE-2]]_9.1
// CHECK:STDOUT:   %.loc[[@LINE-3]]_12.1: type = tuple_type (i32)
// CHECK:STDOUT:   %.loc[[@LINE-4]]_12.2: (i32,) = tuple_literal (%.loc[[@LINE-4]]_9.2)
// CHECK:STDOUT:   %.loc[[@LINE-5]]_12.3: type = tuple_type (<error>)
// CHECK:STDOUT:   %x: ref (<error>,) = var "x"
// CHECK:STDOUT: }
