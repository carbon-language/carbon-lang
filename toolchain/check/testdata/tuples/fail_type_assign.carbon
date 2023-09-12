// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_type_assign.carbon" {

// CHECK:STDERR: fail_type_assign.carbon:[[@LINE+3]]:25: Cannot implicitly convert from `(type,) as type` to `(i32,) as type`.
// CHECK:STDERR: var x: (i32, ) = (i32, );
// CHECK:STDERR:                         ^
var x: (i32, ) = (i32, );
// CHECK:STDOUT:   %.loc[[@LINE-1]]_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-2]]_14.1: type = tuple_type (type)
// CHECK:STDOUT:   %.loc[[@LINE-3]]_14.2: (type,) = tuple_literal (%.loc[[@LINE-3]]_9)
// CHECK:STDOUT:   %.loc[[@LINE-4]]_14.3: type = tuple_type (i32)
// CHECK:STDOUT:   %x: ref (i32,) = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-6]]_19: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-7]]_24: (type,) = tuple_literal (%.loc[[@LINE-7]]_19)
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
