// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "two_elements.carbon" {

var x: (i32, i32) = (4, 102);
// CHECK:STDOUT:   %.loc[[@LINE-1]]_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-2]]_14: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-3]]_17.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc[[@LINE-4]]_17.2: (type, type) = tuple_literal (%.loc[[@LINE-4]]_9, %.loc[[@LINE-4]]_14)
// CHECK:STDOUT:   %.loc[[@LINE-5]]_17.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %x: ref (i32, i32) = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-7]]_22.1: i32 = int_literal 4
// CHECK:STDOUT:   %.loc[[@LINE-8]]_22.2: i32 = stub_reference %.loc[[@LINE-8]]_22.1
// CHECK:STDOUT:   %.loc[[@LINE-9]]_25.1: i32 = int_literal 102
// CHECK:STDOUT:   %.loc[[@LINE-10]]_25.2: i32 = stub_reference %.loc[[@LINE-10]]_25.1
// CHECK:STDOUT:   %.loc[[@LINE-11]]_28: (i32, i32) = tuple_literal (%.loc[[@LINE-11]]_22.2, %.loc[[@LINE-11]]_25.2)
// CHECK:STDOUT:   assign %x, %.loc[[@LINE-12]]_28
var y: (i32, i32) = x;
// CHECK:STDOUT:   %.loc[[@LINE-1]]_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-2]]_14: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-3]]_17: (type, type) = tuple_literal (%.loc[[@LINE-3]]_9, %.loc[[@LINE-3]]_14)
// CHECK:STDOUT:   %y: ref (i32, i32) = var "y"
// CHECK:STDOUT:   %.loc[[@LINE-18]]_5: (i32, i32) = bind_value %x
// CHECK:STDOUT:   assign %y, %.loc[[@LINE-19]]_5
// CHECK:STDOUT: }
