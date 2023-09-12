// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "one_element.carbon" {

var x: (i32,) = (4,);
// CHECK:STDOUT:   %.loc[[@LINE-1]]_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-2]]_13.1: type = tuple_type (type)
// CHECK:STDOUT:   %.loc[[@LINE-3]]_13.2: (type,) = tuple_literal (%.loc[[@LINE-3]]_9)
// CHECK:STDOUT:   %.loc[[@LINE-4]]_13.3: type = tuple_type (i32)
// CHECK:STDOUT:   %x: ref (i32,) = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-6]]_18.1: i32 = int_literal 4
// CHECK:STDOUT:   %.loc[[@LINE-7]]_18.2: i32 = stub_reference %.loc[[@LINE-7]]_18.1
// CHECK:STDOUT:   %.loc[[@LINE-8]]_20: (i32,) = tuple_literal (%.loc[[@LINE-8]]_18.2)
// CHECK:STDOUT:   assign %x, %.loc[[@LINE-9]]_20
var y: (i32,) = x;
// CHECK:STDOUT:   %.loc[[@LINE-1]]_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-2]]_13: (type,) = tuple_literal (%.loc[[@LINE-2]]_9)
// CHECK:STDOUT:   %y: ref (i32,) = var "y"
// CHECK:STDOUT:   %.loc[[@LINE-14]]_5: (i32,) = bind_value %x
// CHECK:STDOUT:   assign %y, %.loc[[@LINE-15]]_5
// CHECK:STDOUT: }
