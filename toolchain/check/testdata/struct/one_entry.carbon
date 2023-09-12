// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "one_entry.carbon" {

var x: {.a: i32} = {.a = 4};
// CHECK:STDOUT:   %.loc[[@LINE-1]]_16: type = struct_type {.a: i32}
// CHECK:STDOUT:   %x: ref {.a: i32} = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-3]]_26: i32 = int_literal 4
// CHECK:STDOUT:   %.loc[[@LINE-4]]_24: i32 = stub_reference %.loc[[@LINE-4]]_26
// CHECK:STDOUT:   %.loc[[@LINE-5]]_27: {.a: i32} = struct_literal (%.loc[[@LINE-5]]_24)
// CHECK:STDOUT:   assign %x, %.loc[[@LINE-6]]_27
var y: {.a: i32} = x;
// CHECK:STDOUT:   %.loc[[@LINE-1]]: type = struct_type {.a: i32}
// CHECK:STDOUT:   %y: ref {.a: i32} = var "y"
// CHECK:STDOUT:   %.loc[[@LINE-10]]_5: {.a: i32} = bind_value %x
// CHECK:STDOUT:   assign %y, %.loc[[@LINE-11]]_5
// CHECK:STDOUT: }
