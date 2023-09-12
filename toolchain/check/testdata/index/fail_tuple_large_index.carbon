// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_tuple_large_index.carbon" {

var a: (i32,) = (12,);
// CHECK:STDOUT:   %.loc[[@LINE-1]]_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-2]]_13.1: type = tuple_type (type)
// CHECK:STDOUT:   %.loc[[@LINE-3]]_13.2: (type,) = tuple_literal (%.loc[[@LINE-3]]_9)
// CHECK:STDOUT:   %.loc[[@LINE-4]]_13.3: type = tuple_type (i32)
// CHECK:STDOUT:   %a: ref (i32,) = var "a"
// CHECK:STDOUT:   %.loc[[@LINE-6]]_18.1: i32 = int_literal 12
// CHECK:STDOUT:   %.loc[[@LINE-7]]_18.2: i32 = stub_reference %.loc[[@LINE-7]]_18.1
// CHECK:STDOUT:   %.loc[[@LINE-8]]_21: (i32,) = tuple_literal (%.loc[[@LINE-8]]_18.2)
// CHECK:STDOUT:   assign %a, %.loc[[@LINE-9]]_21
var b: (i32,) = a;
// CHECK:STDOUT:   %.loc[[@LINE-1]]_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-2]]_13: (type,) = tuple_literal (%.loc[[@LINE-2]]_9)
// CHECK:STDOUT:   %b: ref (i32,) = var "b"
// CHECK:STDOUT:   %.loc[[@LINE-14]]_5: (i32,) = bind_value %a
// CHECK:STDOUT:   assign %b, %.loc[[@LINE-15]]_5
// CHECK:STDOUT:   %c: ref i32 = var "c"
// CHECK:STDERR: fail_tuple_large_index.carbon:[[@LINE+3]]:35: Index `295147905179352825855` is past the end of `(i32,) as type`.
// CHECK:STDERR: var c: i32 = b[0xFFFFFFFFFFFFFFFFF];
// CHECK:STDERR:                                   ^
var c: i32 = b[0xFFFFFFFFFFFFFFFFF];
// CHECK:STDOUT:   %.loc[[@LINE-1]]_16: i32 = int_literal 295147905179352825855
// CHECK:STDOUT:   %.loc[[@LINE-2]]_35: ref <error> = tuple_index %b, <error>
// CHECK:STDOUT:   assign %c, <error>
// CHECK:STDOUT: }
