// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "nested_tuple.carbon" {

var x: ((i32, i32), i32) = ((12, 76), 6);
// CHECK:STDOUT:   %.loc[[@LINE-1]]_10: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-2]]_15: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-3]]_18.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc[[@LINE-4]]_18.2: (type, type) = tuple_literal (%.loc[[@LINE-4]]_10, %.loc[[@LINE-4]]_15)
// CHECK:STDOUT:   %.loc[[@LINE-5]]_18.3: (type, type) = stub_reference %.loc[[@LINE-5]]_18.2
// CHECK:STDOUT:   %.loc[[@LINE-6]]_21: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-7]]_24.1: type = tuple_type ((type, type), type)
// CHECK:STDOUT:   %.loc[[@LINE-8]]_24.2: ((type, type), type) = tuple_literal (%.loc[[@LINE-8]]_18.3, %.loc[[@LINE-8]]_21)
// CHECK:STDOUT:   %.loc[[@LINE-9]]_18.4: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc[[@LINE-10]]_24.3: type = tuple_type ((i32, i32), i32)
// CHECK:STDOUT:   %x: ref ((i32, i32), i32) = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-12]]_30.1: i32 = int_literal 12
// CHECK:STDOUT:   %.loc[[@LINE-13]]_30.2: i32 = stub_reference %.loc[[@LINE-13]]_30.1
// CHECK:STDOUT:   %.loc[[@LINE-14]]_34.1: i32 = int_literal 76
// CHECK:STDOUT:   %.loc[[@LINE-15]]_34.2: i32 = stub_reference %.loc[[@LINE-15]]_34.1
// CHECK:STDOUT:   %.loc[[@LINE-16]]_36.1: (i32, i32) = tuple_literal (%.loc[[@LINE-16]]_30.2, %.loc[[@LINE-16]]_34.2)
// CHECK:STDOUT:   %.loc[[@LINE-17]]_36.2: (i32, i32) = stub_reference %.loc[[@LINE-17]]_36.1
// CHECK:STDOUT:   %.loc[[@LINE-18]]_39.1: i32 = int_literal 6
// CHECK:STDOUT:   %.loc[[@LINE-19]]_39.2: i32 = stub_reference %.loc[[@LINE-19]]_39.1
// CHECK:STDOUT:   %.loc[[@LINE-20]]_40: ((i32, i32), i32) = tuple_literal (%.loc[[@LINE-20]]_36.2, %.loc[[@LINE-20]]_39.2)
// CHECK:STDOUT:   assign %x, %.loc[[@LINE-21]]_40
// CHECK:STDOUT: }
