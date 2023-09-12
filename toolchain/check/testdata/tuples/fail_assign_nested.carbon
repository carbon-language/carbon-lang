// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_assign_nested.carbon" {

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+3]]:57: Cannot implicitly convert from `((i32, i32, i32), (i32, i32, i32)) as type` to `((i32, i32), (i32, i32)) as type`.
// CHECK:STDERR: var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));
// CHECK:STDERR:                                                         ^
var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));
// CHECK:STDOUT:   %.loc[[@LINE-1]]_10: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-2]]_15: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-3]]_18.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc[[@LINE-4]]_18.2: (type, type) = tuple_literal (%.loc[[@LINE-4]]_10, %.loc[[@LINE-4]]_15)
// CHECK:STDOUT:   %.loc[[@LINE-5]]_18.3: (type, type) = stub_reference %.loc[[@LINE-5]]_18.2
// CHECK:STDOUT:   %.loc[[@LINE-6]]_22: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-7]]_27: type = stub_reference i32
// CHECK:STDOUT:   %.loc[[@LINE-8]]_30.1: (type, type) = tuple_literal (%.loc[[@LINE-8]]_22, %.loc[[@LINE-8]]_27)
// CHECK:STDOUT:   %.loc[[@LINE-9]]_30.2: (type, type) = stub_reference %.loc[[@LINE-9]]_30.1
// CHECK:STDOUT:   %.loc[[@LINE-10]]_31.1: type = tuple_type ((type, type), (type, type))
// CHECK:STDOUT:   %.loc[[@LINE-11]]_31.2: ((type, type), (type, type)) = tuple_literal (%.loc[[@LINE-11]]_18.3, %.loc[[@LINE-11]]_30.2)
// CHECK:STDOUT:   %.loc[[@LINE-12]]_18.4: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc[[@LINE-13]]_31.3: type = tuple_type ((i32, i32), (i32, i32))
// CHECK:STDOUT:   %x: ref ((i32, i32), (i32, i32)) = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-15]]_37.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-16]]_37.2: i32 = stub_reference %.loc[[@LINE-16]]_37.1
// CHECK:STDOUT:   %.loc[[@LINE-17]]_40.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc[[@LINE-18]]_40.2: i32 = stub_reference %.loc[[@LINE-18]]_40.1
// CHECK:STDOUT:   %.loc[[@LINE-19]]_43.1: i32 = int_literal 3
// CHECK:STDOUT:   %.loc[[@LINE-20]]_43.2: i32 = stub_reference %.loc[[@LINE-20]]_43.1
// CHECK:STDOUT:   %.loc[[@LINE-21]]_44.1: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %.loc[[@LINE-22]]_44.2: (i32, i32, i32) = tuple_literal (%.loc[[@LINE-22]]_37.2, %.loc[[@LINE-22]]_40.2, %.loc[[@LINE-22]]_43.2)
// CHECK:STDOUT:   %.loc[[@LINE-23]]_44.3: (i32, i32, i32) = stub_reference %.loc[[@LINE-23]]_44.2
// CHECK:STDOUT:   %.loc[[@LINE-24]]_48.1: i32 = int_literal 4
// CHECK:STDOUT:   %.loc[[@LINE-25]]_48.2: i32 = stub_reference %.loc[[@LINE-25]]_48.1
// CHECK:STDOUT:   %.loc[[@LINE-26]]_51.1: i32 = int_literal 5
// CHECK:STDOUT:   %.loc[[@LINE-27]]_51.2: i32 = stub_reference %.loc[[@LINE-27]]_51.1
// CHECK:STDOUT:   %.loc[[@LINE-28]]_54.1: i32 = int_literal 6
// CHECK:STDOUT:   %.loc[[@LINE-29]]_54.2: i32 = stub_reference %.loc[[@LINE-29]]_54.1
// CHECK:STDOUT:   %.loc[[@LINE-30]]_55.1: (i32, i32, i32) = tuple_literal (%.loc[[@LINE-30]]_48.2, %.loc[[@LINE-30]]_51.2, %.loc[[@LINE-30]]_54.2)
// CHECK:STDOUT:   %.loc[[@LINE-31]]_55.2: (i32, i32, i32) = stub_reference %.loc[[@LINE-31]]_55.1
// CHECK:STDOUT:   %.loc[[@LINE-32]]_56.1: type = tuple_type ((i32, i32, i32), (i32, i32, i32))
// CHECK:STDOUT:   %.loc[[@LINE-33]]_56.2: ((i32, i32, i32), (i32, i32, i32)) = tuple_literal (%.loc[[@LINE-33]]_44.3, %.loc[[@LINE-33]]_55.2)
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
