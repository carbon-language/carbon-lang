// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: nested_tuple.carbon:[[@LINE+6]]:24: Cannot implicitly convert from `(type, type) as type` to `type`.
// CHECK:STDERR: var x: ((i32, i32), i32) = ((12, 76), 6);
// CHECK:STDERR:                        ^
// CHECK:STDERR: nested_tuple.carbon:[[@LINE+3]]:41: Cannot implicitly convert from `((i32, i32), i32) as type` to `(<error>, i32) as type`.
// CHECK:STDERR: var x: ((i32, i32), i32) = ((12, 76), 6);
// CHECK:STDERR:                                         ^
var x: ((i32, i32), i32) = ((12, 76), 6);

// CHECK:STDOUT: file "nested_tuple.carbon" {
// CHECK:STDOUT:   %.loc13_10: type = stub_reference i32
// CHECK:STDOUT:   %.loc13_15: type = stub_reference i32
// CHECK:STDOUT:   %.loc13_18.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc13_18.2: (type, type) = tuple_literal (%.loc13_10, %.loc13_15)
// CHECK:STDOUT:   %.loc13_18.3: (type, type) = tuple_literal_as_value %.loc13_18.2, (%.loc13_10, %.loc13_15)
// CHECK:STDOUT:   %.loc13_18.4: (type, type) = stub_reference %.loc13_18.3
// CHECK:STDOUT:   %.loc13_21: type = stub_reference i32
// CHECK:STDOUT:   %.loc13_24.1: type = tuple_type ((type, type), type)
// CHECK:STDOUT:   %.loc13_24.2: ((type, type), type) = tuple_literal (%.loc13_18.4, %.loc13_21)
// CHECK:STDOUT:   %.loc13_24.3: type = tuple_type (<error>, i32)
// CHECK:STDOUT:   %x: ref (<error>, i32) = var "x"
// CHECK:STDOUT:   %.loc13_30.1: i32 = int_literal 12
// CHECK:STDOUT:   %.loc13_30.2: i32 = stub_reference %.loc13_30.1
// CHECK:STDOUT:   %.loc13_34.1: i32 = int_literal 76
// CHECK:STDOUT:   %.loc13_34.2: i32 = stub_reference %.loc13_34.1
// CHECK:STDOUT:   %.loc13_36.1: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc13_36.2: (i32, i32) = tuple_literal (%.loc13_30.2, %.loc13_34.2)
// CHECK:STDOUT:   %.loc13_36.3: (i32, i32) = tuple_literal_as_value %.loc13_36.2, (%.loc13_30.2, %.loc13_34.2)
// CHECK:STDOUT:   %.loc13_36.4: (i32, i32) = stub_reference %.loc13_36.3
// CHECK:STDOUT:   %.loc13_39.1: i32 = int_literal 6
// CHECK:STDOUT:   %.loc13_39.2: i32 = stub_reference %.loc13_39.1
// CHECK:STDOUT:   %.loc13_40.1: type = tuple_type ((i32, i32), i32)
// CHECK:STDOUT:   %.loc13_40.2: ((i32, i32), i32) = tuple_literal (%.loc13_36.4, %.loc13_39.2)
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
