// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+9]]:31: Cannot implicitly convert from `(type, type) as type` to `type`.
// CHECK:STDERR: var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));
// CHECK:STDERR:                               ^
// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+6]]:31: Cannot implicitly convert from `(type, type) as type` to `type`.
// CHECK:STDERR: var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));
// CHECK:STDERR:                               ^
// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+3]]:57: Cannot implicitly convert from `((i32, i32, i32), (i32, i32, i32)) as type` to `(<error>, <error>) as type`.
// CHECK:STDERR: var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));
// CHECK:STDERR:                                                         ^
var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));

// CHECK:STDOUT: file "fail_assign_nested.carbon" {
// CHECK:STDOUT:   %.loc16_10: type = stub_reference i32
// CHECK:STDOUT:   %.loc16_15: type = stub_reference i32
// CHECK:STDOUT:   %.loc16_18.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc16_18.2: (type, type) = tuple_literal (%.loc16_10, %.loc16_15)
// CHECK:STDOUT:   %.loc16_18.3: (type, type) = tuple_literal_as_value %.loc16_18.2, (%.loc16_10, %.loc16_15)
// CHECK:STDOUT:   %.loc16_18.4: (type, type) = stub_reference %.loc16_18.3
// CHECK:STDOUT:   %.loc16_22: type = stub_reference i32
// CHECK:STDOUT:   %.loc16_27: type = stub_reference i32
// CHECK:STDOUT:   %.loc16_30.1: (type, type) = tuple_literal (%.loc16_22, %.loc16_27)
// CHECK:STDOUT:   %.loc16_30.2: (type, type) = tuple_literal_as_value %.loc16_30.1, (%.loc16_22, %.loc16_27)
// CHECK:STDOUT:   %.loc16_30.3: (type, type) = stub_reference %.loc16_30.2
// CHECK:STDOUT:   %.loc16_31.1: type = tuple_type ((type, type), (type, type))
// CHECK:STDOUT:   %.loc16_31.2: ((type, type), (type, type)) = tuple_literal (%.loc16_18.4, %.loc16_30.3)
// CHECK:STDOUT:   %.loc16_31.3: type = tuple_type (<error>, <error>)
// CHECK:STDOUT:   %x: ref (<error>, <error>) = var "x"
// CHECK:STDOUT:   %.loc16_37.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc16_37.2: i32 = stub_reference %.loc16_37.1
// CHECK:STDOUT:   %.loc16_40.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc16_40.2: i32 = stub_reference %.loc16_40.1
// CHECK:STDOUT:   %.loc16_43.1: i32 = int_literal 3
// CHECK:STDOUT:   %.loc16_43.2: i32 = stub_reference %.loc16_43.1
// CHECK:STDOUT:   %.loc16_44.1: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %.loc16_44.2: (i32, i32, i32) = tuple_literal (%.loc16_37.2, %.loc16_40.2, %.loc16_43.2)
// CHECK:STDOUT:   %.loc16_44.3: (i32, i32, i32) = tuple_literal_as_value %.loc16_44.2, (%.loc16_37.2, %.loc16_40.2, %.loc16_43.2)
// CHECK:STDOUT:   %.loc16_44.4: (i32, i32, i32) = stub_reference %.loc16_44.3
// CHECK:STDOUT:   %.loc16_48.1: i32 = int_literal 4
// CHECK:STDOUT:   %.loc16_48.2: i32 = stub_reference %.loc16_48.1
// CHECK:STDOUT:   %.loc16_51.1: i32 = int_literal 5
// CHECK:STDOUT:   %.loc16_51.2: i32 = stub_reference %.loc16_51.1
// CHECK:STDOUT:   %.loc16_54.1: i32 = int_literal 6
// CHECK:STDOUT:   %.loc16_54.2: i32 = stub_reference %.loc16_54.1
// CHECK:STDOUT:   %.loc16_55.1: (i32, i32, i32) = tuple_literal (%.loc16_48.2, %.loc16_51.2, %.loc16_54.2)
// CHECK:STDOUT:   %.loc16_55.2: (i32, i32, i32) = tuple_literal_as_value %.loc16_55.1, (%.loc16_48.2, %.loc16_51.2, %.loc16_54.2)
// CHECK:STDOUT:   %.loc16_55.3: (i32, i32, i32) = stub_reference %.loc16_55.2
// CHECK:STDOUT:   %.loc16_56.1: type = tuple_type ((i32, i32, i32), (i32, i32, i32))
// CHECK:STDOUT:   %.loc16_56.2: ((i32, i32, i32), (i32, i32, i32)) = tuple_literal (%.loc16_44.4, %.loc16_55.3)
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
