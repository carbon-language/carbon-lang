// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+3]]:57: Cannot implicitly convert from `((i32, i32, i32), (i32, i32, i32)) as type` to `((i32, i32), (i32, i32)) as type`.
// CHECK:STDERR: var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));
// CHECK:STDERR:                                                         ^
var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));

// CHECK:STDOUT: file "fail_assign_nested.carbon" {
// CHECK:STDOUT:   %.loc10_10: type = stub_reference i32
// CHECK:STDOUT:   %.loc10_15: type = stub_reference i32
// CHECK:STDOUT:   %.loc10_18.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc10_18.2: (type, type) = tuple_literal (%.loc10_10, %.loc10_15)
// CHECK:STDOUT:   %.loc10_18.3: (type, type) = stub_reference %.loc10_18.2
// CHECK:STDOUT:   %.loc10_22: type = stub_reference i32
// CHECK:STDOUT:   %.loc10_27: type = stub_reference i32
// CHECK:STDOUT:   %.loc10_30.1: (type, type) = tuple_literal (%.loc10_22, %.loc10_27)
// CHECK:STDOUT:   %.loc10_30.2: (type, type) = stub_reference %.loc10_30.1
// CHECK:STDOUT:   %.loc10_31.1: type = tuple_type ((type, type), (type, type))
// CHECK:STDOUT:   %.loc10_31.2: ((type, type), (type, type)) = tuple_literal (%.loc10_18.3, %.loc10_30.2)
// CHECK:STDOUT:   %.loc10_18.4: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc10_31.3: type = tuple_type ((i32, i32), (i32, i32))
// CHECK:STDOUT:   %x: ref ((i32, i32), (i32, i32)) = var "x"
// CHECK:STDOUT:   %.loc10_37.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_37.2: i32 = stub_reference %.loc10_37.1
// CHECK:STDOUT:   %.loc10_40.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc10_40.2: i32 = stub_reference %.loc10_40.1
// CHECK:STDOUT:   %.loc10_43.1: i32 = int_literal 3
// CHECK:STDOUT:   %.loc10_43.2: i32 = stub_reference %.loc10_43.1
// CHECK:STDOUT:   %.loc10_44.1: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %.loc10_44.2: (i32, i32, i32) = tuple_literal (%.loc10_37.2, %.loc10_40.2, %.loc10_43.2)
// CHECK:STDOUT:   %.loc10_44.3: (i32, i32, i32) = stub_reference %.loc10_44.2
// CHECK:STDOUT:   %.loc10_48.1: i32 = int_literal 4
// CHECK:STDOUT:   %.loc10_48.2: i32 = stub_reference %.loc10_48.1
// CHECK:STDOUT:   %.loc10_51.1: i32 = int_literal 5
// CHECK:STDOUT:   %.loc10_51.2: i32 = stub_reference %.loc10_51.1
// CHECK:STDOUT:   %.loc10_54.1: i32 = int_literal 6
// CHECK:STDOUT:   %.loc10_54.2: i32 = stub_reference %.loc10_54.1
// CHECK:STDOUT:   %.loc10_55.1: (i32, i32, i32) = tuple_literal (%.loc10_48.2, %.loc10_51.2, %.loc10_54.2)
// CHECK:STDOUT:   %.loc10_55.2: (i32, i32, i32) = stub_reference %.loc10_55.1
// CHECK:STDOUT:   %.loc10_56.1: type = tuple_type ((i32, i32, i32), (i32, i32, i32))
// CHECK:STDOUT:   %.loc10_56.2: ((i32, i32, i32), (i32, i32, i32)) = tuple_literal (%.loc10_44.3, %.loc10_55.2)
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
