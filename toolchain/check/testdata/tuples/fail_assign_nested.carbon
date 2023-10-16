// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+3]]:44: ERROR: Cannot initialize tuple of 2 element(s) from tuple with 3 element(s).
// CHECK:STDERR: var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));
// CHECK:STDERR:                                            ^
var x: ((i32, i32), (i32, i32)) = ((1, 2, 3), (4, 5, 6));

// CHECK:STDOUT: file "fail_assign_nested.carbon" {
// CHECK:STDOUT:   %.loc10_18.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc10_18.2: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc10_30: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc10_31.1: type = tuple_type ((type, type), (type, type))
// CHECK:STDOUT:   %.loc10_31.2: ((type, type), (type, type)) = tuple_literal (%.loc10_18.2, %.loc10_30)
// CHECK:STDOUT:   %.loc10_31.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc10_31.4: type = tuple_type ((i32, i32), (i32, i32))
// CHECK:STDOUT:   %.loc10_31.5: type = ptr_type (i32, i32)
// CHECK:STDOUT:   %.loc10_31.6: type = tuple_type ((i32, i32)*, (i32, i32)*)
// CHECK:STDOUT:   %.loc10_31.7: type = ptr_type ((i32, i32)*, (i32, i32)*)
// CHECK:STDOUT:   %x: ref ((i32, i32), (i32, i32)) = var "x"
// CHECK:STDOUT:   %.loc10_37: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_40: i32 = int_literal 2
// CHECK:STDOUT:   %.loc10_43: i32 = int_literal 3
// CHECK:STDOUT:   %.loc10_44.1: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %.loc10_44.2: (i32, i32, i32) = tuple_literal (%.loc10_37, %.loc10_40, %.loc10_43)
// CHECK:STDOUT:   %.loc10_48: i32 = int_literal 4
// CHECK:STDOUT:   %.loc10_51: i32 = int_literal 5
// CHECK:STDOUT:   %.loc10_54: i32 = int_literal 6
// CHECK:STDOUT:   %.loc10_55: (i32, i32, i32) = tuple_literal (%.loc10_48, %.loc10_51, %.loc10_54)
// CHECK:STDOUT:   %.loc10_56.1: type = tuple_type ((i32, i32, i32), (i32, i32, i32))
// CHECK:STDOUT:   %.loc10_56.2: ((i32, i32, i32), (i32, i32, i32)) = tuple_literal (%.loc10_44.2, %.loc10_55)
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
