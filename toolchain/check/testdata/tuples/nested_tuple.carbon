// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: ((i32, i32), i32) = ((12, 76), 6);

// CHECK:STDOUT: file "nested_tuple.carbon" {
// CHECK:STDOUT:   %.loc7_18.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc7_18.2: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc7_24.1: type = tuple_type ((type, type), type)
// CHECK:STDOUT:   %.loc7_24.2: ((type, type), type) = tuple_literal (%.loc7_18.2, i32)
// CHECK:STDOUT:   %.loc7_24.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc7_24.4: type = tuple_type ((i32, i32), i32)
// CHECK:STDOUT:   %.loc7_24.5: type = ptr_type (i32, i32)
// CHECK:STDOUT:   %.loc7_24.6: type = tuple_type ((i32, i32)*, i32)
// CHECK:STDOUT:   %.loc7_24.7: type = ptr_type ((i32, i32)*, i32)
// CHECK:STDOUT:   %x: ref ((i32, i32), i32) = var "x"
// CHECK:STDOUT:   %.loc7_30: i32 = int_literal 12
// CHECK:STDOUT:   %.loc7_34: i32 = int_literal 76
// CHECK:STDOUT:   %.loc7_36.1: (i32, i32) = tuple_literal (%.loc7_30, %.loc7_34)
// CHECK:STDOUT:   %.loc7_39: i32 = int_literal 6
// CHECK:STDOUT:   %.loc7_40.1: ((i32, i32), i32) = tuple_literal (%.loc7_36.1, %.loc7_39)
// CHECK:STDOUT:   %.loc7_40.2: ref (i32, i32) = tuple_access %x, member0
// CHECK:STDOUT:   %.loc7_36.2: ref i32 = tuple_access %.loc7_40.2, member0
// CHECK:STDOUT:   %.loc7_36.3: init i32 = initialize_from %.loc7_30 to %.loc7_36.2
// CHECK:STDOUT:   %.loc7_36.4: ref i32 = tuple_access %.loc7_40.2, member1
// CHECK:STDOUT:   %.loc7_36.5: init i32 = initialize_from %.loc7_34 to %.loc7_36.4
// CHECK:STDOUT:   %.loc7_36.6: init (i32, i32) = tuple_init %.loc7_36.1, (%.loc7_36.3, %.loc7_36.5)
// CHECK:STDOUT:   %.loc7_40.3: ref i32 = tuple_access %x, member1
// CHECK:STDOUT:   %.loc7_40.4: init i32 = initialize_from %.loc7_39 to %.loc7_40.3
// CHECK:STDOUT:   %.loc7_40.5: init ((i32, i32), i32) = tuple_init %.loc7_40.1, (%.loc7_36.6, %.loc7_40.4)
// CHECK:STDOUT:   assign %x, %.loc7_40.5
// CHECK:STDOUT: }
