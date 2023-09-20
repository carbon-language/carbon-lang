// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: ((i32, i32), i32) = ((12, 76), 6);

// CHECK:STDOUT: file "nested_tuple.carbon" {
// CHECK:STDOUT:   %.loc7_10: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_15: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_18.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc7_18.2: (type, type) = tuple_literal (%.loc7_10, %.loc7_15)
// CHECK:STDOUT:   %.loc7_18.3: (type, type) = stub_reference %.loc7_18.2
// CHECK:STDOUT:   %.loc7_21: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_24.1: type = tuple_type ((type, type), type)
// CHECK:STDOUT:   %.loc7_24.2: ((type, type), type) = tuple_literal (%.loc7_18.3, %.loc7_21)
// CHECK:STDOUT:   %.loc7_18.4: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc7_24.3: type = tuple_type ((i32, i32), i32)
// CHECK:STDOUT:   %x: ref ((i32, i32), i32) = var "x"
// CHECK:STDOUT:   %.loc7_30.1: i32 = int_literal 12
// CHECK:STDOUT:   %.loc7_30.2: i32 = stub_reference %.loc7_30.1
// CHECK:STDOUT:   %.loc7_34.1: i32 = int_literal 76
// CHECK:STDOUT:   %.loc7_34.2: i32 = stub_reference %.loc7_34.1
// CHECK:STDOUT:   %.loc7_36.1: (i32, i32) = tuple_literal (%.loc7_30.2, %.loc7_34.2)
// CHECK:STDOUT:   %.loc7_36.2: (i32, i32) = stub_reference %.loc7_36.1
// CHECK:STDOUT:   %.loc7_39.1: i32 = int_literal 6
// CHECK:STDOUT:   %.loc7_39.2: i32 = stub_reference %.loc7_39.1
// CHECK:STDOUT:   %.loc7_40: ((i32, i32), i32) = tuple_literal (%.loc7_36.2, %.loc7_39.2)
// CHECK:STDOUT:   %.loc7_41.1: ref (i32, i32) = tuple_access %x, member0
// CHECK:STDOUT:   %.loc7_41.2: ref i32 = tuple_access %.loc7_41.1, member0
// CHECK:STDOUT:   %.loc7_41.3: init i32 = initialize_from %.loc7_30.2 to %.loc7_41.2
// CHECK:STDOUT:   %.loc7_41.4: ref i32 = tuple_access %.loc7_41.1, member1
// CHECK:STDOUT:   %.loc7_41.5: init i32 = initialize_from %.loc7_34.2 to %.loc7_41.4
// CHECK:STDOUT:   %.loc7_41.6: init (i32, i32) = tuple_init %.loc7_36.1, (%.loc7_41.3, %.loc7_41.5)
// CHECK:STDOUT:   %.loc7_41.7: ref i32 = tuple_access %x, member1
// CHECK:STDOUT:   %.loc7_41.8: init i32 = initialize_from %.loc7_39.2 to %.loc7_41.7
// CHECK:STDOUT:   %.loc7_41.9: init ((i32, i32), i32) = tuple_init %.loc7_40, (%.loc7_41.6, %.loc7_41.8)
// CHECK:STDOUT:   assign %x, %.loc7_41.9
// CHECK:STDOUT: }
