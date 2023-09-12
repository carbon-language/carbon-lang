// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "nine_elements.carbon" {

var a: [i32; 9] = (1, 2, 3, 4, 5, 6, 7, 8, 9);
// CHECK:STDOUT:   %.loc[[@LINE-1]]_14: i32 = int_literal 9
// CHECK:STDOUT:   %.loc[[@LINE-2]]_15: type = array_type %.loc[[@LINE-2]]_14, i32
// CHECK:STDOUT:   %a: ref [i32; 9] = var "a"
// CHECK:STDOUT:   %.loc[[@LINE-4]]_20.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-5]]_20.2: i32 = stub_reference %.loc[[@LINE-5]]_20.1
// CHECK:STDOUT:   %.loc[[@LINE-6]]_23.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc[[@LINE-7]]_23.2: i32 = stub_reference %.loc[[@LINE-7]]_23.1
// CHECK:STDOUT:   %.loc[[@LINE-8]]_26.1: i32 = int_literal 3
// CHECK:STDOUT:   %.loc[[@LINE-9]]_26.2: i32 = stub_reference %.loc[[@LINE-9]]_26.1
// CHECK:STDOUT:   %.loc[[@LINE-10]]_29.1: i32 = int_literal 4
// CHECK:STDOUT:   %.loc[[@LINE-11]]_29.2: i32 = stub_reference %.loc[[@LINE-11]]_29.1
// CHECK:STDOUT:   %.loc[[@LINE-12]]_32.1: i32 = int_literal 5
// CHECK:STDOUT:   %.loc[[@LINE-13]]_32.2: i32 = stub_reference %.loc[[@LINE-13]]_32.1
// CHECK:STDOUT:   %.loc[[@LINE-14]]_35.1: i32 = int_literal 6
// CHECK:STDOUT:   %.loc[[@LINE-15]]_35.2: i32 = stub_reference %.loc[[@LINE-15]]_35.1
// CHECK:STDOUT:   %.loc[[@LINE-16]]_38.1: i32 = int_literal 7
// CHECK:STDOUT:   %.loc[[@LINE-17]]_38.2: i32 = stub_reference %.loc[[@LINE-17]]_38.1
// CHECK:STDOUT:   %.loc[[@LINE-18]]_41.1: i32 = int_literal 8
// CHECK:STDOUT:   %.loc[[@LINE-19]]_41.2: i32 = stub_reference %.loc[[@LINE-19]]_41.1
// CHECK:STDOUT:   %.loc[[@LINE-20]]_44.1: i32 = int_literal 9
// CHECK:STDOUT:   %.loc[[@LINE-21]]_44.2: i32 = stub_reference %.loc[[@LINE-21]]_44.1
// CHECK:STDOUT:   %.loc[[@LINE-22]]_45.1: type = tuple_type (i32, i32, i32, i32, i32, i32, i32, i32, i32)
// CHECK:STDOUT:   %.loc[[@LINE-23]]_45.2: (i32, i32, i32, i32, i32, i32, i32, i32, i32) = tuple_literal (%.loc[[@LINE-23]]_20.2, %.loc[[@LINE-23]]_23.2, %.loc[[@LINE-23]]_26.2, %.loc[[@LINE-23]]_29.2, %.loc[[@LINE-23]]_32.2, %.loc[[@LINE-23]]_35.2, %.loc[[@LINE-23]]_38.2, %.loc[[@LINE-23]]_41.2, %.loc[[@LINE-23]]_44.2)
// CHECK:STDOUT:   %.loc[[@LINE-24]]_45.3: [i32; 9] = array_value %.loc[[@LINE-24]]_45.2
// CHECK:STDOUT:   assign %a, %.loc[[@LINE-25]]_45.3
// CHECK:STDOUT: }
