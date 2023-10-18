// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 1] = (1,);
var b: [f64; 2] = (11.1, 2.2,);
var c: [(); 5] = ((), (), (), (), (),);

// CHECK:STDOUT: file "base.carbon" {
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_15.1: type = array_type %.loc7_14, i32
// CHECK:STDOUT:   %.loc7_15.2: type = ptr_type [i32; 1]
// CHECK:STDOUT:   %a: ref [i32; 1] = var "a"
// CHECK:STDOUT:   %.loc7_20: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_22.1: type = tuple_type (i32)
// CHECK:STDOUT:   %.loc7_22.2: (i32,) = tuple_literal (%.loc7_20)
// CHECK:STDOUT:   %.loc7_22.3: i32 = int_literal 0
// CHECK:STDOUT:   %.loc7_22.4: ref i32 = array_index %a, %.loc7_22.3
// CHECK:STDOUT:   %.loc7_22.5: init i32 = initialize_from %.loc7_20 to %.loc7_22.4
// CHECK:STDOUT:   %.loc7_22.6: init [i32; 1] = array_init %.loc7_22.2, (%.loc7_22.5) to %a
// CHECK:STDOUT:   assign %a, %.loc7_22.6
// CHECK:STDOUT:   %.loc8_14: i32 = int_literal 2
// CHECK:STDOUT:   %.loc8_15.1: type = array_type %.loc8_14, f64
// CHECK:STDOUT:   %.loc8_15.2: type = ptr_type [f64; 2]
// CHECK:STDOUT:   %b: ref [f64; 2] = var "b"
// CHECK:STDOUT:   %.loc8_20: f64 = real_literal 111e-1
// CHECK:STDOUT:   %.loc8_26: f64 = real_literal 22e-1
// CHECK:STDOUT:   %.loc8_30.1: type = tuple_type (f64, f64)
// CHECK:STDOUT:   %.loc8_30.2: (f64, f64) = tuple_literal (%.loc8_20, %.loc8_26)
// CHECK:STDOUT:   %.loc8_30.3: i32 = int_literal 0
// CHECK:STDOUT:   %.loc8_30.4: ref f64 = array_index %b, %.loc8_30.3
// CHECK:STDOUT:   %.loc8_30.5: init f64 = initialize_from %.loc8_20 to %.loc8_30.4
// CHECK:STDOUT:   %.loc8_30.6: i32 = int_literal 1
// CHECK:STDOUT:   %.loc8_30.7: ref f64 = array_index %b, %.loc8_30.6
// CHECK:STDOUT:   %.loc8_30.8: init f64 = initialize_from %.loc8_26 to %.loc8_30.7
// CHECK:STDOUT:   %.loc8_30.9: init [f64; 2] = array_init %.loc8_30.2, (%.loc8_30.5, %.loc8_30.8) to %b
// CHECK:STDOUT:   assign %b, %.loc8_30.9
// CHECK:STDOUT:   %.loc9_10.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc9_10.2: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_13: i32 = int_literal 5
// CHECK:STDOUT:   %.loc9_14.1: type = array_type %.loc9_13, ()
// CHECK:STDOUT:   %.loc9_14.2: type = ptr_type [(); 5]
// CHECK:STDOUT:   %c: ref [(); 5] = var "c"
// CHECK:STDOUT:   %.loc9_20.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_24.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_28.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_32.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_36.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_38.1: type = tuple_type ((), (), (), (), ())
// CHECK:STDOUT:   %.loc9_38.2: ((), (), (), (), ()) = tuple_literal (%.loc9_20.1, %.loc9_24.1, %.loc9_28.1, %.loc9_32.1, %.loc9_36.1)
// CHECK:STDOUT:   %.loc9_20.2: init () = tuple_init %.loc9_20.1, ()
// CHECK:STDOUT:   %.loc9_24.2: init () = tuple_init %.loc9_24.1, ()
// CHECK:STDOUT:   %.loc9_28.2: init () = tuple_init %.loc9_28.1, ()
// CHECK:STDOUT:   %.loc9_32.2: init () = tuple_init %.loc9_32.1, ()
// CHECK:STDOUT:   %.loc9_36.2: init () = tuple_init %.loc9_36.1, ()
// CHECK:STDOUT:   %.loc9_38.3: init [(); 5] = array_init %.loc9_38.2, (%.loc9_20.2, %.loc9_24.2, %.loc9_28.2, %.loc9_32.2, %.loc9_36.2) to %c
// CHECK:STDOUT:   assign %c, %.loc9_38.3
// CHECK:STDOUT: }
