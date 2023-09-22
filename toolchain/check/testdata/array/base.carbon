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
// CHECK:STDOUT:   %.loc7_15: type = array_type %.loc7_14, i32
// CHECK:STDOUT:   %a: ref [i32; 1] = var "a"
// CHECK:STDOUT:   %.loc7_20: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_22.1: type = tuple_type (i32)
// CHECK:STDOUT:   %.loc7_22.2: (i32,) = tuple_literal (%.loc7_20)
// CHECK:STDOUT:   %.loc7_22.3: ref [i32; 1] = stub_reference %a
// CHECK:STDOUT:   %.loc7_22.4: i32 = int_literal 0
// CHECK:STDOUT:   %.loc7_22.5: ref i32 = array_index %.loc7_22.3, %.loc7_22.4
// CHECK:STDOUT:   %.loc7_22.6: init i32 = initialize_from %.loc7_20 to %.loc7_22.5
// CHECK:STDOUT:   %.loc7_22.7: init [i32; 1] = array_init %.loc7_22.2, (%.loc7_22.6) to %.loc7_22.3
// CHECK:STDOUT:   assign %a, %.loc7_22.7
// CHECK:STDOUT:   %.loc8_14: i32 = int_literal 2
// CHECK:STDOUT:   %.loc8_15: type = array_type %.loc8_14, f64
// CHECK:STDOUT:   %b: ref [f64; 2] = var "b"
// CHECK:STDOUT:   %.loc8_20: f64 = real_literal 111e-1
// CHECK:STDOUT:   %.loc8_26: f64 = real_literal 22e-1
// CHECK:STDOUT:   %.loc8_30.1: type = tuple_type (f64, f64)
// CHECK:STDOUT:   %.loc8_30.2: (f64, f64) = tuple_literal (%.loc8_20, %.loc8_26)
// CHECK:STDOUT:   %.loc8_30.3: ref [f64; 2] = stub_reference %b
// CHECK:STDOUT:   %.loc8_30.4: i32 = int_literal 0
// CHECK:STDOUT:   %.loc8_30.5: ref f64 = array_index %.loc8_30.3, %.loc8_30.4
// CHECK:STDOUT:   %.loc8_30.6: init f64 = initialize_from %.loc8_20 to %.loc8_30.5
// CHECK:STDOUT:   %.loc8_30.7: i32 = int_literal 1
// CHECK:STDOUT:   %.loc8_30.8: ref f64 = array_index %.loc8_30.3, %.loc8_30.7
// CHECK:STDOUT:   %.loc8_30.9: init f64 = initialize_from %.loc8_26 to %.loc8_30.8
// CHECK:STDOUT:   %.loc8_30.10: init [f64; 2] = array_init %.loc8_30.2, (%.loc8_30.6, %.loc8_30.9) to %.loc8_30.3
// CHECK:STDOUT:   assign %b, %.loc8_30.10
// CHECK:STDOUT:   %.loc9_10.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc9_10.2: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_13: i32 = int_literal 5
// CHECK:STDOUT:   %.loc9_14: type = array_type %.loc9_13, ()
// CHECK:STDOUT:   %c: ref [(); 5] = var "c"
// CHECK:STDOUT:   %.loc9_20: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_24: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_28: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_32: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_36: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_38.1: type = tuple_type ((), (), (), (), ())
// CHECK:STDOUT:   %.loc9_38.2: ((), (), (), (), ()) = tuple_literal (%.loc9_20, %.loc9_24, %.loc9_28, %.loc9_32, %.loc9_36)
// CHECK:STDOUT:   %.loc9_38.3: ref [(); 5] = stub_reference %c
// CHECK:STDOUT:   %.loc9_38.4: i32 = int_literal 0
// CHECK:STDOUT:   %.loc9_38.5: ref () = array_index %.loc9_38.3, %.loc9_38.4
// CHECK:STDOUT:   %.loc9_38.6: init () = tuple_init %.loc9_20, ()
// CHECK:STDOUT:   %.loc9_38.7: i32 = int_literal 1
// CHECK:STDOUT:   %.loc9_38.8: ref () = array_index %.loc9_38.3, %.loc9_38.7
// CHECK:STDOUT:   %.loc9_38.9: init () = tuple_init %.loc9_24, ()
// CHECK:STDOUT:   %.loc9_38.10: i32 = int_literal 2
// CHECK:STDOUT:   %.loc9_38.11: ref () = array_index %.loc9_38.3, %.loc9_38.10
// CHECK:STDOUT:   %.loc9_38.12: init () = tuple_init %.loc9_28, ()
// CHECK:STDOUT:   %.loc9_38.13: i32 = int_literal 3
// CHECK:STDOUT:   %.loc9_38.14: ref () = array_index %.loc9_38.3, %.loc9_38.13
// CHECK:STDOUT:   %.loc9_38.15: init () = tuple_init %.loc9_32, ()
// CHECK:STDOUT:   %.loc9_38.16: i32 = int_literal 4
// CHECK:STDOUT:   %.loc9_38.17: ref () = array_index %.loc9_38.3, %.loc9_38.16
// CHECK:STDOUT:   %.loc9_38.18: init () = tuple_init %.loc9_36, ()
// CHECK:STDOUT:   %.loc9_38.19: init [(); 5] = array_init %.loc9_38.2, (%.loc9_38.6, %.loc9_38.9, %.loc9_38.12, %.loc9_38.15, %.loc9_38.18) to %.loc9_38.3
// CHECK:STDOUT:   assign %c, %.loc9_38.19
// CHECK:STDOUT: }
