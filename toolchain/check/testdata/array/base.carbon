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
// CHECK:STDOUT:   %.loc7_20.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_20.2: i32 = stub_reference %.loc7_20.1
// CHECK:STDOUT:   %.loc7_22.1: type = tuple_type (i32)
// CHECK:STDOUT:   %.loc7_22.2: (i32,) = tuple_literal (%.loc7_20.2)
// CHECK:STDOUT:   %.loc7_22.3: [i32; 1] = array_value %.loc7_22.2
// CHECK:STDOUT:   assign %a, %.loc7_22.3
// CHECK:STDOUT:   %.loc8_14: i32 = int_literal 2
// CHECK:STDOUT:   %.loc8_15: type = array_type %.loc8_14, f64
// CHECK:STDOUT:   %b: ref [f64; 2] = var "b"
// CHECK:STDOUT:   %.loc8_20.1: f64 = real_literal 111e-1
// CHECK:STDOUT:   %.loc8_20.2: f64 = stub_reference %.loc8_20.1
// CHECK:STDOUT:   %.loc8_26.1: f64 = real_literal 22e-1
// CHECK:STDOUT:   %.loc8_26.2: f64 = stub_reference %.loc8_26.1
// CHECK:STDOUT:   %.loc8_30.1: type = tuple_type (f64, f64)
// CHECK:STDOUT:   %.loc8_30.2: (f64, f64) = tuple_literal (%.loc8_20.2, %.loc8_26.2)
// CHECK:STDOUT:   %.loc8_30.3: [f64; 2] = array_value %.loc8_30.2
// CHECK:STDOUT:   assign %b, %.loc8_30.3
// CHECK:STDOUT:   %.loc9_10.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc9_10.2: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_13: i32 = int_literal 5
// CHECK:STDOUT:   %.loc9_14: type = array_type %.loc9_13, ()
// CHECK:STDOUT:   %c: ref [(); 5] = var "c"
// CHECK:STDOUT:   %.loc9_20.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_20.2: () = stub_reference %.loc9_20.1
// CHECK:STDOUT:   %.loc9_24.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_24.2: () = stub_reference %.loc9_24.1
// CHECK:STDOUT:   %.loc9_28.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_28.2: () = stub_reference %.loc9_28.1
// CHECK:STDOUT:   %.loc9_32.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_32.2: () = stub_reference %.loc9_32.1
// CHECK:STDOUT:   %.loc9_36.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_36.2: () = stub_reference %.loc9_36.1
// CHECK:STDOUT:   %.loc9_38.1: type = tuple_type ((), (), (), (), ())
// CHECK:STDOUT:   %.loc9_38.2: ((), (), (), (), ()) = tuple_literal (%.loc9_20.2, %.loc9_24.2, %.loc9_28.2, %.loc9_32.2, %.loc9_36.2)
// CHECK:STDOUT:   %.loc9_38.3: [(); 5] = array_value %.loc9_38.2
// CHECK:STDOUT:   assign %c, %.loc9_38.3
// CHECK:STDOUT: }
