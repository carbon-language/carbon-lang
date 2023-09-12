// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "base.carbon" {

var a: [i32; 1] = (1,);
// CHECK:STDOUT:   %.loc[[@LINE-1]]_14: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-2]]_15: type = array_type %.loc[[@LINE-2]]_14, i32
// CHECK:STDOUT:   %a: ref [i32; 1] = var "a"
// CHECK:STDOUT:   %.loc[[@LINE-4]]_20.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-5]]_20.2: i32 = stub_reference %.loc[[@LINE-5]]_20.1
// CHECK:STDOUT:   %.loc[[@LINE-6]]_22.1: type = tuple_type (i32)
// CHECK:STDOUT:   %.loc[[@LINE-7]]_22.2: (i32,) = tuple_literal (%.loc[[@LINE-7]]_20.2)
// CHECK:STDOUT:   %.loc[[@LINE-8]]_22.3: [i32; 1] = array_value %.loc[[@LINE-8]]_22.2
// CHECK:STDOUT:   assign %a, %.loc[[@LINE-9]]_22.3
var b: [f64; 2] = (11.1, 2.2,);
// CHECK:STDOUT:   %.loc[[@LINE-1]]_14: i32 = int_literal 2
// CHECK:STDOUT:   %.loc[[@LINE-2]]_15: type = array_type %.loc[[@LINE-2]]_14, f64
// CHECK:STDOUT:   %b: ref [f64; 2] = var "b"
// CHECK:STDOUT:   %.loc[[@LINE-4]]_20.1: f64 = real_literal 111e-1
// CHECK:STDOUT:   %.loc[[@LINE-5]]_20.2: f64 = stub_reference %.loc[[@LINE-5]]_20.1
// CHECK:STDOUT:   %.loc[[@LINE-6]]_26.1: f64 = real_literal 22e-1
// CHECK:STDOUT:   %.loc[[@LINE-7]]_26.2: f64 = stub_reference %.loc[[@LINE-7]]_26.1
// CHECK:STDOUT:   %.loc[[@LINE-8]]_30.1: type = tuple_type (f64, f64)
// CHECK:STDOUT:   %.loc[[@LINE-9]]_30.2: (f64, f64) = tuple_literal (%.loc[[@LINE-9]]_20.2, %.loc[[@LINE-9]]_26.2)
// CHECK:STDOUT:   %.loc[[@LINE-10]]_30.3: [f64; 2] = array_value %.loc[[@LINE-10]]_30.2
// CHECK:STDOUT:   assign %b, %.loc[[@LINE-11]]_30.3
var c: [(); 5] = ((), (), (), (), (),);
// CHECK:STDOUT:   %.loc[[@LINE-1]]_10.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc[[@LINE-2]]_10.2: () = tuple_literal ()
// CHECK:STDOUT:   %.loc[[@LINE-3]]_13: i32 = int_literal 5
// CHECK:STDOUT:   %.loc[[@LINE-4]]_14: type = array_type %.loc[[@LINE-4]]_13, ()
// CHECK:STDOUT:   %c: ref [(); 5] = var "c"
// CHECK:STDOUT:   %.loc[[@LINE-6]]_20.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc[[@LINE-7]]_20.2: () = stub_reference %.loc[[@LINE-7]]_20.1
// CHECK:STDOUT:   %.loc[[@LINE-8]]_24.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc[[@LINE-9]]_24.2: () = stub_reference %.loc[[@LINE-9]]_24.1
// CHECK:STDOUT:   %.loc[[@LINE-10]]_28.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc[[@LINE-11]]_28.2: () = stub_reference %.loc[[@LINE-11]]_28.1
// CHECK:STDOUT:   %.loc[[@LINE-12]]_32.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc[[@LINE-13]]_32.2: () = stub_reference %.loc[[@LINE-13]]_32.1
// CHECK:STDOUT:   %.loc[[@LINE-14]]_36.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc[[@LINE-15]]_36.2: () = stub_reference %.loc[[@LINE-15]]_36.1
// CHECK:STDOUT:   %.loc[[@LINE-16]]_38.1: type = tuple_type ((), (), (), (), ())
// CHECK:STDOUT:   %.loc[[@LINE-17]]_38.2: ((), (), (), (), ()) = tuple_literal (%.loc[[@LINE-17]]_20.2, %.loc[[@LINE-17]]_24.2, %.loc[[@LINE-17]]_28.2, %.loc[[@LINE-17]]_32.2, %.loc[[@LINE-17]]_36.2)
// CHECK:STDOUT:   %.loc[[@LINE-18]]_38.3: [(); 5] = array_value %.loc[[@LINE-18]]_38.2
// CHECK:STDOUT:   assign %c, %.loc[[@LINE-19]]_38.3
// CHECK:STDOUT: }
