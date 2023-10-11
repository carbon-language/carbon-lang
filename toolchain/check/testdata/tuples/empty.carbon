// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: () = ();
var y: () = x;

// CHECK:STDOUT: file "empty.carbon" {
// CHECK:STDOUT:   %.loc7_9.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc7_9.2: () = tuple_literal ()
// CHECK:STDOUT:   %x: ref () = var "x"
// CHECK:STDOUT:   %.loc7_14.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_14.2: init () = tuple_init %.loc7_14.1, ()
// CHECK:STDOUT:   assign %x, %.loc7_14.2
// CHECK:STDOUT:   %.loc8_9: () = tuple_literal ()
// CHECK:STDOUT:   %y: ref () = var "y"
// CHECK:STDOUT:   %x.ref: ref () = name_reference "x", %x
// CHECK:STDOUT:   %.loc8_13: init () = tuple_init %x.ref, ()
// CHECK:STDOUT:   assign %y, %.loc8_13
// CHECK:STDOUT: }
