// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {} = {};
var y: {} = x;

// CHECK:STDOUT: file "empty.carbon" {
// CHECK:STDOUT:   %.loc7_9.1: type = struct_type {}
// CHECK:STDOUT:   %.loc7_9.2: {} = struct_literal ()
// CHECK:STDOUT:   %.loc7_9.3: type = tuple_type ()
// CHECK:STDOUT:   %x: ref {} = var "x"
// CHECK:STDOUT:   %.loc7_14.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc7_14.2: init {} = struct_init %.loc7_14.1, ()
// CHECK:STDOUT:   assign %x, %.loc7_14.2
// CHECK:STDOUT:   %.loc8_9: {} = struct_literal ()
// CHECK:STDOUT:   %y: ref {} = var "y"
// CHECK:STDOUT:   %x.ref: ref {} = name_reference "x", %x
// CHECK:STDOUT:   %.loc8_13: init {} = struct_init %x.ref, ()
// CHECK:STDOUT:   assign %y, %.loc8_13
// CHECK:STDOUT: }
