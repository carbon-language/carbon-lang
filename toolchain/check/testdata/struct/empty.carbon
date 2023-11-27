// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {} = {};
var y: {} = x;

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7_9.1: type = struct_type {}
// CHECK:STDOUT:   %.loc7_9.2: type = tuple_type ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "empty.carbon" {
// CHECK:STDOUT:   %.loc7_9.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc7_9.2: type = converted %.loc7_9.1, constants.%.loc7_9.1
// CHECK:STDOUT:   %x.var: ref {} = var x
// CHECK:STDOUT:   %x: ref {} = bind_name x, %x.var
// CHECK:STDOUT:   %.loc7_14.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc7_14.2: init {} = struct_init () to %x.var
// CHECK:STDOUT:   %.loc7_14.3: init {} = converted %.loc7_14.1, %.loc7_14.2
// CHECK:STDOUT:   assign %x.var, %.loc7_14.3
// CHECK:STDOUT:   %.loc8_9: {} = struct_literal ()
// CHECK:STDOUT:   %.loc7_9.3: type = converted %.loc8_9, constants.%.loc7_9.1
// CHECK:STDOUT:   %y.var: ref {} = var y
// CHECK:STDOUT:   %y: ref {} = bind_name y, %y.var
// CHECK:STDOUT:   %x.ref: ref {} = name_ref x, %x
// CHECK:STDOUT:   %.loc8_13.1: init {} = struct_init () to %y.var
// CHECK:STDOUT:   %.loc8_13.2: init {} = converted %x.ref, %.loc8_13.1
// CHECK:STDOUT:   assign %y.var, %.loc8_13.2
// CHECK:STDOUT: }
