// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "empty.carbon" {

var x: () = ();
// CHECK:STDOUT:   %.loc[[@LINE-1]]_9.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc[[@LINE-2]]_9.2: () = tuple_literal ()
// CHECK:STDOUT:   %x: ref () = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-4]]_14: () = tuple_literal ()
// CHECK:STDOUT:   assign %x, %.loc[[@LINE-5]]_14
var y: () = x;
// CHECK:STDOUT:   %.loc[[@LINE-1]]: () = tuple_literal ()
// CHECK:STDOUT:   %y: ref () = var "y"
// CHECK:STDOUT:   %.loc[[@LINE-9]]_5: () = bind_value %x
// CHECK:STDOUT:   assign %y, %.loc[[@LINE-10]]_5
// CHECK:STDOUT: }
