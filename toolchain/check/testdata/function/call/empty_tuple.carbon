// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Echo(a: ()) -> () {
  return a;
}

fn Main() {
  Echo(());
}

// CHECK:STDOUT: file "empty_tuple.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @Echo
// CHECK:STDOUT:   %.loc11 = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Echo(%a: ()) -> () {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc12_9.2: () = tuple_value %.loc12_9.1, ()
// CHECK:STDOUT:   %.loc12_7: init () = call @Echo(%.loc12_9.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
