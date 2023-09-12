// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "empty_struct.carbon" {

fn Echo(a: {}) -> {} {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Echo
  return a;
}

fn Main() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Echo(%a: {}) -> {} {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
  Echo({});
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_9.1: {} = struct_literal ()
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_9.2: {} = stub_reference %.loc[[@LINE-2]]_9.1
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_7: init {} = call @Echo(%.loc[[@LINE-3]]_9.2)
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
