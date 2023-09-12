// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "return_implicit.carbon" {

fn MakeImplicitEmptyTuple() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @MakeImplicitEmptyTuple
}

fn Main() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeImplicitEmptyTuple() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
  var b: () = MakeImplicitEmptyTuple();
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_11.1: type = tuple_type ()
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_11.2: () = tuple_literal ()
  // CHECK:STDOUT:   %b: ref () = var "b"
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_37: init () = call @MakeImplicitEmptyTuple()
  // CHECK:STDOUT:   assign %b, %.loc[[@LINE-5]]_37
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
