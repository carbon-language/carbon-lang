// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "simple.carbon" {

fn F();
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F

fn G() { F(); }
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc[[@LINE-8]]_11.1: type = tuple_type ()
// CHECK:STDOUT:   %.loc[[@LINE-9]]_11.2: init () = call @F()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
