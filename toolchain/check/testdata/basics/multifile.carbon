// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon
fn A() {}
// CHECK:STDOUT: file "a.carbon" {
// CHECK:STDOUT:   %.loc[[@LINE-2]] = fn_decl @A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }

// --- b.carbon
fn B() {}
// CHECK:STDOUT: file "b.carbon" {
// CHECK:STDOUT:   %.loc[[@LINE-2]] = fn_decl @B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
