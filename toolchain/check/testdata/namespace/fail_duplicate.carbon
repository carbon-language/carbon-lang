// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_duplicate.carbon" {

namespace Foo;
// CHECK:STDOUT:   %.loc[[@LINE-1]] = namespace {.Baz = %.loc[[@LINE+2]]}

fn Foo.Baz() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Baz
}

// CHECK:STDERR: fail_duplicate.carbon:[[@LINE+6]]:8: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn Foo.Baz() {
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_duplicate.carbon:[[@LINE-7]]:1: Name is previously declared here.
// CHECK:STDERR: fn Foo.Baz() {
// CHECK:STDERR: ^
fn Foo.Baz() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Baz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
}
