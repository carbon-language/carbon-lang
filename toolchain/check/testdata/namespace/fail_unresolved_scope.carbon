// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_unresolved_scope.carbon:[[@LINE+3]]:4: ERROR: Name `Foo` not found.
// CHECK:STDERR: fn Foo.Baz() {
// CHECK:STDERR:    ^~~
fn Foo.Baz() {
}

// CHECK:STDOUT: --- fail_unresolved_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT:   %.loc10: <function> = fn_decl @.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
