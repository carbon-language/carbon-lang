// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: dump semantics-ir %s
//
// Just check that the raw IR format isn't included in `dump semantics-ir` mode.
// AUTOUPDATE

fn Foo() {
  return;
}

// CHECK:STDOUT: package {
// CHECK:STDOUT:   %(10:1) = fn_decl @Foo
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
