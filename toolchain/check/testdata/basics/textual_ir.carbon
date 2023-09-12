// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=check --dump-semantics-ir %s
//
// Check that the command-line flag to dump textual IR works.
//
// AUTOUPDATE
// CHECK:STDERR: ERROR: Unknown option '--dump-semantics-ir'

fn Foo(n: i32) -> (i32, f64) {
  return (n + 2, 3.4);
}
