// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: Time elapsed in Parse: {{[0-9]+}}ms

package ExplorerTest api;

interface TestInterface {}

namespace N;

fn N.Foo(n: i32) -> i32 {
  return n + 1;
}

fn Main() -> i32 {
  var x: i32 = N.Foo(0);
  return x;
// CHECK:STDERR: SYNTAX ERROR: full_trace.carbon:[[@LINE+1]]: syntax error, unexpected LESS_LESS
}
