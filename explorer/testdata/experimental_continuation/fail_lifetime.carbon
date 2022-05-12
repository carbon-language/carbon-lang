// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: RUNTIME ERROR: {{.*}}/explorer/testdata/experimental_continuation/fail_lifetime.carbon:21: undefined behavior: access to dead value 1

package ExplorerTest api;

// Test that the read from x triggers an error because x is dead.
// This test also demonstrates how by-reference free-variable capture
// is dangerous and can happen inside continuations.

fn capture() -> __Continuation {
  var x: i32 = 1;
  __continuation k {
    var y: i32 = x;
  }
  return k;
}

fn Main() -> i32 {
  var k: __Continuation = capture();
  // error, lifetime of x is over
  __run k;
  return 0;
}
