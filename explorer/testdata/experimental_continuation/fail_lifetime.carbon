// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

// Test that the read from x triggers an error because x is dead.
// This test also demonstrates how by-reference free-variable capture
// is dangerous and can happen inside continuations.

fn capture() -> __Continuation {
  var x: i32 = 1;
  __continuation k {
    // CHECK:STDERR: RUNTIME ERROR: {{.*}}/explorer/testdata/experimental_continuation/fail_lifetime.carbon:[[@LINE+1]]: undefined behavior: access to dead value 1
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
