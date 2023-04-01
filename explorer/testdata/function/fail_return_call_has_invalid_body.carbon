// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package EmptyIdentifier impl;

fn apply[T:! type, U:! type](f: T, EmptyIdentifier: U) {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/function/fail_return_call_has_invalid_body.carbon:[[@LINE+1]]: only arrays and tuples can be indexed, found bool
  match (true[true]) {}
}

fn EmptyIdentifier() -> apply(true, true);

fn Main() -> i32 {
  return 0;
}
