// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/tuple/fail_type_tuple_as_type.carbon:[[@LINE+1]]: value of generic binding T is not known
fn F[T:! ((), ())](x: T) -> () {
  return x[0];
}

fn Main() -> i32 {
  F(((), ()));
  return 0;
}
