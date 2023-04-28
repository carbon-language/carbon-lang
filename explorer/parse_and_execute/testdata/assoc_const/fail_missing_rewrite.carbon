// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface Container {
  let Element:! type;
  fn Front[self: Self]() -> Element;
}

fn A[T:! Container where .Element = i32](x: T) -> T.Element {
  return x.Front();
}

fn B[T:! Container](x: T) -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_missing_rewrite.carbon:[[@LINE+1]]: constraint requires that (T).(Container.Element) (with value (T).(Container.Element)) == i32, which is not known to be true
  return A(x);
}

fn Main() -> i32 {
  return 0;
}
