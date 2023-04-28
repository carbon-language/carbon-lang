// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface Iface {
  let N:! i32;
}

fn F(T:! Iface where .N == 5) {}

class Good {}
class Bad {}
external impl Good as Iface where .N = 5 {}
external impl Bad as Iface where .N = 4 {}

fn Main() -> i32 {
  F(Good);
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_different_value.carbon:[[@LINE+1]]: constraint requires that (T).(Iface.N) (with value 4) == 5, which is not known to be true
  F(Bad);
  return 0;
}
