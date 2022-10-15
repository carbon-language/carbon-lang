// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

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
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_different_value.carbon:[[@LINE+1]]: could not determine that 4 == 5
  F(Bad);
  return 0;
}
