// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

interface Iface {
  let T:! Type;
}

fn F(T:! Iface where .T == i32) {}

class Good {}
class Bad {}
external impl Good as Iface where .T = i32 {}
external impl Bad as Iface where .T = Bad {}

fn Main() -> i32 {
  F(Good);
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_different_type.carbon:[[@LINE+1]]: could not determine that class Bad == i32
  F(Bad);
  return 0;
}
