// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: strings less: 1
// CHECK:STDOUT: ints less: 0
// CHECK:STDOUT: strings less eq: 1
// CHECK:STDOUT: ints less eq: 1
// CHECK:STDOUT: strings greater: 0
// CHECK:STDOUT: ints greater: 0
// CHECK:STDOUT: strings greater eq: 0
// CHECK:STDOUT: ints greater eq: 1
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn Main() -> i32 {
  Print("strings less: {0}", if "hello" < "world" then 1 else 0);
  Print("ints less: {0}", if 1 < 1 then 1 else 0);
  Print("strings less eq: {0}", if "hello" <= "world" then 1 else 0);
  Print("ints less eq: {0}", if 1 <= 1 then 1 else 0);
  Print("strings greater: {0}", if "hello" > "world" then 1 else 0);
  Print("ints greater: {0}", if 1 > 1 then 1 else 0);
  Print("strings greater eq: {0}", if "hello" >= "world" then 1 else 0);
  Print("ints greater eq: {0}", if 1 >= 1 then 1 else 0);
  return 0;
}
