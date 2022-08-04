// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: strings less: 1
// CHECK: ints less: 0
// CHECK: strings less eq: 1
// CHECK: ints less eq: 1
// CHECK: strings greater: 0
// CHECK: ints greater: 0
// CHECK: strings greater eq: 0
// CHECK: ints greater eq: 1
// CHECK: result: 0

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
