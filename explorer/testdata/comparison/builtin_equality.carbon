// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: strings equal: 0
// CHECK: ints equal: 1
// CHECK: result: 0

package ExplorerTest api;

fn Main() -> i32 {
  Print("strings equal: {0}", if "hello" == "world" then 1 else 0);
  Print("ints equal: {0}", if 1 == 1 then 1 else 0);
  return 0;
}
