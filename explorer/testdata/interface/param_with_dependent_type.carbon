// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 42

package ExplorerTest api;

interface HasValueParam(T:! Type, V:! T) {
  fn Go[me: Self]() -> T;
}

impl () as HasValueParam(i32, 5) {
  fn Go[me: Self]() -> i32 { return 42; }
}

fn Main() -> i32 {
  return ().(HasValueParam(i32, 5).Go)();
}
