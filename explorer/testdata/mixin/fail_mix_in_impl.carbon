// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

__mixin Operations {
  fn Square[me: Self](x:i32) -> i32{
    return x * x;
  }
}

interface A {
  fn F();
}

external impl i32 as A {
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail-mix-in-impl.carbon:[[@LINE+1]]: syntax error, unexpected MIX, expecting ALIAS or FN or RIGHT_CURLY_BRACE
  __mix Operations;
  fn F() {}
}

fn Main() -> i32 { return 0; }
