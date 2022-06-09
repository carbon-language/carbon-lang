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

interface A {
  // TODO: Use `let T:! Type;` here once we support it.
  fn T();
}

external impl i32 as A {
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/impl/fail_bad_member_kind.carbon:[[@LINE+1]]: syntax error, unexpected CLASS, expecting ALIAS or FN or RIGHT_CURLY_BRACE
  class T {}
}

fn Main() -> i32 { return 0; }
