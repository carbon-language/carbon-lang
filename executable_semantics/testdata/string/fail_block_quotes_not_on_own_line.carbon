// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{executable_semantics} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{executable_semantics} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s

package ExecutableSemanticsTest api;

fn Main() -> i32 {
  // CHECK: COMPILATION ERROR: {{.*}}/executable_semantics/testdata/string/fail_block_quotes_not_on_own_line.carbon:[[@LINE+1]]: Invalid block string: Should end with triple quotes: error: closing """
  var s: String = """
    error: closing """ is not on its own line.
  """;

  return 0;
}
