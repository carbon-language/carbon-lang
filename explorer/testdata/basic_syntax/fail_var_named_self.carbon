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

fn Main() -> i32 {
  // Error: can't use keyword `Self` as the name of a variable.
  // TODO: Current error message is unclear, better would be to say
  // something like: unexpected `Self`, expecting identifier
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/basic_syntax/fail_var_named_self.carbon:[[@LINE+1]]: syntax error, unexpected COLON, expecting SLASH or binary *
  var Self : i32 = 0;
  return Self;
}
