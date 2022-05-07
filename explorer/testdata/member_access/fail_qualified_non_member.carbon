// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

package Foo api;
fn F[me: i32]() {}
fn Main() -> i32 {
  // TODO: It's unclear whether this is valid per the current rules. See
  // https://github.com/carbon-language/carbon-lang/pull/1122
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/member_access/fail_qualified_non_member.carbon:[[@LINE+1]]: expected member name in compound member access, found fn () -> ()
  42.(F)();
  return 0;
}
