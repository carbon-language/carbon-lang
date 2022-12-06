// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package Foo api;
fn F[me: i32]() {}
fn Main() -> i32 {
  // TODO: It's unclear whether this is valid per the current rules. See
  // https://github.com/carbon-language/carbon-lang/pull/1122
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/member_access/fail_qualified_non_member.carbon:[[@LINE+1]]: expected name of instance member or interface member in compound member access, found fn () -> ()
  42.(F)();
  return 0;
}
