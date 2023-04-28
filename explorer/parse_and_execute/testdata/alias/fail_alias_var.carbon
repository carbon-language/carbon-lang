// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

var v: i32;
// TODO: This should probably be valid if we allow global variables at all.
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/alias/fail_alias_var.carbon:[[@LINE+1]]: invalid target for alias declaration
alias A = v;

fn Main() -> i32 { return 0; }
