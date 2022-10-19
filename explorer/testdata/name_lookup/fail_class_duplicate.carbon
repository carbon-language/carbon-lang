// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class Foo {
  var x: i32;
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/name_lookup/fail_class_duplicate.carbon:[[@LINE+1]]: Duplicate name `x` also found at {{.*}}/explorer/testdata/name_lookup/fail_class_duplicate.carbon:[[@LINE-1]]
  var x: i32;
}

fn Main() -> i32 {
  return 0;
}
