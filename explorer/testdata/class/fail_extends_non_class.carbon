// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class C extends 3 {
  var x: i32;
  var y: i32;
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_extends_non_class.carbon:[[@LINE+1]]: Unsupported base class type for class `C`. Only simple classes are currently supported as base class.
}

fn Main() -> i32 {
  return 0;
}
