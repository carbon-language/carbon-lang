// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

class C {
  fn F() {}
}

class D extends C {
  fn G() {}
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_extends_final_class.carbon:[[@LINE+1]]: Base class `C` is `final` and cannot inherited. Add the `base` or `abstract` class prefix to `C` to allow it to be inherited
}

fn Main() -> i32 {
  return 0;
}
