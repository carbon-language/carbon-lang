// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

choice Ints {
  None,
  One(i32),
  Two(i32, i32)
}

fn Main() -> i32 {
  var x: auto = Ints.None();
  var n: auto = 0;
  match (x) {
    case Ints.One(var x: auto) => {
      x = 2;
    }
    case Ints.None() => {
      n = n - 1;
    }
    case Ints.Two(x: auto, y: auto) => {
      // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/let/fail_match_choice.carbon:[[@LINE+1]]: Cannot assign to rvalue 'x'
      x = 0;
    }
  }
  return 0;
}
