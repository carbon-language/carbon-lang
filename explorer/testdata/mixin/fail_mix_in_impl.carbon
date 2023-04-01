// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

__mixin Operations {
  fn Square[self: Self](x:i32) -> i32{
    return x * x;
  }
}

interface A {
  fn F();
}

external impl i32 as A {
  // CHECK:STDERR: SYNTAX ERROR: {{.*}}/explorer/testdata/mixin/fail_mix_in_impl.carbon:[[@LINE+1]]: syntax error, unexpected MIX
  __mix Operations;
  fn F() {}
}

fn Main() -> i32 { return 0; }
