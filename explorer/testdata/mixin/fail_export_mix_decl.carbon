// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{not} %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
package ExplorerTest api;
__mixin M {
  export fn Square[me: Self](x:i32) -> i32 {
    return x * x;
  }
}

__mixin M1 {
  export fn Cube[me: Self](x:i32) -> i32 {
    return x * me.Square(x);
  }

  __mix M;
}

__mixin M2 {
  export fn Tesseract[me: Self](x:i32) -> i32 {
// CHECK:COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail_export_mix_decl.carbon:[[@LINE+1]]: mixin M2 does not have a field named Square
    return me.Square(x) * me.Square(x);
  }
  __mix M1;
}

fn Main() -> i32 {
  return 0;
}
