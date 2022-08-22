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

interface I {
  fn Foo();
}

// CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/mixin/fail-unimplemented-mixin-for.carbon:[[@LINE+1]]: 'for' not supported currently
__mixin M1 for I {
  fn F1[me: Self](x: Self) -> Self{
     me.foo();
     return x;
  }
}

class C {
 __mix M1;
}

external impl C as I {
  fn Foo() {}
}

fn Main() -> i32 {
  var c: C = {.v = 0};
  return c.v;
}
