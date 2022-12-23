// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

base class A {}

class B extends A {
  virtual fn Foo[self: Self]() -> i32 { return 0; }
}

fn Main() -> i32 {
  var b: B = { .base = {}};
  var bp: B* = &b;
  var ap: A* = bp;
  // Shouldn't look into the vtable for B.
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_virtual_method_absent.carbon:[[@LINE+1]]: class A does not have a field named Foo
  return ap->Foo();
}
