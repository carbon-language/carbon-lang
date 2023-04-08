// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

abstract class A1 {
  abstract fn Foo[self:Self]() -> i32;
}

abstract class A2 extends A1 {}

base class B1 extends A1 {}

base class B2 extends A1 {
  impl fn Foo[self:Self]() -> i32 {
    return 0;
  }
}

base class B3 extends A2 {
  impl fn Foo[self:Self]() -> i32 {
    return 1;
  }
}

class C1 extends B1 {
  impl fn Foo[self:Self]() -> i32 {
    return 2;
  }
}

class C2 extends B2 {}

class C3 extends B2 {
  impl fn Foo[self:Self]() -> i32 {
    return 3;
  }
}

class C4 extends B3 {
  impl fn Foo[self:Self]() -> i32 {
    return 4;
  }
}

fn Main() -> i32 {
  let b2: B2 = {.base = {}};
  let b3: B3 = {.base = {.base = {}}};
  let c1: C1 = {.base = {.base = {}}};
  let c2: C2 = {.base = {.base = {}}};
  let c3: C3 = {.base = {.base = {}}};
  let c4: C4 = {.base = {.base = {.base = {}}}};

  Print("b2.Foo() -> {0}", b2.Foo());
  Print("b3.Foo() -> {0}", b3.Foo());
  Print("c1.Foo() -> {0}", c1.Foo());
  Print("c2.Foo() -> {0}", c2.Foo());
  Print("c3.Foo() -> {0}", c3.Foo());
  Print("c4.Foo() -> {0}", c4.Foo());

  return 0;
}
