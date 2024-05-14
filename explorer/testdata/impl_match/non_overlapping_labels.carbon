// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface HasType {
  let AssocType:! type;
}

interface Iface {
  fn F();
}

impl forall [T:! HasType where .Self.AssocType impls Iface] T as Iface {
  fn F() {
    Print("T as Iface");
    T.AssocType.(Iface.F)();
  }
}

class A {
  extend impl as Iface {
    fn F() { Print("A as Iface"); }
  }
}

class B {
  extend impl as HasType where .AssocType = A {}
}

class C {
  extend impl as HasType where .AssocType = B {}
}

fn Main() -> i32 {
  let c: C = {};
  c.(Iface.F)();
  return 0;
}

// CHECK:STDOUT: T as Iface
// CHECK:STDOUT: T as Iface
// CHECK:STDOUT: A as Iface
// CHECK:STDOUT: result: 0
