// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 0

package ExplorerTest api;

// Ensure that we can declare all different kinds of declarations as namespace
// members.
namespace N;

__mixin N.Mixin {}

base class N.BaseClass {
  __mix Mixin;
}

class N.DerivedClass extends BaseClass {}

namespace N.Inner;

alias N.Inner.InnerClass = DerivedClass;

choice N.Choice { A, B, C(DerivedClass) }

interface N.Iface {
  fn F() -> DerivedClass*;
}

constraint N.Constraint {
  extends Iface;
}

var v: N.DerivedClass = {.base = {}};

fn N.F[T:! Constraint]() {
  var v: Inner.InnerClass = {.base = {}};
}

impl i32 as N.Constraint {
  fn F() -> N.DerivedClass* { return &v; }
}

fn Main() -> i32 { return 0; }
