// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface CallF {
  fn DoIt[self: Self]();
}

interface HasF {
  fn F[self: Self]();
}

impl forall [template T:! HasF] T as CallF {
  // TODO: This case should be accepted, using `ClassWithExternalF.(HasF.F)`.
  // TODO: The other case should be rejected due to ambiguity.
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/template/fail_name_lookup.carbon:[[@LINE+1]]: class ClassWithExternalF does not have a field named F
  fn DoIt[self: Self]() { self.F(); }
}

class ClassWithInternalF {
  fn F[self: Self]() { Print("ClassWithInternalF.F"); }
}

external impl ClassWithInternalF as HasF {
  fn F[self: Self]() { Print("ClassWithInternalF.(HasF.F)"); }
}

class ClassWithExternalF {}

external impl ClassWithExternalF as HasF {
  fn F[self: Self]() { Print("ClassWithExternalF.(HasF.F)"); }
}

fn Main() -> i32 {
  var a: ClassWithInternalF = {};
  var b: ClassWithExternalF = {};
  a.(CallF.DoIt)();
  b.(CallF.DoIt)();
  return 0;
}
