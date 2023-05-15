// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: MemberF.F
// CHECK:STDOUT: ImplF.(HasF.F)
// CHECK:STDOUT: BothFs.(HasF.F)
// CHECK:STDOUT: BothFs.F
// CHECK:STDOUT: BothFs.F
// CHECK:STDOUT: result: 0

package ExplorerTest api;

choice ImplKind {
  Checked,
  ConstrainedTemplate,
  UnconstrainedTemplate
}

interface CallF(K:! ImplKind) {
  fn DoIt[self: Self]();
}

interface HasF {
  fn F[self: Self]();
}

impl forall [T:! HasF] T as CallF(ImplKind.Checked) {
  fn DoIt[self: Self]() { self.F(); }
}

impl forall [template T:! HasF] T as CallF(ImplKind.ConstrainedTemplate) {
  fn DoIt[self: Self]() { self.F(); }
}

impl forall [template T:! type] T as CallF(ImplKind.UnconstrainedTemplate) {
  fn DoIt[self: Self]() { self.F(); }
}

class MemberF {
  fn F[self: Self]() { Print("MemberF.F"); }
}

class ImplF {}
external impl ImplF as HasF {
  fn F[self: Self]() { Print("ImplF.(HasF.F)"); }
}

class BothFs {
  fn F[self: Self]() { Print("BothFs.F"); }
}
external impl BothFs as HasF {
  fn F[self: Self]() { Print("BothFs.(HasF.F)"); }
}

fn Main() -> i32 {
  var mem: MemberF = {};
  var imp: ImplF = {};
  var both: BothFs = {};

  mem.(CallF(ImplKind.UnconstrainedTemplate).DoIt)();

  imp.(CallF(ImplKind.Checked).DoIt)();
  // TODO: Should be valid, but currently fails during instantiation.
  //imp.(CallF(ImplKind.ConstrainedTemplate).DoIt)();

  both.(CallF(ImplKind.Checked).DoIt)();
  // TODO: Should be rejected, but currently incorrectly accepted.
  // This line can be deleted once it starts failing; we test that this is
  // rejected in fail_name_lookup.carbon.
  both.(CallF(ImplKind.ConstrainedTemplate).DoIt)();
  both.(CallF(ImplKind.UnconstrainedTemplate).DoIt)();
  return 0;
}
