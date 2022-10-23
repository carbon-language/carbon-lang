// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface A {
  let TA:! Type;
  fn FA() -> TA;
}
interface B {
  let TB:! Type;
  fn FB() -> TB;
}

class C(T:! Type) {
  impl as A & B where .TA = i32 and .TB = i32 {
    fn FA() -> i32 {
      // OK, know that TA is i32 here.
      let v: Self.(A.TA) = 1;
      let w: i32 = v;
      return w;
    }
    fn FB() -> i32 {
      // OK, know that TB is i32 here.
      let v: Self.(B.TB) = 2;
      // Don't know that TA is i32; it could be specialized.
      // TODO: We should not accept this.
      let w: Self.(A.TA) = 3;
      return v + w;
    }
  }
}

external impl C(i32) as B where .TB == () {
  fn FB() -> () { return (); }
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_multi_impl_scoping.carbon:[[@LINE+1]]: ambiguous implementations of interface B for class C(T = i32)
}

fn Main() -> i32 { return C(i32).FB(); }
