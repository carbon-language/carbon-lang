// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

interface MyAddWith(T:! type) {
  let Result:! type;
  fn Op[self: Self](other: T) -> Result;
}

constraint MyAdd {
  extends MyAddWith(Self) where .Result = Self;
}

external impl i32 as MyAdd {
  fn Op[self: i32](other: i32) -> i32 { return self + other; }
}

fn Main() -> i32 {
  let n: i32 = 1;
  // TODO: This should be valid, but isn't representable in our current
  // MemberName value. We will likely need MemberName to remember the
  // constraint as written, not only the interface in which the member was
  // found.
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/named_constraint/fail_compound_member_access.carbon:[[@LINE+1]]: could not find implementation of interface MyAddWith(T = Self) for i32
  return n.(MyAdd.Op)(n);
}
