// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: Carrot.G
// CHECK:STDOUT: result: 5

package ExplorerTest api;

interface MyAddWith(T:! Type) {
  let Result:! Type;
  fn Op[me: Self](other: T) -> Result;
}

constraint MyAdd {
  extends MyAddWith(Self) where .Result = Self;
}

external impl i32 as MyAdd {
  fn Op[me: i32](other: i32) -> i32 { return me + other; }
}

fn Main() -> i32 {
  let n: i32 = 1;
  // TODO: This should be valid, but isn't representable in our current
  // MemberName value. We will likely need MemberName to remember the
  // constraint as written, not only the interface in which the member was
  // found.
  return n.(MyAdd.Op)(n);
}
