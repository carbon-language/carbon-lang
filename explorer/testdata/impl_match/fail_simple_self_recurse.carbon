// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface Foo {}

impl forall [T:! Foo] T as Foo {}

fn F[T:! Foo](x: T) {}

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/impl_match/fail_simple_self_recurse.carbon:[[@LINE+1]]: impl matching for i32 as interface Foo recursively performed a match for the same type and interface
  F(0);
  return 0;
}
