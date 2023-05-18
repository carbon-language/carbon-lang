// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface Foo {}

impl forall [T:! type where {.a: .Self} impls Foo] T as Foo {}

fn F[T:! Foo](x: T) {}

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/impl_match/fail_struct_recurse.carbon:[[@LINE+3]]: impl matching recursively performed a more complex match using the same impl
  // CHECK:STDERR:   outer match: i32 as interface Foo
  // CHECK:STDERR:   inner match: {.a: i32} as interface Foo
  F(0);
  return 0;
}
