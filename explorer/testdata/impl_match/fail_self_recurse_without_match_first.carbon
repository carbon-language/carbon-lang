// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface Foo {}

impl forall [T:! type where .Self == i32*] T as Foo {}
impl forall [T:! type where .Self* impls Foo] T as Foo {}

fn F[T:! Foo](x: T) {}

fn Main() -> i32 {
  // When checking the second impl, we recursively look for a match for
  // i32* as Foo, which considers the second impl again, even though the first
  // impl would match.
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/impl_match/fail_self_recurse_without_match_first.carbon:[[@LINE+3]]: impl matching recursively performed a more complex match using the same impl
  // CHECK:STDERR:   outer match: i32 as interface Foo
  // CHECK:STDERR:   inner match: i32* as interface Foo
  F(0);
  return 0;
}
