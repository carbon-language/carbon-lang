// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface A {}
interface B {}
interface C {}

impl forall [T:! type where .Self* impls A] T as B {}
impl forall [T:! type where .Self* impls B] T as C {}
impl forall [T:! type where .Self* impls C] T as A {}

fn F[T:! A](x: T) {}

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/impl_match/fail_three_step_recurse.carbon:[[@LINE+3]]: impl matching recursively performed a more complex match using the same impl
  // CHECK:STDERR:   outer match: i32 as interface A
  // CHECK:STDERR:   inner match: i32*** as interface A
  F(0);
  return 0;
}
