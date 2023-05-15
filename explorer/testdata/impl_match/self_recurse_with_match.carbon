// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface Foo {}

impl i32* as Foo {}
impl forall [T:! type where .Self* impls Foo] T as Foo {}

fn F[T:! Foo](x: T) {}

fn Main() -> i32 {
  // This is OK: we don't consider the second impl when recursively checking
  // for `impl i32* as Foo` because the first impl is a better match.
  F(0);
  return 0;
}
