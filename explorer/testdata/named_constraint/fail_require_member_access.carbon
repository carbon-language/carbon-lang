// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface HasFoo {
  fn Foo();
}

constraint ImplAsHasFoo {
  require Self impls HasFoo;
}

fn CallFoo[T:! ImplAsHasFoo](x: T) {
  // OK, T is HasFoo.
  x.(HasFoo.Foo)();
  // CHECK:STDERR: COMPILATION ERROR: fail_require_member_access.carbon:[[@LINE+1]]: member access, Foo not in constraint ImplAsHasFoo where T impls interface HasFoo
  x.Foo();
}

fn Main() -> i32 { return 0; }
