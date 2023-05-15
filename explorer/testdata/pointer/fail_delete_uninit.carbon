// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var p: i32*;
  // CHECK:STDERR: RUNTIME ERROR: explorer/testdata/pointer/fail_delete_uninit.carbon:[[@LINE+1]]: undefined behavior: access to uninitialized value Uninit<Placeholder<p>>
  heap.Delete(p);
  return 0;
}
