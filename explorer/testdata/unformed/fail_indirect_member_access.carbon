// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var pt: {.x: i32, .y: i32};
  // TODO: Without this, we don't allow taking the address of `pt.x`.
  // That's probably too restrictive.
  if (1 == 0) { pt = {.x = 1, .y = 2}; }
  var p: i32* = &pt.x;
  // CHECK:STDERR: RUNTIME ERROR: fail_indirect_member_access.carbon:[[@LINE+1]]: undefined behavior: access to uninitialized value Uninit<Placeholder<pt>>
  return *p;
}
