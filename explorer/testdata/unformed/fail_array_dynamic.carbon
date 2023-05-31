// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  var v: [i32; 2];
  if (0 == 1) {
    v = (1, 2);
  }
  // CHECK:STDERR: RUNTIME ERROR: fail_array_dynamic.carbon:[[@LINE+1]]: undefined behavior: access to uninitialized value Uninit<Placeholder<v>>
  return v[0];
}
