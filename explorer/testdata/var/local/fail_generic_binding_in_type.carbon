// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/var/local/fail_generic_binding_in_type.carbon:[[@LINE+1]]: Binding types cannot contain bindings, but `T:! type` is a binding.
  var x: (T:! type) = 1;
  return 1;
}
