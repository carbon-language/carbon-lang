// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

var my_array : [i32; 1];

fn Main() -> i32 {
  // CHECK:STDERR: RUNTIME ERROR: fail_store_to_uninitialized_global_array.carbon:[[@LINE+1]]: undefined behavior: store to subobject of uninitialized value Uninit<Placeholder<my_array>>
  my_array[0] = 100;
  Print("{0}", my_array[0]);
  return 0;
}
