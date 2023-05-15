// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest impl;

fn Main() -> i32 {
  var my_array : [i32; 1];
  // CHECK:STDERR: RUNTIME ERROR: explorer/testdata/array/fail_print_uninitalized_array_element.carbon:[[@LINE+1]]: Printing uninitialized value
  Print("{0}", my_array[0]);
  return 0;
}
