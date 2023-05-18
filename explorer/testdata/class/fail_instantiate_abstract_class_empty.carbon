// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

abstract class C {
  var a: i32;
}

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/class/fail_instantiate_abstract_class_empty.carbon:[[@LINE+1]]: Cannot instantiate abstract class C
  var c: C;
  return 0;
}
