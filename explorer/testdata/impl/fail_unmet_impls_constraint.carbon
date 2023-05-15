// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface A {}
interface B {}

// CHECK:STDERR: COMPILATION ERROR: explorer/testdata/impl/fail_unmet_impls_constraint.carbon:[[@LINE+1]]: could not find implementation of interface B for i32
external impl i32 as A where i32 impls B {}

fn Main() -> i32 {
  return 0;
}
