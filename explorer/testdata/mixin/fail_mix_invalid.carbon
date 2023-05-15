// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

__mixin Mixin {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/mixin/fail_mix_invalid.carbon:[[@LINE+1]]: Not a valid mixin: `()`
  __mix ();
}

fn Main() -> i32 {
  return 0;
}
