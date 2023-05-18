// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/tuple/fail_length_mismatch_with_auto.carbon:[[@LINE+1]]: tuples of different length
  var a: (auto,) = (1, 2);
  return 0;
}
