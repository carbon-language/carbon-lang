// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/tuple/fail_unexpected_tuple.carbon:[[@LINE+1]]: didn't expect a tuple
  var a: (auto,) = 1;
  return 0;
}
