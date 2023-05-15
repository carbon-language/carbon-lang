// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface IFace {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/interface/fail_use_assoc_const_in_defn.carbon:[[@LINE+1]]: 'T' is not usable until after it has been completely declared
  let T:! T.IFace;
}

fn Main() -> i32 {
  return 0;
}
