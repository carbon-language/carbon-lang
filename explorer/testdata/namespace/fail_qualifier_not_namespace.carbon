// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn F() {}

// CHECK:STDERR: COMPILATION ERROR: explorer/testdata/namespace/fail_qualifier_not_namespace.carbon:[[@LINE+1]]: fn F cannot be used as a name qualifier
fn F.G() {}

fn Main() -> i32 {
  return 0;
}
