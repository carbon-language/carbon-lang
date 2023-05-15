// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

namespace N;

fn N.F() -> i32 { return 0; }

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/namespace/fail_members_not_in_outer_scope.carbon:[[@LINE+1]]: could not resolve 'F'
  return F();
}
