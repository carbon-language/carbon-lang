// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

namespace N;
alias M = N;
// TODO: It's unclear whether this should be permitted. If not, we should
// produce a better diagnostic.
// CHECK:STDERR: COMPILATION ERROR: explorer/testdata/alias/fail_namespace_alias_member.carbon:[[@LINE+1]]: alias M cannot be used as a name qualifier
fn M.F() {}

fn Main() -> i32 {
  N.F();
  return 0;
}
