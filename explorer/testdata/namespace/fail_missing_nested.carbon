// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

namespace A;
namespace A.B;
namespace A.B.C;
fn A.B.C.
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/namespace/fail_missing_nested.carbon:[[@LINE+1]]: name 'D' has not been declared in namespace A.B.C
  D
  // Don't combine with the previous line. This test is verifying that the
  // diagnostic points at the right token.
  .E.F() {}

fn Main() -> i32 {
  return 0;
}
