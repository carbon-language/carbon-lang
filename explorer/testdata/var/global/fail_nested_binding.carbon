// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

// If support for this is added, it'll be necessary to verify that either the
// implementation is shared with locals or that tests are copied over for edge
// cases.
// CHECK:STDERR: SYNTAX ERROR: explorer/testdata/var/global/fail_nested_binding.carbon:[[@LINE+1]]: syntax error, unexpected LEFT_PARENTHESIS, expecting identifier
var (x: i32,);

fn Main() -> i32 {
  return 0;
}
