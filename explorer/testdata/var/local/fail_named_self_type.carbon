// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Main() -> i32 {
  // Error: can't use keyword `Self` as the name of a variable.
  // TODO: Current error message is unclear, better would be to say
  // something like: unexpected `Self`, expecting identifier
  // CHECK:STDERR: SYNTAX ERROR: explorer/testdata/var/local/fail_named_self_type.carbon:[[@LINE+1]]: syntax error, unexpected COLON, expecting EQUAL or SEMICOLON
  var Self: i32 = 0;
  return 0;
}
