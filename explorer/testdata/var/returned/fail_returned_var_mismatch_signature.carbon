// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

fn Foo() {
  returned var x: () = ();
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/var/returned/fail_returned_var_mismatch_signature.carbon:[[@LINE+1]]: return var; should not provide a return value, to match the function's signature.
  return var;
}

fn Main() -> i32 {
  Foo();
  return 0;
}
