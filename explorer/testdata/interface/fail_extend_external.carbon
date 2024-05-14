// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class IntHolder {
  var n: i32;
}

// CHECK:STDERR: SYNTAX ERROR: fail_extend_external.carbon:[[@LINE+1]]: syntax error, unexpected EXTEND, expecting END_OF_FILE
extend impl IntHolder as ImplicitAs(i32) {
  fn Convert[self: Self]() -> i32 { return self.n; }
}

fn Main() -> i32 {
  return ({.n = 5} as IntHolder).(ImplicitAs(i32).Convert)();
}
