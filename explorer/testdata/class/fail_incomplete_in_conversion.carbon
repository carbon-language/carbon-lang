// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class C {
  // CHECK:STDERR: COMPILATION ERROR: fail_incomplete_in_conversion.carbon:[[@LINE+2]]: type error in `as`: `{}` is not explicitly convertible to `class C`:
  // CHECK:STDERR: incomplete type `class C` used in implicit conversion
  var x: ({} as C);
}

fn Main() -> i32 { return 0; }
