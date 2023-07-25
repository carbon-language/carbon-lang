// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

class C {
  // CHECK:STDERR: COMPILATION ERROR: fail_circular_inheritance.carbon:[[@LINE+4]]: could not find `class C {
  // CHECK:STDERR: extend base: C;
  // CHECK:STDERR: }
  // CHECK:STDERR: `
  extend base: C;
}

fn Main() -> i32 { return 0; }
