// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface Interface {}
impl {} as Interface {}
impl {.x: i32} as Interface {}

fn Main() -> i32 {
  return 0;
}
