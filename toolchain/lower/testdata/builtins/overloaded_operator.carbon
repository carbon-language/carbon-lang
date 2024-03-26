// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- prelude.carbon

package Core api;

interface Add {
  fn Op[self: Self](other: Self) -> Self;
}

// --- user.carbon

import Core;

// TODO: Move this into prelude.carbon when possible. For now, impl lookup only
// looks in the current file.
impl i32 as Core.Add {
  // CHECK:STDERR: user.carbon:[[@LINE+3]]:3: ERROR: Invalid signature for builtin function "int.add".
  // CHECK:STDERR:   fn Op[self: i32](other: i32) -> i32 = "int.add";
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  fn Op[self: i32](other: i32) -> i32 = "int.add";
}

fn AddThreeIntegers(a: i32, b: i32, c: i32) -> i32 {
  return a + b + c;
}
