// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface I {}

fn Main() -> i32 {
  var a: auto = I where .Self == i32;
  // OK: Type of `I where ...` is `Type`.
  a = i32;
  return 0;
}
