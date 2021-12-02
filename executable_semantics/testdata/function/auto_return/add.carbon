// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{executable_semantics} --prelude=%{prelude} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{executable_semantics} --prelude=%{prelude} --trace %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{executable_semantics} %s
// CHECK: result: 0

package ExecutableSemanticsTest api;

fn add(x: i32, y: i32) -> auto { return x + y; }

fn Main() -> i32 {
  return add(1, 2) - 3;
}
