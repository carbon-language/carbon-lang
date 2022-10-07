// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:result: 1

package ExplorerTest api;

fn Main() -> i32 {
  match (0) {
    case x: i32 => {
      // TODO: If we disallow declaring shadowed names, we should reject this.
      var x: i32 = 1;
      // TODO: If we disallow using shadowed names, we should reject this.
      return x;
    }
  }
  return 0;
}
