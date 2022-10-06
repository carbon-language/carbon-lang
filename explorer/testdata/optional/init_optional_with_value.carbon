// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s 2>&1 | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s
// CHECK:Hallo Welt
// CHECK:result: 0

package ExplorerTest api;

fn Main() -> i32{
  var x: Optional(String) = Optional(String).Create( "Hallo Welt" );
  if(x.HasValue()){
    Print(x.Get());
    return 0;
  }
  return -1;
}
