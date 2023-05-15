// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 0

package ExplorerTest api;

namespace N;
namespace N.M;
fn N.F() {}
fn N.M.FM() {}

alias A = N;
alias AM = N.M;
alias N.A = N.M;
alias N.A2 = N;

fn Main() -> i32 {
  A.F();
  AM.FM();
  N.A.FM();
  A.A2.A2.A2.A2.A2.A2.A2.F();
  return 0;
}
