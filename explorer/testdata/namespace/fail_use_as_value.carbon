// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

namespace N;

fn F[T:! type](x: T) -> T { return x; }

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/namespace/fail_use_as_value.carbon:[[@LINE+1]]: expected `.member_name` after name of namespace N
  return F(N);
}
