// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s | %{FileCheck-strict} %s
// RUN: %{explorer-trace} %s | %{FileCheck-allow-unmatched} %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

class Holder(N:! i32) {
  fn Get() -> i32 {
    return N;
  }
}

class IntLike {
  impl as ImplicitAs(i32) {
    fn Convert[me: Self]() -> i32 { return 4; }
  }
  fn Make() -> IntLike { return {}; }
}

fn Main() -> i32 {
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/generic_class/parameter_type_conversion.carbon:[[@LINE+1]]: Function would be called at compile-time
  return Holder(IntLike.Make()).Get();
}
