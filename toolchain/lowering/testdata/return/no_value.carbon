// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{carbon-run-lowering}
// CHECK:STDOUT: ; ModuleID = '{{.*}}/toolchain/lowering/testdata/return/no_value.carbon'
// CHECK:STDOUT: source_filename = "{{.*}}/toolchain/lowering/testdata/return/no_value.carbon"
// CHECK-EMPTY:
// CHECK:STDOUT: define void @Main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

fn Main() {
  return;
}
