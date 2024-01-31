// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Empty {
}

// CHECK:STDERR: empty.carbon:[[@LINE+3]]:13: ERROR: Semantics TODO: `Unimplemented use of interface`.
// CHECK:STDERR: impl i32 as Empty {
// CHECK:STDERR:             ^~~~~
impl i32 as Empty {
}

// CHECK:STDOUT: --- empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
