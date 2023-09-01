// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// CHECK:STDERR: unattached message 3
// CHECK:STDERR: unattached message 4

// --- a.carbon
aaa

// --- b.carbon
bbb

// CHECK:STDOUT: 3 args: `default_args`, `a.carbon`, `b.carbon`
// CHECK:STDOUT: unattached message 1
// CHECK:STDOUT: unattached message 2
