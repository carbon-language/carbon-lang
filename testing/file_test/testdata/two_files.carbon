// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// CHECK:STDOUT: 3 args: `default_args`, `a.carbon`, `b.carbon`

// --- a.carbon
// CHECK:STDOUT: a.carbon:[[@LINE+1]]: 1
aaa

// --- b.carbon
// CHECK:STDOUT: b.carbon:[[@LINE+1]]: 2
bbb
