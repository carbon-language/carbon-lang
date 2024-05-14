// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// CHECK:STDOUT: 3 args: `default_args`, `a.carbon`, `b.carbon`

// --- a.carbon
aaa
// CHECK:STDOUT: a.carbon:[[@LINE-1]]: starts with "aaa\n// CHECK:STDOUT: a.carbon:{{\[\[}}@LINE-1]", length 3 lines

// --- b.carbon
bbb
// CHECK:STDOUT: b.carbon:[[@LINE-1]]: starts with "bbb\n// CHECK:STDOUT: b.carbon:{{\[\[}}@LINE-1]", length 2 lines
