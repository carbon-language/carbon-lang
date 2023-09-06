// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// CHECK:STDERR: unattached message 1
// CHECK:STDOUT: 3 args: `default_args`, `a.carbon`, `b.carbon`
// CHECK:STDOUT: unattached message 1

// --- a.carbon
// CHECK:STDERR: a.carbon:[[@LINE+1]]: message 2
aaa
// CHECK:STDOUT: a.carbon:[[@LINE-1]]: message 2

// --- b.carbon
// CHECK:STDERR: b.carbon:[[@LINE+4]]: message 3
// CHECK:STDERR: a.carbon:2: message 4
// CHECK:STDERR: b.carbon:[[@LINE+2]]: message 5
// CHECK:STDERR: unattached message 6
bbb
// CHECK:STDOUT: b.carbon:[[@LINE-1]]: message 3
// CHECK:STDOUT: a.carbon:2: message 4
// CHECK:STDOUT: b.carbon:[[@LINE-3]]: message 5
// CHECK:STDOUT: unattached message 6
bbbbbb
