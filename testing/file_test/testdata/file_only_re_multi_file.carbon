// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// CHECK:STDOUT: 3 args: `default_args`, `a.carbon`, `b.carbon`
// CHECK:STDOUT: unattached message 1

// --- a.carbon
// CHECK:STDOUT: file: a.carbon
// CHECK:STDOUT: unattached message 2
aaa
// CHECK:STDOUT: line: [[@LINE-1]]: attached message 3
// CHECK:STDOUT: unattached message 4

// CHECK:STDOUT: line: [[@LINE+1]]: late message 5
// CHECK:STDOUT: unattached message 6
// --- b.carbon
// CHECK:STDOUT: file: b.carbon
bbb
// CHECK:STDOUT: line: [[@LINE-1]]: attached message 7
// CHECK:STDOUT: unattached message 8

// CHECK:STDOUT: line: [[@LINE+1]]: late message 9
// CHECK:STDOUT: unattached message 10
