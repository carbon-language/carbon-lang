// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE

// --- a.carbon
a

// --- b.carbon
// CHECK:STDERR: In b.carbon:
// CHECK:STDERR: an error
b

// --- c.carbon
// CHECK:STDERR: In c.carbon:
c1
c2
// CHECK:STDERR: c.carbon:[[@LINE+1]]: an error
c3
c4
c5

// CHECK:STDOUT: 4 args: `default_args`, `a.carbon`, `b.carbon`, `c.carbon`
