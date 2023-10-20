// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=lex --dump-shared-values %s
//
// AUTOUPDATE

var a: f64 = 1.0;

// CHECK:STDOUT: shared_values:
// CHECK:STDOUT:   - integers:
// CHECK:STDOUT:       - 64
// CHECK:STDOUT:   - reals:
// CHECK:STDOUT:       - 10*10^-1
// CHECK:STDOUT:   - strings:
// CHECK:STDOUT:       - "a"
