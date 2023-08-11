// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --target=x86_64-unknown-linux-gnu --asm-output %s
// AUTOUPDATE
// CHECK:STDERR: ERROR: Could not open file: No such file or directory

fn Main() -> i32 { return 0; }
