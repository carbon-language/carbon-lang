// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --target=x86_64-unknown-linux-gnu --output=%t
// AUTOUPDATE
// CHECK:STDERR: ERROR: Not all required positional arguments were provided. First missing and required positional argument: 'FILE'

fn Main() -> i32 { return 0; }
