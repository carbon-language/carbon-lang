// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: dump objcode --target_triple=x86_64-unknown-linux-gnu --output_file=%t %s
// AUTOUPDATE
// CHECK:STDOUT: Success: Object file is generated!

fn Main() -> i32 { return 0; }
