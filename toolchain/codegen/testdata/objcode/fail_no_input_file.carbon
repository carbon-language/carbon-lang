// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: dump objcode --output_file=%t --target_triple=x86_64-unknown-linux-gnu
// TODO: Support autoupdate with ARGS.
// NOAUTOUPDATE
// CHECK:STDERR: ERROR: No input file specified.

fn Main() -> i32 { return 0; }
