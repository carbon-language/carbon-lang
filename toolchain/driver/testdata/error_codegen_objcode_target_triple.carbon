// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} dump objcode --target_triple=x86_684-unknown-linux-gnu --output_file=output.o %s | %{FileCheck-strict}
// CHECK:STDERR: ERROR: Invalid -target_triple:{{.*}}

fn Main() -> i32 { return 0; }
