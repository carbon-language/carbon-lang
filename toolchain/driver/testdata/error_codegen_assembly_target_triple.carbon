// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} dump assembly --target_triple=x86_687-unknown-linux-gnu %s | %{FileCheck-strict}
// CHECK:STDERR: ERROR: Invalid -target_triple:{{.*}}

fn Main() -> i32 { return 0; }
