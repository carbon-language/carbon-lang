// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{carbon} compile --output=- --target=x86_687-unknown-linux-gnu \
// RUN:   %s | %{FileCheck-strict}
// CHECK:STDERR: ERROR: {{.*}}

fn Main() -> i32 { return 0; }
