// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --target=x86_687-unknown-linux-gnu --output=- %s
// No autoupdate because the message comes from LLVM.
// NOAUTOUPDATE
// CHECK:STDERR: ERROR: Invalid target: {{.*}}x86_687{{.*}}

fn Main() -> i32 { return 0; }
