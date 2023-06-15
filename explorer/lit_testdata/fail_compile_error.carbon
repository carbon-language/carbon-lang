// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// NOAUTOUPDATE
// RUN: %{not} %{explorer} | %{FileCheck-strict}
// RUN: %{not} %{explorer} --parser_debug --trace_file=- | %{FileCheck-allow-unmatched}

package ExplorerTest api;

// Just needs to produce an error to validate basic structure.
fn Main() -> i32 {
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/lit_testdata/fail_compile_error.carbon:[[@LINE+1]]: {{.*}}
}
