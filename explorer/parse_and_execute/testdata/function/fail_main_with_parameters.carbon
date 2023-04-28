// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main(i: i32) -> i32 {
    return 0;
// CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/function/fail_main_with_parameters.carbon:[[@LINE+1]]: `Main` must not take any parameters
}
