// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn Main() -> i32 {
    var i: i32 = 1;
    var x: [i32;] = (i, i, i);
    return x[0] + x[1] + x[2] + x[3];
    // CHECK: RUNTIME ERROR: {{.*}}/explorer/testdata/tuple/fail_implicitly_to_array_access.carbon:[[@LINE-1]]: index 3 out of range in (1, 1, 1)
}
