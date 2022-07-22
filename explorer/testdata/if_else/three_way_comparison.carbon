// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: less: -1
// CHECK: equal: 0
// CHECK: greater: 1
// CHECK: result: 0

package ExplorerTest api;

fn Compare(x: i32, y: i32) -> i32 {
    if (x < y) {
        return -1;
    }
    if (x == y) {
        return 0;
    }
    if (x > y) {
        return 1;
    }

    return -2;
}

fn Main() -> i32 {
    Print("less: {0}", Compare(1, 2));
    Print("equal: {0}", Compare(1, 1));
    Print("greater: {0}", Compare(2, 1));
    return 0;
}
