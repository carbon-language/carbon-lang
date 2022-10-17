// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: i32
// CHECK:STDOUT: String
// CHECK:STDOUT: result: 0

package Testcase api;

interface HasName {
    let Name:! String;
}

external impl i32 as HasName where .Name == "i32" {}
external impl String as HasName where .Name == "String" {}

fn Main() -> i32 {
    Print(i32.(HasName.Name));
    Print(String.(HasName.Name));
    return 0;
}
