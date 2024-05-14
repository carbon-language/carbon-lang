// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

// CHECK:STDERR: COMPILATION ERROR: fail_impl_as_symbolic.carbon:[[@LINE+1]]: value of generic binding T is not known
impl forall [T:! ({}, )] {} as T {
}

fn Main() -> i32 {
}
