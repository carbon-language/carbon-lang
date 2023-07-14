// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface X {
}
// CHECK:STDERR: COMPILATION ERROR: fail_impl_symbolic_as.carbon:[[@LINE+1]]: value of generic binding Y is not known
impl forall [Y:! (X, )] Y as true {
}
fn Main() -> i32 {
}
