// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface Vector(Scalar:! type) {
}

// CHECK:STDERR: COMPILATION ERROR: fail_impl_as_parameterized.carbon:[[@LINE+1]]: 'Vector' must be given an argument list
impl i32 as Vector {}

fn Main() -> i32 {
}
