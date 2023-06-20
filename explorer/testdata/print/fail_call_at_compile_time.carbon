// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest impl;

interface I {}

// CHECK:STDERR: COMPILATION ERROR: fail_call_at_compile_time.carbon:[[@LINE+1]]: Print called before run time
impl forall [T:! String] Print(T) as I {}

fn Main() -> i32 { return 0; }
