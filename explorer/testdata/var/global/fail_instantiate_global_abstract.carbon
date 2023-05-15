// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

abstract class B {
}

// CHECK:STDERR: COMPILATION ERROR: explorer/testdata/var/global/fail_instantiate_global_abstract.carbon:[[@LINE+1]]: Cannot instantiate abstract class B
var b: B = {};

fn Main() -> i32 {
    return 0;
}
