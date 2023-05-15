// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

namespace Foo;
alias A = Foo;

fn Main() -> i32 {
    // CHECK:STDERR: COMPILATION ERROR: explorer/testdata/alias/fail_wrong_namepsace_alias_use.carbon:[[@LINE+1]]: expected `.member_name` after name of namespace Foo
    return A;
}
