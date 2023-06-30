// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

__mixin Operations {}

// CHECK:STDERR: COMPILATION ERROR: fail_mix_as_alias_target.carbon:[[@LINE+1]]: invalid target for alias declaration
alias OperationsAlias = Operations;

fn Main() -> i32 {
   return 0;
}
