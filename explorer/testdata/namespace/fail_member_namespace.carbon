// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package Foo api;

class C {
  // CHECK:STDERR: SYNTAX ERROR: fail_member_namespace.carbon:[[@LINE+1]]: syntax error, unexpected NAMESPACE, expecting CLASS
  namespace N;
}

fn Main() -> i32 { return 0; }
