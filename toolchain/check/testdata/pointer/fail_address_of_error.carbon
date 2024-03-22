// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Test() {
  // CHECK:STDERR: fail_address_of_error.carbon:[[@LINE+4]]:4: ERROR: Name `undeclared` not found.
  // CHECK:STDERR:   &undeclared;
  // CHECK:STDERR:    ^~~~~~~~~~
  // CHECK:STDERR:
  &undeclared;
  // CHECK:STDERR: fail_address_of_error.carbon:[[@LINE+7]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(&undeclared);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_address_of_error.carbon:[[@LINE+3]]:6: ERROR: Name `undeclared` not found.
  // CHECK:STDERR:   &(&undeclared);
  // CHECK:STDERR:      ^~~~~~~~~~
  &(&undeclared);
}

// CHECK:STDOUT: --- fail_address_of_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Test = %Test
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test: <function> = fn_decl @Test [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %undeclared.ref.loc12: <error> = name_ref undeclared, <error> [template = <error>]
// CHECK:STDOUT:   %.loc12: <error> = addr_of %undeclared.ref.loc12 [template = <error>]
// CHECK:STDOUT:   %undeclared.ref.loc20: <error> = name_ref undeclared, <error> [template = <error>]
// CHECK:STDOUT:   %.loc20_5: <error> = addr_of %undeclared.ref.loc20 [template = <error>]
// CHECK:STDOUT:   %.loc20_3: <error> = addr_of <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
