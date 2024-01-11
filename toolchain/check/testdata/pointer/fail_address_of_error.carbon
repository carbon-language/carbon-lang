// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Test() {
  // CHECK:STDERR: fail_address_of_error.carbon:[[@LINE+3]]:4: ERROR: Name `undeclared` not found.
  // CHECK:STDERR:   &undeclared;
  // CHECK:STDERR:    ^~~~~~~~~~
  &undeclared;
  // CHECK:STDERR: fail_address_of_error.carbon:[[@LINE+6]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(&undeclared);
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_address_of_error.carbon:[[@LINE+3]]:6: ERROR: Name `undeclared` not found.
  // CHECK:STDERR:   &(&undeclared);
  // CHECK:STDERR:      ^~~~~~~~~~
  &(&undeclared);
}

// CHECK:STDOUT: --- fail_address_of_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc11: type = ptr_type <error>, const
// CHECK:STDOUT:   %.loc18: type = ptr_type <error>*, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.Test = %Test}
// CHECK:STDOUT:   %Test: <function> = fn_decl @Test, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %undeclared.ref.loc11: <error> = name_ref undeclared, <error>
// CHECK:STDOUT:   %.loc11: <error>* = addr_of %undeclared.ref.loc11
// CHECK:STDOUT:   %undeclared.ref.loc18: <error> = name_ref undeclared, <error>
// CHECK:STDOUT:   %.loc18_5: <error>* = addr_of %undeclared.ref.loc18
// CHECK:STDOUT:   %.loc18_3: <error>** = addr_of %.loc18_5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
