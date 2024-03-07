// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I {
  // CHECK:STDERR: fail_assoc_const_not_binding.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `tuple pattern in let`.
  // CHECK:STDERR:   let (T:! type, U:! type);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  let (T:! type, U:! type);
}

// CHECK:STDOUT: --- fail_assoc_const_not_binding.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = <unexpected instref inst+3>
// CHECK:STDOUT:   witness = invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
