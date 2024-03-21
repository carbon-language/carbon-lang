// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I {
  // CHECK:STDERR: fail_todo_assoc_const_default.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `interface modifier`.
  // CHECK:STDERR:   default let T:! type = (i32, i32);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default let T:! type = (i32, i32);
  // CHECK:STDERR: fail_todo_assoc_const_default.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `interface modifier`.
  // CHECK:STDERR:   default let N:! i32 = 42;
  // CHECK:STDERR:   ^~~~~~~
  default let N:! i32 = 42;
}

// CHECK:STDOUT: --- fail_todo_assoc_const_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type @I, type [template]
// CHECK:STDOUT:   %.5: <associated type in I> = assoc_entity element0, @I.%T [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 42 [template]
// CHECK:STDOUT:   %.7: type = assoc_entity_type @I, i32 [template]
// CHECK:STDOUT:   %.8: <associated i32 in I> = assoc_entity element1, @I.%N [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %.loc12_35.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc12_35.2: type = converted %.loc12_35.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc12_36: <associated type in I> = assoc_entity element0, %T [template = constants.%.5]
// CHECK:STDOUT:   %.loc16_25: i32 = int_literal 42 [template = constants.%.6]
// CHECK:STDOUT:   %N: i32 = assoc_const_decl N [template]
// CHECK:STDOUT:   %.loc16_27: <associated i32 in I> = assoc_entity element1, %N [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %.loc12_36
// CHECK:STDOUT:   .N = %.loc16_27
// CHECK:STDOUT:   witness = (%T, %N)
// CHECK:STDOUT: }
// CHECK:STDOUT:
