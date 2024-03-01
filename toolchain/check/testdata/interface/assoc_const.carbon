// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I {
  let T:! type;
  let N:! i32;
}

// CHECK:STDOUT: --- assoc_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type @I, type [template]
// CHECK:STDOUT:   %.3: <associated type in I> = assoc_entity element0, @I.%.loc8_15.1 [template]
// CHECK:STDOUT:   %.4: type = assoc_entity_type @I, i32 [template]
// CHECK:STDOUT:   %.5: <associated i32 in I> = assoc_entity element1, @I.%.loc9_14.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl = interface_decl @I [template = constants.%.1] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %.loc8_15.1: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc8_15.2: <associated type in I> = assoc_entity element0, %.loc8_15.1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc9_14.1: i32 = assoc_const_decl N [template]
// CHECK:STDOUT:   %.loc9_14.2: <associated i32 in I> = assoc_entity element1, %.loc9_14.1 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .T = %.loc8_15.2
// CHECK:STDOUT:   .N = %.loc9_14.2
// CHECK:STDOUT:   witness = (%.loc8_15.1, %.loc9_14.1)
// CHECK:STDOUT: }
// CHECK:STDOUT:
