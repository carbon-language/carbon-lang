// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I[T:! type](N:! T);

// CHECK:STDOUT: --- generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %N: T = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {
// CHECK:STDOUT:     %T.loc7_13.1: type = param T
// CHECK:STDOUT:     %T.loc7_13.2: type = bind_symbolic_name T 0, %T.loc7_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %N.loc7_23.1: T = param N
// CHECK:STDOUT:     %N.loc7_23.2: T = bind_symbolic_name N 1, %N.loc7_23.1 [symbolic = constants.%N]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I;
// CHECK:STDOUT:
