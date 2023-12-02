// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class A;

class X {
  // OK, a different A.
  class A { class B; }
  class A.B {}
}

class A { class B; }

class Y {
  // CHECK:STDERR: fail_redeclaration_scope.carbon:[[@LINE+3]]:9: ERROR: Name `A` not found.
  // CHECK:STDERR:   class A.B {}
  // CHECK:STDERR:         ^
  class A.B {}
}

// CHECK:STDOUT: --- fail_redeclaration_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc11: type = struct_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl.loc7: type = class_decl @A.1, ()
// CHECK:STDOUT:   %A: type = class_type @A.1
// CHECK:STDOUT:   %X.decl: type = class_decl @X, ()
// CHECK:STDOUT:   %X: type = class_type @X
// CHECK:STDOUT:   %A.decl.loc15: type = class_decl @A.1, ()
// CHECK:STDOUT:   %Y.decl: type = class_decl @Y, ()
// CHECK:STDOUT:   %Y: type = class_type @Y
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A.1 {
// CHECK:STDOUT:   %B.decl: type = class_decl @B.2, ()
// CHECK:STDOUT:   %B: type = class_type @B.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %A.decl: type = class_decl @A.2, ()
// CHECK:STDOUT:   %A: type = class_type @A.2
// CHECK:STDOUT:   %B.decl: type = class_decl @B.1, ()
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = %A.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A.2 {
// CHECK:STDOUT:   %B.decl: type = class_decl @B.1, ()
// CHECK:STDOUT:   %B: type = class_type @B.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B.1 {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B.2;
// CHECK:STDOUT:
// CHECK:STDOUT: class @Y {
// CHECK:STDOUT:   %.decl: type = class_decl @.1, ()
// CHECK:STDOUT:   %.loc21: type = class_type @.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
