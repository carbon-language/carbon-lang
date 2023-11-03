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

// CHECK:STDOUT: file "fail_redeclaration_scope.carbon" {
// CHECK:STDOUT:   class_declaration @A.1, ()
// CHECK:STDOUT:   %A: type = class_type @A.1
// CHECK:STDOUT:   class_declaration @X, ()
// CHECK:STDOUT:   %X: type = class_type @X
// CHECK:STDOUT:   class_declaration @A.1, ()
// CHECK:STDOUT:   class_declaration @Y, ()
// CHECK:STDOUT:   %Y: type = class_type @Y
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A.1 {
// CHECK:STDOUT:   class_declaration @B.2, ()
// CHECK:STDOUT:   %B: type = class_type @B.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = <unexpected instref 20>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   class_declaration @A.2, ()
// CHECK:STDOUT:   %A: type = class_type @A.2
// CHECK:STDOUT:   %.loc11: type = struct_type {}
// CHECK:STDOUT:   class_declaration @B.1, ()
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .A = <unexpected instref 13>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A.2 {
// CHECK:STDOUT:   class_declaration @B.1, ()
// CHECK:STDOUT:   %B: type = class_type @B.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = <unexpected instref 15>
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
// CHECK:STDOUT:   class_declaration @.1, ()
// CHECK:STDOUT:   %.loc21: type = class_type @.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
