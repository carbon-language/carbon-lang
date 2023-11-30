// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class A;
base class B;
abstract class C;

class A {}
base class B {}
abstract class C {}

// CHECK:STDOUT: --- redeclaration_introducer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc11: type = struct_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.A = %A.decl.loc7, .B = %B.decl.loc8, .C = %C.decl.loc9}
// CHECK:STDOUT:   %A.decl.loc7 = class_decl @A, ()
// CHECK:STDOUT:   %A: type = class_type @A
// CHECK:STDOUT:   %B.decl.loc8 = class_decl @B, ()
// CHECK:STDOUT:   %B: type = class_type @B
// CHECK:STDOUT:   %C.decl.loc9 = class_decl @C, ()
// CHECK:STDOUT:   %C: type = class_type @C
// CHECK:STDOUT:   %A.decl.loc11 = class_decl @A, ()
// CHECK:STDOUT:   %B.decl.loc12 = class_decl @B, ()
// CHECK:STDOUT:   %C.decl.loc13 = class_decl @C, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
