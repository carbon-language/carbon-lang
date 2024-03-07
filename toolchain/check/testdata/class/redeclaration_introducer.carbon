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
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl.loc7
// CHECK:STDOUT:     .B = %B.decl.loc8
// CHECK:STDOUT:     .C = %C.decl.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl.loc7: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl.loc8: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %C.decl.loc9: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %A.decl.loc11: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl.loc12: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %C.decl.loc13: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
