// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class A;
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE+7]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: base class A {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: class A;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
base class A {}

class B;
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE+7]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: abstract class B {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: class B;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract class B {}

base class C;
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE+7]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: base class C;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
class C {}

base class D;
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE+7]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: abstract class D {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: base class D;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
abstract class D {}

abstract class E;
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE+7]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: class E {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: abstract class E;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
class E {}

abstract class F;
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE+7]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: base class F {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: abstract class F;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
base class F {}

class G {}
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE+7]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: abstract class G;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: class G {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
abstract class G;
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE+6]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: base class G;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE-12]]:1: Previously declared here.
// CHECK:STDERR: class G {}
// CHECK:STDERR: ^~~~~~~~~
base class G;

// CHECK:STDOUT: --- fail_redeclaration_introducer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %E: type = class_type @E [template]
// CHECK:STDOUT:   %F: type = class_type @F [template]
// CHECK:STDOUT:   %G: type = class_type @G [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl.loc7
// CHECK:STDOUT:     .B = %B.decl.loc17
// CHECK:STDOUT:     .C = %C.decl.loc27
// CHECK:STDOUT:     .D = %D.decl.loc37
// CHECK:STDOUT:     .E = %E.decl.loc47
// CHECK:STDOUT:     .F = %F.decl.loc57
// CHECK:STDOUT:     .G = %G.decl.loc67
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl.loc7: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %A.decl.loc15: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl.loc17: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %B.decl.loc25: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %C.decl.loc27: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.decl.loc35: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %D.decl.loc37: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %D.decl.loc45: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %E.decl.loc47: type = class_decl @E [template = constants.%E] {}
// CHECK:STDOUT:   %E.decl.loc55: type = class_decl @E [template = constants.%E] {}
// CHECK:STDOUT:   %F.decl.loc57: type = class_decl @F [template = constants.%F] {}
// CHECK:STDOUT:   %F.decl.loc65: type = class_decl @F [template = constants.%F] {}
// CHECK:STDOUT:   %G.decl.loc67: type = class_decl @G [template = constants.%G] {}
// CHECK:STDOUT:   %G.decl.loc75: type = class_decl @G [template = constants.%G] {}
// CHECK:STDOUT:   %G.decl.loc82: type = class_decl @G [template = constants.%G] {}
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
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @F {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @G {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%G
// CHECK:STDOUT: }
// CHECK:STDOUT:
