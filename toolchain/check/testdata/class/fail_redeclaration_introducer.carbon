// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_add_base.carbon

library "fail_add_base" api;

class A;
// CHECK:STDERR: fail_add_base.carbon:[[@LINE+7]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: base class A {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_add_base.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: class A;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
base class A {}

// --- fail_add_abstract.carbon

library "fail_add_abstract" api;

class B;
// CHECK:STDERR: fail_add_abstract.carbon:[[@LINE+7]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: abstract class B {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_add_abstract.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: class B;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract class B {}

// --- fail_remove_base.carbon

library "fail_remove_base" api;

base class C;
// CHECK:STDERR: fail_remove_base.carbon:[[@LINE+7]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_remove_base.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: base class C;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
class C {}

// --- fail_base_to_abstract.carbon

library "fail_base_to_abstract" api;

base class D;
// CHECK:STDERR: fail_base_to_abstract.carbon:[[@LINE+7]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: abstract class D {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_base_to_abstract.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: base class D;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
abstract class D {}

// --- fail_remove_abstract.carbon

library "fail_remove_abstract" api;

abstract class E;
// CHECK:STDERR: fail_remove_abstract.carbon:[[@LINE+7]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: class E {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR: fail_remove_abstract.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: abstract class E;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
class E {}

// --- fail_abstract_to_base.carbon

library "fail_abstract_to_base" api;

abstract class F;
// CHECK:STDERR: fail_abstract_to_base.carbon:[[@LINE+6]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: base class F {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_abstract_to_base.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: abstract class F;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
base class F {}

// CHECK:STDOUT: --- fail_add_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl.loc4: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %A.decl.loc12: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_add_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .B = %B.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %B.decl.loc4: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %B.decl.loc12: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_remove_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl.loc4: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %C.decl.loc12: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_base_to_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .D = %D.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %D.decl.loc4: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %D.decl.loc12: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_remove_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %E: type = class_type @E [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .E = %E.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %E.decl.loc4: type = class_decl @E [template = constants.%E] {}
// CHECK:STDOUT:   %E.decl.loc12: type = class_decl @E [template = constants.%E] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_abstract_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F: type = class_type @F [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl.loc4: type = class_decl @F [template = constants.%F] {}
// CHECK:STDOUT:   %F.decl.loc11: type = class_decl @F [template = constants.%F] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @F {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%F
// CHECK:STDOUT: }
// CHECK:STDOUT:
