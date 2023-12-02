// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class A;
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE+6]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: base class A {}
// CHECK:STDERR: ^~~~
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: class A;
// CHECK:STDERR: ^~~~~
base class A {}

class B;
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE+6]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: abstract class B {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: class B;
// CHECK:STDERR: ^~~~~
abstract class B {}

base class C;
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE+6]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: class C {}
// CHECK:STDERR: ^~~~~
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: base class C;
// CHECK:STDERR: ^~~~
class C {}

base class D;
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE+6]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: abstract class D {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: base class D;
// CHECK:STDERR: ^~~~
abstract class D {}

abstract class E;
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE+6]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: class E {}
// CHECK:STDERR: ^~~~~
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: abstract class E;
// CHECK:STDERR: ^~~~~~~~
class E {}

abstract class F;
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE+6]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: base class F {}
// CHECK:STDERR: ^~~~
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: abstract class F;
// CHECK:STDERR: ^~~~~~~~
base class F {}

class G {}
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE+6]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: abstract class G;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: class G {}
// CHECK:STDERR: ^~~~~
abstract class G;
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE+6]]:1: ERROR: Class redeclared with different inheritance kind.
// CHECK:STDERR: base class G;
// CHECK:STDERR: ^~~~
// CHECK:STDERR: fail_redeclaration_introducer.carbon:[[@LINE-11]]:1: Previously declared here.
// CHECK:STDERR: class G {}
// CHECK:STDERR: ^~~~~
base class G;

// CHECK:STDOUT: --- fail_redeclaration_introducer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc14: type = struct_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl.loc7: type = class_decl @A, ()
// CHECK:STDOUT:   %A: type = class_type @A
// CHECK:STDOUT:   %A.decl.loc14: type = class_decl @A, ()
// CHECK:STDOUT:   %B.decl.loc16: type = class_decl @B, ()
// CHECK:STDOUT:   %B: type = class_type @B
// CHECK:STDOUT:   %B.decl.loc23: type = class_decl @B, ()
// CHECK:STDOUT:   %C.decl.loc25: type = class_decl @C, ()
// CHECK:STDOUT:   %C: type = class_type @C
// CHECK:STDOUT:   %C.decl.loc32: type = class_decl @C, ()
// CHECK:STDOUT:   %D.decl.loc34: type = class_decl @D, ()
// CHECK:STDOUT:   %D: type = class_type @D
// CHECK:STDOUT:   %D.decl.loc41: type = class_decl @D, ()
// CHECK:STDOUT:   %E.decl.loc43: type = class_decl @E, ()
// CHECK:STDOUT:   %E: type = class_type @E
// CHECK:STDOUT:   %E.decl.loc50: type = class_decl @E, ()
// CHECK:STDOUT:   %F.decl.loc52: type = class_decl @F, ()
// CHECK:STDOUT:   %F: type = class_type @F
// CHECK:STDOUT:   %F.decl.loc59: type = class_decl @F, ()
// CHECK:STDOUT:   %G.decl.loc61: type = class_decl @G, ()
// CHECK:STDOUT:   %G: type = class_type @G
// CHECK:STDOUT:   %G.decl.loc68: type = class_decl @G, ()
// CHECK:STDOUT:   %G.decl.loc75: type = class_decl @G, ()
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
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @F {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @G {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
