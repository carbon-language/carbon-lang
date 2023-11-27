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

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc14: type = struct_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_redeclaration_introducer.carbon" {
// CHECK:STDOUT:   class_decl @A, ()
// CHECK:STDOUT:   %A: type = class_type @A
// CHECK:STDOUT:   class_decl @A, ()
// CHECK:STDOUT:   class_decl @B, ()
// CHECK:STDOUT:   %B: type = class_type @B
// CHECK:STDOUT:   class_decl @B, ()
// CHECK:STDOUT:   class_decl @C, ()
// CHECK:STDOUT:   %C: type = class_type @C
// CHECK:STDOUT:   class_decl @C, ()
// CHECK:STDOUT:   class_decl @D, ()
// CHECK:STDOUT:   %D: type = class_type @D
// CHECK:STDOUT:   class_decl @D, ()
// CHECK:STDOUT:   class_decl @E, ()
// CHECK:STDOUT:   %E: type = class_type @E
// CHECK:STDOUT:   class_decl @E, ()
// CHECK:STDOUT:   class_decl @F, ()
// CHECK:STDOUT:   %F: type = class_type @F
// CHECK:STDOUT:   class_decl @F, ()
// CHECK:STDOUT:   class_decl @G, ()
// CHECK:STDOUT:   %G: type = class_type @G
// CHECK:STDOUT:   class_decl @G, ()
// CHECK:STDOUT:   class_decl @G, ()
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
