// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- valid.carbon

library "valid" api;

class Generic(T:! type);

class Generic(T:! type) {
}

// --- fail_invalid.carbon

library "invalid" api;

class A;
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+7]]:1: ERROR: Redeclaration differs because of parameter list.
// CHECK:STDERR: class A(T:! type) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_invalid.carbon:[[@LINE-4]]:1: Previously declared without parameter list.
// CHECK:STDERR: class A;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
class A(T:! type) {}

class B(N:! i32);
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+7]]:1: ERROR: Redeclaration differs because of implicit parameter list.
// CHECK:STDERR: class B[T:! type](N:! T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_invalid.carbon:[[@LINE-4]]:1: Previously declared without implicit parameter list.
// CHECK:STDERR: class B(N:! i32);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
class B[T:! type](N:! T) {}

class C(T:! type);
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+7]]:1: ERROR: Redeclaration differs because of parameter count of 2.
// CHECK:STDERR: class C(T:! type, U:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_invalid.carbon:[[@LINE-4]]:1: Previously declared with parameter count of 1.
// CHECK:STDERR: class C(T:! type);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
class C(T:! type, U:! i32) {}

class D(T:! type);
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+7]]:9: ERROR: Redeclaration differs at parameter 1.
// CHECK:STDERR: class D(T:! i32) {}
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_invalid.carbon:[[@LINE-4]]:9: Previous declaration's corresponding parameter here.
// CHECK:STDERR: class D(T:! type);
// CHECK:STDERR:         ^
// CHECK:STDERR:
class D(T:! i32) {}

class E(T:! type);
// CHECK:STDERR: fail_invalid.carbon:[[@LINE+6]]:9: ERROR: Redeclaration differs at parameter 1.
// CHECK:STDERR: class E(U:! type) {}
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_invalid.carbon:[[@LINE-4]]:9: Previous declaration's corresponding parameter here.
// CHECK:STDERR: class E(T:! type);
// CHECK:STDERR:         ^
class E(U:! type) {}

// CHECK:STDOUT: --- valid.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Generic: type = class_type @Generic [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Generic = %Generic.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Generic.decl.loc4: type = class_decl @Generic [template = constants.%Generic] {
// CHECK:STDOUT:     %T.loc4_15.1: type = param T
// CHECK:STDOUT:     %T.loc4_15.2: type = bind_symbolic_name T 0, %T.loc4_15.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl.loc6: type = class_decl @Generic [template = constants.%Generic] {
// CHECK:STDOUT:     %T.loc6_15.1: type = param T
// CHECK:STDOUT:     %T.loc6_15.2: type = bind_symbolic_name T 0, %T.loc6_15.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Generic {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Generic
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_invalid.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.1: type = class_type @.1 [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %N.1: i32 = bind_symbolic_name N 0 [symbolic]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %N.2: T = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %.3: type = class_type @.2 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %U.1: i32 = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %.4: type = class_type @.3 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %T.2: i32 = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.5: type = class_type @.4 [template]
// CHECK:STDOUT:   %E: type = class_type @E [template]
// CHECK:STDOUT:   %U.2: type = bind_symbolic_name U 0 [symbolic]
// CHECK:STDOUT:   %.6: type = class_type @.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %.decl.loc12: type = class_decl @.1 [template = constants.%.1] {
// CHECK:STDOUT:     %T.loc12_9.1: type = param T
// CHECK:STDOUT:     %T.loc12_9.2: type = bind_symbolic_name T 0, %T.loc12_9.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %N.loc14_9.1: i32 = param N
// CHECK:STDOUT:     %N.loc14_9.2: i32 = bind_symbolic_name N 0, %N.loc14_9.1 [symbolic = constants.%N.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc22: type = class_decl @.2 [template = constants.%.3] {
// CHECK:STDOUT:     %T.loc22_9.1: type = param T
// CHECK:STDOUT:     %T.loc22_9.2: type = bind_symbolic_name T 0, %T.loc22_9.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc22_9.2 [symbolic = constants.%T.1]
// CHECK:STDOUT:     %N.loc22_19.1: T = param N
// CHECK:STDOUT:     %N.loc22_19.2: T = bind_symbolic_name N 1, %N.loc22_19.1 [symbolic = constants.%N.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %T.loc24_9.1: type = param T
// CHECK:STDOUT:     %T.loc24_9.2: type = bind_symbolic_name T 0, %T.loc24_9.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc32: type = class_decl @.3 [template = constants.%.4] {
// CHECK:STDOUT:     %T.loc32_9.1: type = param T
// CHECK:STDOUT:     %T.loc32_9.2: type = bind_symbolic_name T 0, %T.loc32_9.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:     %U.loc32_19.1: i32 = param U
// CHECK:STDOUT:     %U.loc32_19.2: i32 = bind_symbolic_name U 1, %U.loc32_19.1 [symbolic = constants.%U.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {
// CHECK:STDOUT:     %T.loc34_9.1: type = param T
// CHECK:STDOUT:     %T.loc34_9.2: type = bind_symbolic_name T 0, %T.loc34_9.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc42: type = class_decl @.4 [template = constants.%.5] {
// CHECK:STDOUT:     %T.loc42_9.1: i32 = param T
// CHECK:STDOUT:     %T.loc42_9.2: i32 = bind_symbolic_name T 0, %T.loc42_9.1 [symbolic = constants.%T.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl: type = class_decl @E [template = constants.%E] {
// CHECK:STDOUT:     %T.loc44_9.1: type = param T
// CHECK:STDOUT:     %T.loc44_9.2: type = bind_symbolic_name T 0, %T.loc44_9.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc51: type = class_decl @.5 [template = constants.%.6] {
// CHECK:STDOUT:     %U.loc51_9.1: type = param U
// CHECK:STDOUT:     %U.loc51_9.2: type = bind_symbolic_name U 0, %U.loc51_9.1 [symbolic = constants.%U.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A;
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B;
// CHECK:STDOUT:
// CHECK:STDOUT: class @.2 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: class @.3 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D;
// CHECK:STDOUT:
// CHECK:STDOUT: class @.4 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E;
// CHECK:STDOUT:
// CHECK:STDOUT: class @.5 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
