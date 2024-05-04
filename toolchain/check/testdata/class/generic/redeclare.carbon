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

// --- fail_mismatch_param_list.carbon

library "mismatch_param_list" api;

class A;
// CHECK:STDERR: fail_mismatch_param_list.carbon:[[@LINE+7]]:1: ERROR: Redeclaration differs because of parameter list.
// CHECK:STDERR: class A(T:! type) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_mismatch_param_list.carbon:[[@LINE-4]]:1: Previously declared without parameter list.
// CHECK:STDERR: class A;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
class A(T:! type) {}

// --- fail_mismatch_implicit_param_list.carbon

library "implicit_mismatch_param_list" api;

class B(N:! i32);
// CHECK:STDERR: fail_mismatch_implicit_param_list.carbon:[[@LINE+7]]:1: ERROR: Redeclaration differs because of implicit parameter list.
// CHECK:STDERR: class B[T:! type](N:! T) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_mismatch_implicit_param_list.carbon:[[@LINE-4]]:1: Previously declared without implicit parameter list.
// CHECK:STDERR: class B(N:! i32);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
class B[T:! type](N:! T) {}

// --- fail_mismatch_param_count.carbon

library "mismatch_param_count" api;

class C(T:! type);
// CHECK:STDERR: fail_mismatch_param_count.carbon:[[@LINE+7]]:1: ERROR: Redeclaration differs because of parameter count of 2.
// CHECK:STDERR: class C(T:! type, U:! i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_mismatch_param_count.carbon:[[@LINE-4]]:1: Previously declared with parameter count of 1.
// CHECK:STDERR: class C(T:! type);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
class C(T:! type, U:! i32) {}

// --- fail_mismatch_param_type.carbon

library "mismatch_param_type" api;

class D(T:! type);
// CHECK:STDERR: fail_mismatch_param_type.carbon:[[@LINE+7]]:9: ERROR: Redeclaration differs at parameter 1.
// CHECK:STDERR: class D(T:! i32) {}
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_mismatch_param_type.carbon:[[@LINE-4]]:9: Previous declaration's corresponding parameter here.
// CHECK:STDERR: class D(T:! type);
// CHECK:STDERR:         ^
// CHECK:STDERR:
class D(T:! i32) {}

// --- fail_mismatch_param_name.carbon

library "mismatch_param_name" api;

class E(T:! type);
// CHECK:STDERR: fail_mismatch_param_name.carbon:[[@LINE+6]]:9: ERROR: Redeclaration differs at parameter 1.
// CHECK:STDERR: class E(U:! type) {}
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_mismatch_param_name.carbon:[[@LINE-4]]:9: Previous declaration's corresponding parameter here.
// CHECK:STDERR: class E(T:! type);
// CHECK:STDERR:         ^
class E(U:! type) {}

// CHECK:STDOUT: --- valid.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Generic.1: type = generic_class_type @Generic [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Generic = struct_value () [template]
// CHECK:STDOUT:   %Generic.2: type = class_type @Generic [template]
// CHECK:STDOUT:   %struct.2: type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Generic = %Generic.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Generic.decl.loc4: Generic = class_decl @Generic [template = constants.%struct.1] {
// CHECK:STDOUT:     %T.loc4_15.1: type = param T
// CHECK:STDOUT:     %T.loc4_15.2: type = bind_symbolic_name T 0, %T.loc4_15.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl.loc6: type = class_decl @Generic [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.loc6_15.1: type = param T
// CHECK:STDOUT:     %T.loc6_15.2: type = bind_symbolic_name T 0, %T.loc6_15.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Generic {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Generic.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_list.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.1: type = generic_class_type @.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %.3: type = class_type @.1 [template]
// CHECK:STDOUT:   %.4: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %.decl: <invalid> = class_decl @.1 [template = constants.%struct] {
// CHECK:STDOUT:     %T.loc12_9.1: type = param T
// CHECK:STDOUT:     %T.loc12_9.2: type = bind_symbolic_name T 0, %T.loc12_9.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A;
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_implicit_param_list.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %N.1: i32 = bind_symbolic_name N 0 [symbolic]
// CHECK:STDOUT:   %B.1: type = generic_class_type @B [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: B = struct_value () [template]
// CHECK:STDOUT:   %B.2: type = class_type @B [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %N.2: T = bind_symbolic_name N 1 [symbolic]
// CHECK:STDOUT:   %.2: type = generic_class_type @.1 [template]
// CHECK:STDOUT:   %struct.2: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %.3: type = class_type @.1 [template]
// CHECK:STDOUT:   %.4: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %B.decl: B = class_decl @B [template = constants.%struct.1] {
// CHECK:STDOUT:     %N.loc4_9.1: i32 = param N
// CHECK:STDOUT:     %N.loc4_9.2: i32 = bind_symbolic_name N 0, %N.loc4_9.1 [symbolic = constants.%N.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: <invalid> = class_decl @.1 [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.loc12_9.1: type = param T
// CHECK:STDOUT:     %T.loc12_9.2: type = bind_symbolic_name T 0, %T.loc12_9.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc12_9.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %N.loc12_19.1: T = param N
// CHECK:STDOUT:     %N.loc12_19.2: T = bind_symbolic_name N 1, %N.loc12_19.1 [symbolic = constants.%N.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B;
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_count.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %C.1: type = generic_class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: C = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C [template]
// CHECK:STDOUT:   %U: i32 = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %.2: type = generic_class_type @.1 [template]
// CHECK:STDOUT:   %struct.2: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %.3: type = class_type @.1 [template]
// CHECK:STDOUT:   %.4: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: C = class_decl @C [template = constants.%struct.1] {
// CHECK:STDOUT:     %T.loc4_9.1: type = param T
// CHECK:STDOUT:     %T.loc4_9.2: type = bind_symbolic_name T 0, %T.loc4_9.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: <invalid> = class_decl @.1 [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.loc12_9.1: type = param T
// CHECK:STDOUT:     %T.loc12_9.2: type = bind_symbolic_name T 0, %T.loc12_9.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %U.loc12_19.1: i32 = param U
// CHECK:STDOUT:     %U.loc12_19.2: i32 = bind_symbolic_name U 1, %U.loc12_19.1 [symbolic = constants.%U]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %D.1: type = generic_class_type @D [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: D = struct_value () [template]
// CHECK:STDOUT:   %D.2: type = class_type @D [template]
// CHECK:STDOUT:   %T.2: i32 = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.2: type = generic_class_type @.1 [template]
// CHECK:STDOUT:   %struct.2: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %.3: type = class_type @.1 [template]
// CHECK:STDOUT:   %.4: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %D.decl: D = class_decl @D [template = constants.%struct.1] {
// CHECK:STDOUT:     %T.loc4_9.1: type = param T
// CHECK:STDOUT:     %T.loc4_9.2: type = bind_symbolic_name T 0, %T.loc4_9.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: <invalid> = class_decl @.1 [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.loc12_9.1: i32 = param T
// CHECK:STDOUT:     %T.loc12_9.2: i32 = bind_symbolic_name T 0, %T.loc12_9.1 [symbolic = constants.%T.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D;
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_param_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %E.1: type = generic_class_type @E [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: E = struct_value () [template]
// CHECK:STDOUT:   %E.2: type = class_type @E [template]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 0 [symbolic]
// CHECK:STDOUT:   %.2: type = generic_class_type @.1 [template]
// CHECK:STDOUT:   %struct.2: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %.3: type = class_type @.1 [template]
// CHECK:STDOUT:   %.4: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %E.decl: E = class_decl @E [template = constants.%struct.1] {
// CHECK:STDOUT:     %T.loc4_9.1: type = param T
// CHECK:STDOUT:     %T.loc4_9.2: type = bind_symbolic_name T 0, %T.loc4_9.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: <invalid> = class_decl @.1 [template = constants.%struct.2] {
// CHECK:STDOUT:     %U.loc11_9.1: type = param U
// CHECK:STDOUT:     %U.loc11_9.2: type = bind_symbolic_name U 0, %U.loc11_9.1 [symbolic = constants.%U]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E;
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
