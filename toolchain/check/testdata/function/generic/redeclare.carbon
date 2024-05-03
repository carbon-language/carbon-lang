// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- redeclare.carbon

library "redeclare" api;

fn F(T:! type) -> T*;

fn F(T:! type) -> T* {
  return F(T);
}

// --- fail_different_return_type.carbon

library "fail_different_return_type" api;

fn F(T:! type, U:! type) -> T*;

// CHECK:STDERR: fail_different_return_type.carbon:[[@LINE+7]]:1: ERROR: Function redeclaration differs because return type is `U*`.
// CHECK:STDERR: fn F(T:! type, U:! type) -> U* {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_different_return_type.carbon:[[@LINE-5]]:1: Previously declared with return type `T*`.
// CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! type, U:! type) -> U* {
  // CHECK:STDERR: fail_different_return_type.carbon:[[@LINE+11]]:3: ERROR: Cannot implicitly convert from `T*` to `U*`.
  // CHECK:STDERR:   return F(T);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_different_return_type.carbon:[[@LINE+7]]:10: ERROR: 1 argument(s) passed to function expecting 2 argument(s).
  // CHECK:STDERR:   return F(T);
  // CHECK:STDERR:          ^~
  // CHECK:STDERR: fail_different_return_type.carbon:[[@LINE-17]]:1: Calling function declared here.
  // CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(T);
}

// --- fail_reorder.carbon

library "fail_reorder" api;

fn F(T:! type, U:! type) -> T*;

// CHECK:STDERR: fail_reorder.carbon:[[@LINE+7]]:6: ERROR: Redeclaration differs at parameter 1.
// CHECK:STDERR: fn F(U:! type, T:! type) -> T* {
// CHECK:STDERR:      ^
// CHECK:STDERR: fail_reorder.carbon:[[@LINE-5]]:6: Previous declaration's corresponding parameter here.
// CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn F(U:! type, T:! type) -> T* {
  // CHECK:STDERR: fail_reorder.carbon:[[@LINE+11]]:3: ERROR: Cannot implicitly convert from `T*` to `T*`.
  // CHECK:STDERR:   return F(T);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_reorder.carbon:[[@LINE+7]]:10: ERROR: 1 argument(s) passed to function expecting 2 argument(s).
  // CHECK:STDERR:   return F(T);
  // CHECK:STDERR:          ^~
  // CHECK:STDERR: fail_reorder.carbon:[[@LINE-17]]:1: Calling function declared here.
  // CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(T);
}

// --- fail_rename.carbon

library "fail_rename" api;

fn F(T:! type, U:! type) -> T*;

// CHECK:STDERR: fail_rename.carbon:[[@LINE+7]]:6: ERROR: Redeclaration differs at parameter 1.
// CHECK:STDERR: fn F(U:! type, T:! type) -> U* {
// CHECK:STDERR:      ^
// CHECK:STDERR: fail_rename.carbon:[[@LINE-5]]:6: Previous declaration's corresponding parameter here.
// CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn F(U:! type, T:! type) -> U* {
  // CHECK:STDERR: fail_rename.carbon:[[@LINE+10]]:3: ERROR: Cannot implicitly convert from `T*` to `U*`.
  // CHECK:STDERR:   return F(T);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_rename.carbon:[[@LINE+6]]:10: ERROR: 1 argument(s) passed to function expecting 2 argument(s).
  // CHECK:STDERR:   return F(T);
  // CHECK:STDERR:          ^~
  // CHECK:STDERR: fail_rename.carbon:[[@LINE-17]]:1: Calling function declared here.
  // CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return F(T);
}

// CHECK:STDOUT: --- redeclare.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type T [symbolic]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: F = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl.loc4: F = fn_decl @F [template = constants.%struct] {
// CHECK:STDOUT:     %T.loc4_6.1: type = param T
// CHECK:STDOUT:     %T.loc4_6.2: type = bind_symbolic_name T 0, %T.loc4_6.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.ref.loc4: type = name_ref T, %T.loc4_6.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc4: type = ptr_type T [symbolic = constants.%.1]
// CHECK:STDOUT:     %return.var.loc4: ref T* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc6: F = fn_decl @F [template = constants.%struct] {
// CHECK:STDOUT:     %T.loc6_6.1: type = param T
// CHECK:STDOUT:     @F.%T: type = bind_symbolic_name T 0, %T.loc6_6.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %T.ref.loc6: type = name_ref T, @F.%T [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc6: type = ptr_type T [symbolic = constants.%.1]
// CHECK:STDOUT:     @F.%return: ref T* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%T: type) -> T* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: F = name_ref F, file.%F.decl.loc4 [template = constants.%struct]
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %F.call: init T* = call %F.ref(%T.ref)
// CHECK:STDOUT:   %.loc7_14.1: T* = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc7_14.2: T* = converted %F.call, %.loc7_14.1
// CHECK:STDOUT:   return %.loc7_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_different_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type T [symbolic]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type U [symbolic]
// CHECK:STDOUT:   %.4: type = fn_type @.1 [template]
// CHECK:STDOUT:   %struct.2: <invalid> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     %T.loc4_6.1: type = param T
// CHECK:STDOUT:     @F.%T: type = bind_symbolic_name T 0, %T.loc4_6.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %U.loc4_16.1: type = param U
// CHECK:STDOUT:     @F.%U: type = bind_symbolic_name U 1, %U.loc4_16.1 [symbolic = constants.%U]
// CHECK:STDOUT:     %T.ref: type = name_ref T, @F.%T [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc4: type = ptr_type T [symbolic = constants.%.1]
// CHECK:STDOUT:     @F.%return: ref T* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: <invalid> = fn_decl @.1 [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.loc13_6.1: type = param T
// CHECK:STDOUT:     @.1.%T: type = bind_symbolic_name T 0, %T.loc13_6.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %U.loc13_16.1: type = param U
// CHECK:STDOUT:     @.1.%U: type = bind_symbolic_name U 1, %U.loc13_16.1 [symbolic = constants.%U]
// CHECK:STDOUT:     %U.ref: type = name_ref U, @.1.%U [symbolic = constants.%U]
// CHECK:STDOUT:     %.loc13: type = ptr_type U [symbolic = constants.%.3]
// CHECK:STDOUT:     @.1.%return: ref U* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%T: type, %U: type) -> T*;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%T: type, %U: type) -> U* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: F = name_ref F, file.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %F.call: init T* = call %F.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_reorder.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type T [symbolic]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %U.2: type = bind_symbolic_name U 0 [symbolic]
// CHECK:STDOUT:   %T.2: type = bind_symbolic_name T 1 [symbolic]
// CHECK:STDOUT:   %.3: type = ptr_type T [symbolic]
// CHECK:STDOUT:   %.4: type = fn_type @.1 [template]
// CHECK:STDOUT:   %struct.2: <invalid> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     %T.loc4_6.1: type = param T
// CHECK:STDOUT:     @F.%T: type = bind_symbolic_name T 0, %T.loc4_6.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:     %U.loc4_16.1: type = param U
// CHECK:STDOUT:     @F.%U: type = bind_symbolic_name U 1, %U.loc4_16.1 [symbolic = constants.%U.1]
// CHECK:STDOUT:     %T.ref.loc4: type = name_ref T, @F.%T [symbolic = constants.%T.1]
// CHECK:STDOUT:     %.loc4: type = ptr_type T [symbolic = constants.%.1]
// CHECK:STDOUT:     @F.%return: ref T* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: <invalid> = fn_decl @.1 [template = constants.%struct.2] {
// CHECK:STDOUT:     %U.loc13_6.1: type = param U
// CHECK:STDOUT:     @.1.%U: type = bind_symbolic_name U 0, %U.loc13_6.1 [symbolic = constants.%U.2]
// CHECK:STDOUT:     %T.loc13_16.1: type = param T
// CHECK:STDOUT:     @.1.%T: type = bind_symbolic_name T 1, %T.loc13_16.1 [symbolic = constants.%T.2]
// CHECK:STDOUT:     %T.ref.loc13: type = name_ref T, @.1.%T [symbolic = constants.%T.2]
// CHECK:STDOUT:     %.loc13: type = ptr_type T [symbolic = constants.%.3]
// CHECK:STDOUT:     @.1.%return: ref T* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%T: type, %U: type) -> T*;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%U: type, %T: type) -> T* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: F = name_ref F, file.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T [symbolic = constants.%T.2]
// CHECK:STDOUT:   %F.call: init T* = call %F.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_rename.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %U.1: type = bind_symbolic_name U 1 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type T [symbolic]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %U.2: type = bind_symbolic_name U 0 [symbolic]
// CHECK:STDOUT:   %T.2: type = bind_symbolic_name T 1 [symbolic]
// CHECK:STDOUT:   %.3: type = ptr_type U [symbolic]
// CHECK:STDOUT:   %.4: type = fn_type @.1 [template]
// CHECK:STDOUT:   %struct.2: <invalid> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     %T.loc4_6.1: type = param T
// CHECK:STDOUT:     @F.%T: type = bind_symbolic_name T 0, %T.loc4_6.1 [symbolic = constants.%T.1]
// CHECK:STDOUT:     %U.loc4_16.1: type = param U
// CHECK:STDOUT:     @F.%U: type = bind_symbolic_name U 1, %U.loc4_16.1 [symbolic = constants.%U.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, @F.%T [symbolic = constants.%T.1]
// CHECK:STDOUT:     %.loc4: type = ptr_type T [symbolic = constants.%.1]
// CHECK:STDOUT:     @F.%return: ref T* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: <invalid> = fn_decl @.1 [template = constants.%struct.2] {
// CHECK:STDOUT:     %U.loc13_6.1: type = param U
// CHECK:STDOUT:     @.1.%U: type = bind_symbolic_name U 0, %U.loc13_6.1 [symbolic = constants.%U.2]
// CHECK:STDOUT:     %T.loc13_16.1: type = param T
// CHECK:STDOUT:     @.1.%T: type = bind_symbolic_name T 1, %T.loc13_16.1 [symbolic = constants.%T.2]
// CHECK:STDOUT:     %U.ref: type = name_ref U, @.1.%U [symbolic = constants.%U.2]
// CHECK:STDOUT:     %.loc13: type = ptr_type U [symbolic = constants.%.3]
// CHECK:STDOUT:     @.1.%return: ref U* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%T: type, %U: type) -> T*;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%U: type, %T: type) -> U* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: F = name_ref F, file.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T [symbolic = constants.%T.2]
// CHECK:STDOUT:   %F.call: init T* = call %F.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
