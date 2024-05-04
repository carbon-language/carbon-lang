// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 { return 0; }

fn Main() {
  var a: i32 = 3;
  // CHECK:STDERR: fail_redundant_compound_access.carbon:[[@LINE+4]]:7: ERROR: Member name of type `i32` in compound member access is not an instance member or an interface member.
  // CHECK:STDERR:   a = a.(a);
  // CHECK:STDERR:       ^~~~~
  // CHECK:STDERR:
  a = a.(a);
  // CHECK:STDERR: fail_redundant_compound_access.carbon:[[@LINE+3]]:7: ERROR: Member name of type `F` in compound member access is not an instance member or an interface member.
  // CHECK:STDERR:   a = a.(F)();
  // CHECK:STDERR:       ^~~~~
  a = a.(F)();
}

// CHECK:STDOUT: --- fail_redundant_compound_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %Main: type = fn_type @Main [template]
// CHECK:STDOUT:   %struct.2: Main = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: Main = fn_decl @Main [template = constants.%struct.2] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   return %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc10: i32 = int_literal 3 [template = constants.%.3]
// CHECK:STDOUT:   assign %a.var, %.loc10
// CHECK:STDOUT:   %a.ref.loc15_3: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %a.ref.loc15_7: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %a.ref.loc15_10: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc15: i32 = bind_value %a.ref.loc15_10
// CHECK:STDOUT:   assign %a.ref.loc15_3, %.loc15
// CHECK:STDOUT:   %a.ref.loc19_3: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %a.ref.loc19_7: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %F.ref: F = name_ref F, file.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %F.call: init i32 = call %F.ref()
// CHECK:STDOUT:   assign %a.ref.loc19_3, %F.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
