// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 { return 0; }

fn Main() {
  var a: i32 = 3;
  // CHECK:STDERR: fail_redundant_compound_access.carbon:[[@LINE+3]]:7: ERROR: Member name of type `i32` in compound member access is not an instance member or an interface member.
  // CHECK:STDERR:   a = a.(a);
  // CHECK:STDERR:       ^~~~~
  a = a.(a);
  // CHECK:STDERR: fail_redundant_compound_access.carbon:[[@LINE+3]]:7: ERROR: Member name of type `<function>` in compound member access is not an instance member or an interface member.
  // CHECK:STDERR:   a = a.(F)();
  // CHECK:STDERR:       ^~~~~
  a = a.(F)();
}

// CHECK:STDOUT: --- fail_redundant_compound_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:     .Main = %Main
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   return %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc10: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   assign %a.var, %.loc10
// CHECK:STDOUT:   %a.ref.loc14_3: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %a.ref.loc14_7: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %a.ref.loc14_10: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc14: i32 = bind_value %a.ref.loc14_10
// CHECK:STDOUT:   assign %a.ref.loc14_3, %.loc14
// CHECK:STDOUT:   %a.ref.loc18_3: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %a.ref.loc18_7: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%F [template = file.%F]
// CHECK:STDOUT:   %.loc18: init i32 = call %F.ref()
// CHECK:STDOUT:   assign %a.ref.loc18_3, %.loc18
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
