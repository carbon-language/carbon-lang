// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Deref(n: i32) {
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+4]]:3: ERROR: Cannot dereference operand of non-pointer type `i32`.
  // CHECK:STDERR:   *n;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  *n;
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+4]]:4: ERROR: Cannot apply `->` operator to non-pointer type `i32`.
  // CHECK:STDERR:   n->foo;
  // CHECK:STDERR:    ^~
  // CHECK:STDERR:
  n->foo;
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+4]]:3: ERROR: Cannot dereference operand of non-pointer type `()`.
  // CHECK:STDERR:   *();
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  *();
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+4]]:5: ERROR: Cannot apply `->` operator to non-pointer type `()`.
  // CHECK:STDERR:   ()->foo;
  // CHECK:STDERR:     ^~
  // CHECK:STDERR:
  ()->foo;
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+4]]:3: ERROR: Cannot dereference operand of non-pointer type `{}`.
  // CHECK:STDERR:   *{};
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  *{};
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+3]]:5: ERROR: Cannot apply `->` operator to non-pointer type `{}`.
  // CHECK:STDERR:   {}->foo;
  // CHECK:STDERR:     ^~
  {}->foo;
}

// CHECK:STDOUT: --- fail_deref_not_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Deref: type = fn_type @Deref [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Deref = struct_value () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %struct.2: {} = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Deref = %Deref.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Deref.decl: Deref = fn_decl @Deref [template = constants.%struct.1] {
// CHECK:STDOUT:     %n.loc7_10.1: i32 = param n
// CHECK:STDOUT:     @Deref.%n: i32 = bind_name n, %n.loc7_10.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Deref(%n: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref.loc12: i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc12: ref <error> = deref %n.ref.loc12
// CHECK:STDOUT:   %n.ref.loc17: i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc17: ref <error> = deref %n.ref.loc17
// CHECK:STDOUT:   %.loc22_5.1: () = tuple_literal ()
// CHECK:STDOUT:   %tuple.loc22: () = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc22_5.2: () = converted %.loc22_5.1, %tuple.loc22 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc22_3: ref <error> = deref %.loc22_5.2
// CHECK:STDOUT:   %.loc27_4.1: () = tuple_literal ()
// CHECK:STDOUT:   %tuple.loc27: () = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc27_4.2: () = converted %.loc27_4.1, %tuple.loc27 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc27_5: ref <error> = deref %.loc27_4.2
// CHECK:STDOUT:   %.loc32_5.1: {} = struct_literal ()
// CHECK:STDOUT:   %struct.loc32: {} = struct_value () [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc32_5.2: {} = converted %.loc32_5.1, %struct.loc32 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc32_3: ref <error> = deref %.loc32_5.2
// CHECK:STDOUT:   %.loc36_4.1: {} = struct_literal ()
// CHECK:STDOUT:   %struct.loc36: {} = struct_value () [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc36_4.2: {} = converted %.loc36_4.1, %struct.loc36 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc36_5: ref <error> = deref %.loc36_4.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
