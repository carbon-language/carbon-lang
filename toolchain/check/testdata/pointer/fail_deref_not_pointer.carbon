// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Deref(n: i32) {
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+3]]:3: ERROR: Cannot dereference operand of non-pointer type `i32`.
  // CHECK:STDERR:   *n;
  // CHECK:STDERR:   ^
  *n;
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+3]]:4: ERROR: Cannot apply `->` operator to non-pointer type `i32`.
  // CHECK:STDERR:   n->foo;
  // CHECK:STDERR:    ^~
  n->foo;
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+3]]:3: ERROR: Cannot dereference operand of non-pointer type `()`.
  // CHECK:STDERR:   *();
  // CHECK:STDERR:   ^
  *();
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+3]]:5: ERROR: Cannot apply `->` operator to non-pointer type `()`.
  // CHECK:STDERR:   ()->foo;
  // CHECK:STDERR:     ^~
  ()->foo;
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+3]]:3: ERROR: Cannot dereference operand of non-pointer type `{}`.
  // CHECK:STDERR:   *{};
  // CHECK:STDERR:   ^
  *{};
  // CHECK:STDERR: fail_deref_not_pointer.carbon:[[@LINE+3]]:5: ERROR: Cannot apply `->` operator to non-pointer type `{}`.
  // CHECK:STDERR:   {}->foo;
  // CHECK:STDERR:     ^~
  {}->foo;
}

// CHECK:STDOUT: --- fail_deref_not_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: () = tuple_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: {} = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Deref = %Deref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Deref: <function> = fn_decl @Deref [template] {
// CHECK:STDOUT:     %n.loc7_10.1: i32 = param n
// CHECK:STDOUT:     @Deref.%n: i32 = bind_name n, %n.loc7_10.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Deref(%n: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref.loc11: i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc11: ref <error> = deref %n.ref.loc11
// CHECK:STDOUT:   %n.ref.loc15: i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc15: ref <error> = deref %n.ref.loc15
// CHECK:STDOUT:   %.loc19_5.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc19_5.2: () = tuple_value () [template = constants.%.2]
// CHECK:STDOUT:   %.loc19_5.3: () = converted %.loc19_5.1, %.loc19_5.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc19_3: ref <error> = deref %.loc19_5.3
// CHECK:STDOUT:   %.loc23_4.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc23_4.2: () = tuple_value () [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_4.3: () = converted %.loc23_4.1, %.loc23_4.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_5: ref <error> = deref %.loc23_4.3
// CHECK:STDOUT:   %.loc27_5.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc27_5.2: {} = struct_value () [template = constants.%.4]
// CHECK:STDOUT:   %.loc27_5.3: {} = converted %.loc27_5.1, %.loc27_5.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc27_3: ref <error> = deref %.loc27_5.3
// CHECK:STDOUT:   %.loc31_4.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc31_4.2: {} = struct_value () [template = constants.%.4]
// CHECK:STDOUT:   %.loc31_4.3: {} = converted %.loc31_4.1, %.loc31_4.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc31_5: ref <error> = deref %.loc31_4.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
