// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class A {
  fn F[addr self: A*]();
}

class B {
  var a: A;
}

fn F(s: {.a: A}, b: B) {
  // `s` has only a value representation, so this must be invalid.
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE+6]]:8: ERROR: `addr self` method cannot be invoked on a value.
  // CHECK:STDERR:   s.a.F();
  // CHECK:STDERR:        ^
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE-12]]:13: Initializing `addr self` parameter of method declared here.
  // CHECK:STDERR:   fn F[addr self: A*]();
  // CHECK:STDERR:             ^
  s.a.F();

  // `b` has an object representation for `A`, but this is still invalid for
  // consistency.
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE+6]]:8: ERROR: `addr self` method cannot be invoked on a value.
  // CHECK:STDERR:   b.a.F();
  // CHECK:STDERR:        ^
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE-22]]:13: Initializing `addr self` parameter of method declared here.
  // CHECK:STDERR:   fn F[addr self: A*]();
  // CHECK:STDERR:             ^
  b.a.F();
}

// CHECK:STDOUT: file "fail_memaccess_category.carbon" {
// CHECK:STDOUT:   class_declaration @A, ()
// CHECK:STDOUT:   %A: type = class_type @A
// CHECK:STDOUT:   %.loc9: type = struct_type {}
// CHECK:STDOUT:   class_declaration @B, ()
// CHECK:STDOUT:   %B: type = class_type @B
// CHECK:STDOUT:   %.loc13: type = struct_type {.a: A}
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_reference "A", file.%A
// CHECK:STDOUT:   %.loc9: type = tuple_type ()
// CHECK:STDOUT:   %.loc7: type = ptr_type {}
// CHECK:STDOUT:   %.loc12_8.1: type = unbound_field_type B, A
// CHECK:STDOUT:   %.loc12_8.2: <unbound field of class B> = field "a", member0
// CHECK:STDOUT:   %a: <unbound field of class B> = bind_name "a", %.loc12_8.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[%self.addr: A*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%s: {.a: A}, %b: B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13: type = struct_type {.a: {}*}
// CHECK:STDOUT:   %.loc11: type = ptr_type {.a: A}
// CHECK:STDOUT:   %s.ref: {.a: A} = name_reference "s", %s
// CHECK:STDOUT:   %.loc23_4: A = struct_access %s.ref, member0
// CHECK:STDOUT:   %.loc23_6: <bound method> = bound_method %.loc23_4, @A.%F
// CHECK:STDOUT:   %.loc23_8: init () = call %.loc23_6(<invalid>)
// CHECK:STDOUT:   %b.ref: B = name_reference "b", %b
// CHECK:STDOUT:   %.loc33_4.1: ref A = class_field_access %b.ref, member0
// CHECK:STDOUT:   %.loc33_4.2: A = bind_value %.loc33_4.1
// CHECK:STDOUT:   %.loc33_6: <bound method> = bound_method %.loc33_4.2, @A.%F
// CHECK:STDOUT:   %.loc33_8: init () = call %.loc33_6(<invalid>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
