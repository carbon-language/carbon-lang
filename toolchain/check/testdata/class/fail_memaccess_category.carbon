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
  // CHECK:STDERR:             ^~~~
  s.a.F();

  // `b` has an object representation for `A`, but this is still invalid for
  // consistency.
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE+6]]:8: ERROR: `addr self` method cannot be invoked on a value.
  // CHECK:STDERR:   b.a.F();
  // CHECK:STDERR:        ^
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE-22]]:13: Initializing `addr self` parameter of method declared here.
  // CHECK:STDERR:   fn F[addr self: A*]();
  // CHECK:STDERR:             ^~~~
  b.a.F();
}

// CHECK:STDOUT: --- fail_memaccess_category.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = ptr_type A [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %.4: type = ptr_type {} [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type B, A [template]
// CHECK:STDOUT:   %.6: type = struct_type {.a: A} [template]
// CHECK:STDOUT:   %.7: type = struct_type {.a: {}*} [template]
// CHECK:STDOUT:   %.8: type = ptr_type {.a: A} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.A = %A.decl, .B = %B.decl, .F = %F} [template]
// CHECK:STDOUT:   %A.decl = class_decl @A, ()
// CHECK:STDOUT:   %B.decl = class_decl @B, ()
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, constants.%A [template = constants.%A]
// CHECK:STDOUT:   %.loc12: <unbound element of class B> = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[addr %self: A*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%s: {.a: A}, %b: B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: {.a: A} = name_ref s, %s
// CHECK:STDOUT:   %.loc23_4: A = struct_access %s.ref, element0
// CHECK:STDOUT:   %.loc23_6: <bound method> = bound_method %.loc23_4, @A.%F
// CHECK:STDOUT:   %.loc23_8: init () = call %.loc23_6(<invalid>)
// CHECK:STDOUT:   %b.ref: B = name_ref b, %b
// CHECK:STDOUT:   %.loc33_4.1: ref A = class_element_access %b.ref, element0
// CHECK:STDOUT:   %.loc33_4.2: A = bind_value %.loc33_4.1
// CHECK:STDOUT:   %.loc33_6: <bound method> = bound_method %.loc33_4.2, @A.%F
// CHECK:STDOUT:   %.loc33_8: init () = call %.loc33_6(<invalid>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
