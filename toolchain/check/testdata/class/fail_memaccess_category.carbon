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
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE+7]]:8: ERROR: `addr self` method cannot be invoked on a value.
  // CHECK:STDERR:   s.a.F();
  // CHECK:STDERR:        ^
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE-12]]:13: Initializing `addr self` parameter of method declared here.
  // CHECK:STDERR:   fn F[addr self: A*]();
  // CHECK:STDERR:             ^~~~
  // CHECK:STDERR:
  s.a.F();

  // `b` has an object representation for `A`, but this is still invalid for
  // consistency.
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE+6]]:8: ERROR: `addr self` method cannot be invoked on a value.
  // CHECK:STDERR:   b.a.F();
  // CHECK:STDERR:        ^
  // CHECK:STDERR: fail_memaccess_category.carbon:[[@LINE-23]]:13: Initializing `addr self` parameter of method declared here.
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
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template] {
// CHECK:STDOUT:     %A.ref: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc15: type = struct_type {.a: A} [template = constants.%.6]
// CHECK:STDOUT:     %s.loc15_6.1: {.a: A} = param s
// CHECK:STDOUT:     @F.2.%s: {.a: A} = bind_name s, %s.loc15_6.1
// CHECK:STDOUT:     %B.ref: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:     %b.loc15_18.1: B = param b
// CHECK:STDOUT:     @F.2.%b: B = bind_name b, %b.loc15_18.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template] {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc8_20: type = ptr_type A [template = constants.%.1]
// CHECK:STDOUT:     %self.loc8_13.1: A* = param self
// CHECK:STDOUT:     %self.loc8_13.3: A* = bind_name self, %self.loc8_13.1
// CHECK:STDOUT:     %.loc8_8: A* = addr_pattern %self.loc8_13.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc12: <unbound element of class B> = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .a = %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[addr @A.%self.loc8_13.3: A*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%s: {.a: A}, %b: B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: {.a: A} = name_ref s, %s
// CHECK:STDOUT:   %.loc24_4: A = struct_access %s.ref, element0
// CHECK:STDOUT:   %F.ref.loc24: <function> = name_ref F, @A.%F [template = @A.%F]
// CHECK:STDOUT:   %.loc24_6: <bound method> = bound_method %.loc24_4, %F.ref.loc24
// CHECK:STDOUT:   %.loc24_8: init () = call %.loc24_6(<invalid>) [template = <error>]
// CHECK:STDOUT:   %b.ref: B = name_ref b, %b
// CHECK:STDOUT:   %a.ref: <unbound element of class B> = name_ref a, @B.%.loc12 [template = @B.%.loc12]
// CHECK:STDOUT:   %.loc34_4.1: ref A = class_element_access %b.ref, element0
// CHECK:STDOUT:   %.loc34_4.2: A = bind_value %.loc34_4.1
// CHECK:STDOUT:   %F.ref.loc34: <function> = name_ref F, @A.%F [template = @A.%F]
// CHECK:STDOUT:   %.loc34_6: <bound method> = bound_method %.loc34_4.2, %F.ref.loc34
// CHECK:STDOUT:   %.loc34_8: init () = call %.loc34_6(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
