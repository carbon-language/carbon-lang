// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class A {
  fn F[addr self: Self*]();
}

base class B {
  extend base: A;
  fn F[addr self: Self*]();
}

class C {
  extend base: B;
  fn F[addr self: Self*]();
}

class D {
  extend base: B;
}

fn Call(a: A*, b: B*, c: C*, d: D*) {
  (*a).F();
  (*b).F();
  (*c).F();
  (*d).F();
}

// CHECK:STDOUT: --- base_method_shadow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = ptr_type A [template]
// CHECK:STDOUT:   %F.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.4: type = ptr_type {} [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type B, A [template]
// CHECK:STDOUT:   %.6: type = ptr_type B [template]
// CHECK:STDOUT:   %F.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.7: type = struct_type {.base: A} [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: {}*} [template]
// CHECK:STDOUT:   %.9: type = ptr_type {.base: A} [template]
// CHECK:STDOUT:   %.10: type = unbound_element_type C, B [template]
// CHECK:STDOUT:   %.11: type = ptr_type C [template]
// CHECK:STDOUT:   %F.3: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %struct.3: F = struct_value () [template]
// CHECK:STDOUT:   %.12: type = struct_type {.base: B} [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.13: type = unbound_element_type D, B [template]
// CHECK:STDOUT:   %.14: type = ptr_type D [template]
// CHECK:STDOUT:   %Call: type = fn_type @Call [template]
// CHECK:STDOUT:   %struct.4: Call = struct_value () [template]
// CHECK:STDOUT:   %.15: type = struct_type {.base: {.base: A}*} [template]
// CHECK:STDOUT:   %.16: type = ptr_type {.base: B} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %Call.decl: Call = fn_decl @Call [template = constants.%struct.4] {
// CHECK:STDOUT:     %A.ref: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc25_13: type = ptr_type A [template = constants.%.1]
// CHECK:STDOUT:     %a.loc25_9.1: A* = param a
// CHECK:STDOUT:     @Call.%a: A* = bind_name a, %a.loc25_9.1
// CHECK:STDOUT:     %B.ref: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc25_20: type = ptr_type B [template = constants.%.6]
// CHECK:STDOUT:     %b.loc25_16.1: B* = param b
// CHECK:STDOUT:     @Call.%b: B* = bind_name b, %b.loc25_16.1
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc25_27: type = ptr_type C [template = constants.%.11]
// CHECK:STDOUT:     %c.loc25_23.1: C* = param c
// CHECK:STDOUT:     @Call.%c: C* = bind_name c, %c.loc25_23.1
// CHECK:STDOUT:     %D.ref: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %.loc25_34: type = ptr_type D [template = constants.%.14]
// CHECK:STDOUT:     %d.loc25_30.1: D* = param d
// CHECK:STDOUT:     @Call.%d: D* = bind_name d, %d.loc25_30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A [template = constants.%A]
// CHECK:STDOUT:     %.loc8_23: type = ptr_type A [template = constants.%.1]
// CHECK:STDOUT:     %self.loc8_13.1: A* = param self
// CHECK:STDOUT:     %self.loc8_13.3: A* = bind_name self, %self.loc8_13.1
// CHECK:STDOUT:     %.loc8_8: A* = addr_pattern %self.loc8_13.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc12: <unbound element of class B> = base_decl A, element0 [template]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.2 [template = constants.%struct.2] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [template = constants.%B]
// CHECK:STDOUT:     %.loc13_23: type = ptr_type B [template = constants.%.6]
// CHECK:STDOUT:     %self.loc13_13.1: B* = param self
// CHECK:STDOUT:     %self.loc13_13.3: B* = bind_name self, %self.loc13_13.1
// CHECK:STDOUT:     %.loc13_8: B* = addr_pattern %self.loc13_13.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc12
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc17: <unbound element of class C> = base_decl B, element0 [template]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.3 [template = constants.%struct.3] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %.loc18_23: type = ptr_type C [template = constants.%.11]
// CHECK:STDOUT:     %self.loc18_13.1: C* = param self
// CHECK:STDOUT:     %self.loc18_13.3: C* = bind_name self, %self.loc18_13.1
// CHECK:STDOUT:     %.loc18_8: C* = addr_pattern %self.loc18_13.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc17
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend name_scope3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc22: <unbound element of class D> = base_decl B, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .base = %.loc22
// CHECK:STDOUT:   extend name_scope3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[addr @A.%self.loc8_13.3: A*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[addr @B.%self.loc13_13.3: B*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[addr @C.%self.loc18_13.3: C*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a: A*, %b: B*, %c: C*, %d: D*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: A* = name_ref a, %a
// CHECK:STDOUT:   %.loc26_4.1: ref A = deref %a.ref
// CHECK:STDOUT:   %F.ref.loc26: F = name_ref F, @A.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc26_7: <bound method> = bound_method %.loc26_4.1, %F.ref.loc26
// CHECK:STDOUT:   %.loc26_4.2: A* = addr_of %.loc26_4.1
// CHECK:STDOUT:   %F.call.loc26: init () = call %.loc26_7(%.loc26_4.2)
// CHECK:STDOUT:   %b.ref: B* = name_ref b, %b
// CHECK:STDOUT:   %.loc27_4.1: ref B = deref %b.ref
// CHECK:STDOUT:   %F.ref.loc27: F = name_ref F, @B.%F.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc27_7: <bound method> = bound_method %.loc27_4.1, %F.ref.loc27
// CHECK:STDOUT:   %.loc27_4.2: B* = addr_of %.loc27_4.1
// CHECK:STDOUT:   %F.call.loc27: init () = call %.loc27_7(%.loc27_4.2)
// CHECK:STDOUT:   %c.ref: C* = name_ref c, %c
// CHECK:STDOUT:   %.loc28_4.1: ref C = deref %c.ref
// CHECK:STDOUT:   %F.ref.loc28: F = name_ref F, @C.%F.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc28_7: <bound method> = bound_method %.loc28_4.1, %F.ref.loc28
// CHECK:STDOUT:   %.loc28_4.2: C* = addr_of %.loc28_4.1
// CHECK:STDOUT:   %F.call.loc28: init () = call %.loc28_7(%.loc28_4.2)
// CHECK:STDOUT:   %d.ref: D* = name_ref d, %d
// CHECK:STDOUT:   %.loc29_4.1: ref D = deref %d.ref
// CHECK:STDOUT:   %F.ref.loc29: F = name_ref F, @B.%F.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc29_7: <bound method> = bound_method %.loc29_4.1, %F.ref.loc29
// CHECK:STDOUT:   %.loc29_4.2: D* = addr_of %.loc29_4.1
// CHECK:STDOUT:   %.loc29_9.1: ref D = deref %.loc29_4.2
// CHECK:STDOUT:   %.loc29_9.2: ref B = class_element_access %.loc29_9.1, element0
// CHECK:STDOUT:   %.loc29_9.3: B* = addr_of %.loc29_9.2
// CHECK:STDOUT:   %.loc29_9.4: B* = converted %.loc29_4.2, %.loc29_9.3
// CHECK:STDOUT:   %F.call.loc29: init () = call %.loc29_7(%.loc29_9.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
