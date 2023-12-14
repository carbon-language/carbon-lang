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
// CHECK:STDOUT:   %.loc9_1.1: type = struct_type {}
// CHECK:STDOUT:   %.loc9_1.2: type = tuple_type ()
// CHECK:STDOUT:   %.loc7: type = ptr_type {}
// CHECK:STDOUT:   %.loc14_1.1: type = struct_type {.base: A}
// CHECK:STDOUT:   %.loc14_1.2: type = struct_type {.base: {}*}
// CHECK:STDOUT:   %.loc11: type = ptr_type {.base: A}
// CHECK:STDOUT:   %.loc19_1.1: type = struct_type {.base: B}
// CHECK:STDOUT:   %.loc19_1.2: type = struct_type {.base: {.base: A}*}
// CHECK:STDOUT:   %.loc16: type = ptr_type {.base: B}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.A = %A.decl, .B = %B.decl, .C = %C.decl, .D = %D.decl, .Call = %Call}
// CHECK:STDOUT:   %A.decl = class_decl @A, ()
// CHECK:STDOUT:   %A: type = class_type @A
// CHECK:STDOUT:   %B.decl = class_decl @B, ()
// CHECK:STDOUT:   %B: type = class_type @B
// CHECK:STDOUT:   %C.decl = class_decl @C, ()
// CHECK:STDOUT:   %C: type = class_type @C
// CHECK:STDOUT:   %D.decl = class_decl @D, ()
// CHECK:STDOUT:   %D: type = class_type @D
// CHECK:STDOUT:   %Call: <function> = fn_decl @Call
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
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A
// CHECK:STDOUT:   %.loc12_17.1: type = unbound_element_type B, A
// CHECK:STDOUT:   %.loc12_17.2: <unbound element of class B> = base_decl A, element0
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc12_17.2
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   extend name_scope1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B
// CHECK:STDOUT:   %.loc17_17.1: type = unbound_element_type C, B
// CHECK:STDOUT:   %.loc17_17.2: <unbound element of class C> = base_decl B, element0
// CHECK:STDOUT:   %F: <function> = fn_decl @F.3
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc17_17.2
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B
// CHECK:STDOUT:   %.loc22_17.1: type = unbound_element_type D, B
// CHECK:STDOUT:   %.loc22_17.2: <unbound element of class D> = base_decl B, element0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc22_17.2
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[addr %self: A*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[addr %self: B*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[addr %self: C*]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a: A*, %b: B*, %c: C*, %d: D*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: A* = name_ref a, %a
// CHECK:STDOUT:   %.loc26_4.1: ref A = deref %a.ref
// CHECK:STDOUT:   %.loc26_7: <bound method> = bound_method %.loc26_4.1, @A.%F
// CHECK:STDOUT:   %.loc26_4.2: A* = address_of %.loc26_4.1
// CHECK:STDOUT:   %.loc26_9: init () = call %.loc26_7(%.loc26_4.2)
// CHECK:STDOUT:   %b.ref: B* = name_ref b, %b
// CHECK:STDOUT:   %.loc27_4.1: ref B = deref %b.ref
// CHECK:STDOUT:   %.loc27_7: <bound method> = bound_method %.loc27_4.1, @B.%F
// CHECK:STDOUT:   %.loc27_4.2: B* = address_of %.loc27_4.1
// CHECK:STDOUT:   %.loc27_9: init () = call %.loc27_7(%.loc27_4.2)
// CHECK:STDOUT:   %c.ref: C* = name_ref c, %c
// CHECK:STDOUT:   %.loc28_4.1: ref C = deref %c.ref
// CHECK:STDOUT:   %.loc28_7: <bound method> = bound_method %.loc28_4.1, @C.%F
// CHECK:STDOUT:   %.loc28_4.2: C* = address_of %.loc28_4.1
// CHECK:STDOUT:   %.loc28_9: init () = call %.loc28_7(%.loc28_4.2)
// CHECK:STDOUT:   %d.ref: D* = name_ref d, %d
// CHECK:STDOUT:   %.loc29_4.1: ref D = deref %d.ref
// CHECK:STDOUT:   %.loc29_7: <bound method> = bound_method %.loc29_4.1, @B.%F
// CHECK:STDOUT:   %.loc29_4.2: D* = address_of %.loc29_4.1
// CHECK:STDOUT:   %.loc29_9.1: ref D = deref %.loc29_4.2
// CHECK:STDOUT:   %.loc29_9.2: ref B = class_element_access %.loc29_9.1, element0
// CHECK:STDOUT:   %.loc29_9.3: B* = address_of %.loc29_9.2
// CHECK:STDOUT:   %.loc29_9.4: B* = converted %.loc29_4.2, %.loc29_9.3
// CHECK:STDOUT:   %.loc29_9.5: init () = call %.loc29_7(%.loc29_9.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
