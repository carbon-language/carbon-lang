// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {
  fn Member[self: Self]();
  var field: C*;
}

fn Foo(ptr: C*) {
  (*ptr).Member();
  ptr->Member();

  (*ptr).field;
  ptr->field;

  ptr->field->field;
}

// CHECK:STDOUT: --- arrow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = ptr_type C [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type C, C* [template]
// CHECK:STDOUT:   %.3: type = struct_type {.field: C*} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.field: C*} [template]
// CHECK:STDOUT:   %.5: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo [template] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc12: type = ptr_type C [template = constants.%.1]
// CHECK:STDOUT:     %ptr.loc12_8.1: C* = param ptr
// CHECK:STDOUT:     @Foo.%ptr: C* = bind_name ptr, %ptr.loc12_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Member: <function> = fn_decl @Member [template] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %self.loc8_13.1: C = param self
// CHECK:STDOUT:     %self.loc8_13.2: C = bind_name self, %self.loc8_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc9_15: type = ptr_type C [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_12: <unbound element of class C> = field_decl field, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .Member = %Member
// CHECK:STDOUT:   .field = %.loc9_12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Member[@C.%self.loc8_13.2: C]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%ptr: C*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ptr.ref.loc13: C* = name_ref ptr, %ptr
// CHECK:STDOUT:   %.loc13_4.1: ref C = deref %ptr.ref.loc13
// CHECK:STDOUT:   %Member.ref.loc13: <function> = name_ref Member, @C.%Member [template = @C.%Member]
// CHECK:STDOUT:   %.loc13_9: <bound method> = bound_method %.loc13_4.1, %Member.ref.loc13
// CHECK:STDOUT:   %.loc13_4.2: C = bind_value %.loc13_4.1
// CHECK:STDOUT:   %.loc13_16: init () = call %.loc13_9(%.loc13_4.2)
// CHECK:STDOUT:   %ptr.ref.loc14: C* = name_ref ptr, %ptr
// CHECK:STDOUT:   %.loc14_6.1: ref C = deref %ptr.ref.loc14
// CHECK:STDOUT:   %Member.ref.loc14: <function> = name_ref Member, @C.%Member [template = @C.%Member]
// CHECK:STDOUT:   %.loc14_6.2: <bound method> = bound_method %.loc14_6.1, %Member.ref.loc14
// CHECK:STDOUT:   %.loc14_6.3: C = bind_value %.loc14_6.1
// CHECK:STDOUT:   %.loc14_14: init () = call %.loc14_6.2(%.loc14_6.3)
// CHECK:STDOUT:   %ptr.ref.loc16: C* = name_ref ptr, %ptr
// CHECK:STDOUT:   %.loc16_4: ref C = deref %ptr.ref.loc16
// CHECK:STDOUT:   %field.ref.loc16: <unbound element of class C> = name_ref field, @C.%.loc9_12 [template = @C.%.loc9_12]
// CHECK:STDOUT:   %.loc16_9: ref C* = class_element_access %.loc16_4, element0
// CHECK:STDOUT:   %ptr.ref.loc17: C* = name_ref ptr, %ptr
// CHECK:STDOUT:   %.loc17_6.1: ref C = deref %ptr.ref.loc17
// CHECK:STDOUT:   %field.ref.loc17: <unbound element of class C> = name_ref field, @C.%.loc9_12 [template = @C.%.loc9_12]
// CHECK:STDOUT:   %.loc17_6.2: ref C* = class_element_access %.loc17_6.1, element0
// CHECK:STDOUT:   %ptr.ref.loc19: C* = name_ref ptr, %ptr
// CHECK:STDOUT:   %.loc19_6.1: ref C = deref %ptr.ref.loc19
// CHECK:STDOUT:   %field.ref.loc19_6: <unbound element of class C> = name_ref field, @C.%.loc9_12 [template = @C.%.loc9_12]
// CHECK:STDOUT:   %.loc19_6.2: ref C* = class_element_access %.loc19_6.1, element0
// CHECK:STDOUT:   %.loc19_6.3: C* = bind_value %.loc19_6.2
// CHECK:STDOUT:   %.loc19_13.1: ref C = deref %.loc19_6.3
// CHECK:STDOUT:   %field.ref.loc19_13: <unbound element of class C> = name_ref field, @C.%.loc9_12 [template = @C.%.loc9_12]
// CHECK:STDOUT:   %.loc19_13.2: ref C* = class_element_access %.loc19_13.1, element0
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
