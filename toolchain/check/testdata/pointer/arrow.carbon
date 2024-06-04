// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/arrow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/arrow.carbon

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
// CHECK:STDOUT:   %Member: type = fn_type @Member [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Member = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type C [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type C, C* [template]
// CHECK:STDOUT:   %.4: type = struct_type {.field: C*} [template]
// CHECK:STDOUT:   %Foo: type = fn_type @Foo [template]
// CHECK:STDOUT:   %struct.2: Foo = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type {.field: C*} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %Foo.decl: Foo = fn_decl @Foo [template = constants.%struct.2] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc16: type = ptr_type C [template = constants.%.2]
// CHECK:STDOUT:     %ptr.loc16_8.1: C* = param ptr
// CHECK:STDOUT:     @Foo.%ptr: C* = bind_name ptr, %ptr.loc16_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Member.decl: Member = fn_decl @Member [template = constants.%struct.1] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %self.loc12_13.1: C = param self
// CHECK:STDOUT:     %self.loc12_13.2: C = bind_name self, %self.loc12_13.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc13_15: type = ptr_type C [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_12: <unbound element of class C> = field_decl field, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .Member = %Member.decl
// CHECK:STDOUT:   .field = %.loc13_12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Member[@C.%self.loc12_13.2: C]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%ptr: C*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ptr.ref.loc17: C* = name_ref ptr, %ptr
// CHECK:STDOUT:   %.loc17_4.1: ref C = deref %ptr.ref.loc17
// CHECK:STDOUT:   %Member.ref.loc17: Member = name_ref Member, @C.%Member.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc17_9: <bound method> = bound_method %.loc17_4.1, %Member.ref.loc17
// CHECK:STDOUT:   %.loc17_4.2: C = bind_value %.loc17_4.1
// CHECK:STDOUT:   %Member.call.loc17: init () = call %.loc17_9(%.loc17_4.2)
// CHECK:STDOUT:   %ptr.ref.loc18: C* = name_ref ptr, %ptr
// CHECK:STDOUT:   %.loc18_6.1: ref C = deref %ptr.ref.loc18
// CHECK:STDOUT:   %Member.ref.loc18: Member = name_ref Member, @C.%Member.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc18_6.2: <bound method> = bound_method %.loc18_6.1, %Member.ref.loc18
// CHECK:STDOUT:   %.loc18_6.3: C = bind_value %.loc18_6.1
// CHECK:STDOUT:   %Member.call.loc18: init () = call %.loc18_6.2(%.loc18_6.3)
// CHECK:STDOUT:   %ptr.ref.loc20: C* = name_ref ptr, %ptr
// CHECK:STDOUT:   %.loc20_4: ref C = deref %ptr.ref.loc20
// CHECK:STDOUT:   %field.ref.loc20: <unbound element of class C> = name_ref field, @C.%.loc13_12 [template = @C.%.loc13_12]
// CHECK:STDOUT:   %.loc20_9: ref C* = class_element_access %.loc20_4, element0
// CHECK:STDOUT:   %ptr.ref.loc21: C* = name_ref ptr, %ptr
// CHECK:STDOUT:   %.loc21_6.1: ref C = deref %ptr.ref.loc21
// CHECK:STDOUT:   %field.ref.loc21: <unbound element of class C> = name_ref field, @C.%.loc13_12 [template = @C.%.loc13_12]
// CHECK:STDOUT:   %.loc21_6.2: ref C* = class_element_access %.loc21_6.1, element0
// CHECK:STDOUT:   %ptr.ref.loc23: C* = name_ref ptr, %ptr
// CHECK:STDOUT:   %.loc23_6.1: ref C = deref %ptr.ref.loc23
// CHECK:STDOUT:   %field.ref.loc23_6: <unbound element of class C> = name_ref field, @C.%.loc13_12 [template = @C.%.loc13_12]
// CHECK:STDOUT:   %.loc23_6.2: ref C* = class_element_access %.loc23_6.1, element0
// CHECK:STDOUT:   %.loc23_6.3: C* = bind_value %.loc23_6.2
// CHECK:STDOUT:   %.loc23_13.1: ref C = deref %.loc23_6.3
// CHECK:STDOUT:   %field.ref.loc23_13: <unbound element of class C> = name_ref field, @C.%.loc13_12 [template = @C.%.loc13_12]
// CHECK:STDOUT:   %.loc23_13.2: ref C* = class_element_access %.loc23_13.1, element0
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
