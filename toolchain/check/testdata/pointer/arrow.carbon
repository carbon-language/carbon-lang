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
// CHECK:STDOUT:   %Member.type: type = fn_type @Member [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Member: %Member.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = ptr_type %C [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %C, %.1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.field: %.1} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {
// CHECK:STDOUT:     %ptr.patt: %.1 = binding_pattern ptr
// CHECK:STDOUT:     %ptr.param_patt: %.1 = value_param_pattern %ptr.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc16: type = ptr_type %C [template = constants.%.1]
// CHECK:STDOUT:     %ptr.param: %.1 = value_param runtime_param0
// CHECK:STDOUT:     %ptr: %.1 = bind_name ptr, %ptr.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Member.decl: %Member.type = fn_decl @Member [template = constants.%Member] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc13_15: type = ptr_type %C [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_12: %.2 = field_decl field, element0 [template]
// CHECK:STDOUT:   %.loc14: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .Member = %Member.decl
// CHECK:STDOUT:   .field = %.loc13_12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Member[%self.param_patt: %C]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%ptr.param_patt: %.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ptr.ref.loc17: %.1 = name_ref ptr, %ptr
// CHECK:STDOUT:   %.loc17_4.1: ref %C = deref %ptr.ref.loc17
// CHECK:STDOUT:   %Member.ref.loc17: %Member.type = name_ref Member, @C.%Member.decl [template = constants.%Member]
// CHECK:STDOUT:   %.loc17_9: <bound method> = bound_method %.loc17_4.1, %Member.ref.loc17
// CHECK:STDOUT:   %.loc17_4.2: %C = bind_value %.loc17_4.1
// CHECK:STDOUT:   %Member.call.loc17: init %empty_tuple.type = call %.loc17_9(%.loc17_4.2)
// CHECK:STDOUT:   %ptr.ref.loc18: %.1 = name_ref ptr, %ptr
// CHECK:STDOUT:   %.loc18_6.1: ref %C = deref %ptr.ref.loc18
// CHECK:STDOUT:   %Member.ref.loc18: %Member.type = name_ref Member, @C.%Member.decl [template = constants.%Member]
// CHECK:STDOUT:   %.loc18_6.2: <bound method> = bound_method %.loc18_6.1, %Member.ref.loc18
// CHECK:STDOUT:   %.loc18_6.3: %C = bind_value %.loc18_6.1
// CHECK:STDOUT:   %Member.call.loc18: init %empty_tuple.type = call %.loc18_6.2(%.loc18_6.3)
// CHECK:STDOUT:   %ptr.ref.loc20: %.1 = name_ref ptr, %ptr
// CHECK:STDOUT:   %.loc20_4: ref %C = deref %ptr.ref.loc20
// CHECK:STDOUT:   %field.ref.loc20: %.2 = name_ref field, @C.%.loc13_12 [template = @C.%.loc13_12]
// CHECK:STDOUT:   %.loc20_9: ref %.1 = class_element_access %.loc20_4, element0
// CHECK:STDOUT:   %ptr.ref.loc21: %.1 = name_ref ptr, %ptr
// CHECK:STDOUT:   %.loc21_6.1: ref %C = deref %ptr.ref.loc21
// CHECK:STDOUT:   %field.ref.loc21: %.2 = name_ref field, @C.%.loc13_12 [template = @C.%.loc13_12]
// CHECK:STDOUT:   %.loc21_6.2: ref %.1 = class_element_access %.loc21_6.1, element0
// CHECK:STDOUT:   %ptr.ref.loc23: %.1 = name_ref ptr, %ptr
// CHECK:STDOUT:   %.loc23_6.1: ref %C = deref %ptr.ref.loc23
// CHECK:STDOUT:   %field.ref.loc23_6: %.2 = name_ref field, @C.%.loc13_12 [template = @C.%.loc13_12]
// CHECK:STDOUT:   %.loc23_6.2: ref %.1 = class_element_access %.loc23_6.1, element0
// CHECK:STDOUT:   %.loc23_6.3: %.1 = bind_value %.loc23_6.2
// CHECK:STDOUT:   %.loc23_13.1: ref %C = deref %.loc23_6.3
// CHECK:STDOUT:   %field.ref.loc23_13: %.2 = name_ref field, @C.%.loc13_12 [template = @C.%.loc13_12]
// CHECK:STDOUT:   %.loc23_13.2: ref %.1 = class_element_access %.loc23_13.1, element0
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
