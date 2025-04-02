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
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Member.type: type = fn_type @Member [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Member: %Member.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %ptr.019 [concrete]
// CHECK:STDOUT:   %struct_type.field: type = struct_type {.field: %ptr.019} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.field [concrete]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {
// CHECK:STDOUT:     %ptr.patt: %ptr.019 = binding_pattern ptr
// CHECK:STDOUT:     %ptr.param_patt: %ptr.019 = value_param_pattern %ptr.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ptr.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc16: type = splice_block %ptr.loc16_14 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc16_14: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %ptr.loc16_8: %ptr.019 = bind_name ptr, %ptr.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Member.decl: %Member.type = fn_decl @Member [concrete = constants.%Member] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_12: %C.elem = field_decl field, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc13_3: %C.elem = var_pattern %.loc13_12
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %C.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.field [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .Member = %Member.decl
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .field = %.loc13_12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Member[%self.param_patt: %C]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%ptr.param_patt: %ptr.019) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ptr.ref.loc17: %ptr.019 = name_ref ptr, %ptr.loc16_8
// CHECK:STDOUT:   %.loc17_4.1: ref %C = deref %ptr.ref.loc17
// CHECK:STDOUT:   %Member.ref.loc17: %Member.type = name_ref Member, @C.%Member.decl [concrete = constants.%Member]
// CHECK:STDOUT:   %Member.bound.loc17: <bound method> = bound_method %.loc17_4.1, %Member.ref.loc17
// CHECK:STDOUT:   %.loc17_4.2: %C = bind_value %.loc17_4.1
// CHECK:STDOUT:   %Member.call.loc17: init %empty_tuple.type = call %Member.bound.loc17(%.loc17_4.2)
// CHECK:STDOUT:   %ptr.ref.loc18: %ptr.019 = name_ref ptr, %ptr.loc16_8
// CHECK:STDOUT:   %.loc18_6.1: ref %C = deref %ptr.ref.loc18
// CHECK:STDOUT:   %Member.ref.loc18: %Member.type = name_ref Member, @C.%Member.decl [concrete = constants.%Member]
// CHECK:STDOUT:   %Member.bound.loc18: <bound method> = bound_method %.loc18_6.1, %Member.ref.loc18
// CHECK:STDOUT:   %.loc18_6.2: %C = bind_value %.loc18_6.1
// CHECK:STDOUT:   %Member.call.loc18: init %empty_tuple.type = call %Member.bound.loc18(%.loc18_6.2)
// CHECK:STDOUT:   %ptr.ref.loc20: %ptr.019 = name_ref ptr, %ptr.loc16_8
// CHECK:STDOUT:   %.loc20_4: ref %C = deref %ptr.ref.loc20
// CHECK:STDOUT:   %field.ref.loc20: %C.elem = name_ref field, @C.%.loc13_12 [concrete = @C.%.loc13_12]
// CHECK:STDOUT:   %.loc20_9: ref %ptr.019 = class_element_access %.loc20_4, element0
// CHECK:STDOUT:   %ptr.ref.loc21: %ptr.019 = name_ref ptr, %ptr.loc16_8
// CHECK:STDOUT:   %.loc21_6.1: ref %C = deref %ptr.ref.loc21
// CHECK:STDOUT:   %field.ref.loc21: %C.elem = name_ref field, @C.%.loc13_12 [concrete = @C.%.loc13_12]
// CHECK:STDOUT:   %.loc21_6.2: ref %ptr.019 = class_element_access %.loc21_6.1, element0
// CHECK:STDOUT:   %ptr.ref.loc23: %ptr.019 = name_ref ptr, %ptr.loc16_8
// CHECK:STDOUT:   %.loc23_6.1: ref %C = deref %ptr.ref.loc23
// CHECK:STDOUT:   %field.ref.loc23_6: %C.elem = name_ref field, @C.%.loc13_12 [concrete = @C.%.loc13_12]
// CHECK:STDOUT:   %.loc23_6.2: ref %ptr.019 = class_element_access %.loc23_6.1, element0
// CHECK:STDOUT:   %.loc23_6.3: %ptr.019 = bind_value %.loc23_6.2
// CHECK:STDOUT:   %.loc23_13.1: ref %C = deref %.loc23_6.3
// CHECK:STDOUT:   %field.ref.loc23_13: %C.elem = name_ref field, @C.%.loc13_12 [concrete = @C.%.loc13_12]
// CHECK:STDOUT:   %.loc23_13.2: ref %ptr.019 = class_element_access %.loc23_13.1, element0
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
