// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.Member.type: type = fn_type @C.Member [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.Member: %C.Member.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %ptr.31e [concrete]
// CHECK:STDOUT:   %struct_type.field: type = struct_type {.field: %ptr.31e} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.field [concrete]
// CHECK:STDOUT:   %pattern_type.506: type = pattern_type %ptr.31e [concrete]
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
// CHECK:STDOUT:     %ptr.param_patt: %pattern_type.506 = value_param_pattern [concrete]
// CHECK:STDOUT:     %ptr.patt: %pattern_type.506 = at_binding_pattern ptr, %ptr.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ptr.param: %ptr.31e = value_param call_param0
// CHECK:STDOUT:     %.loc27: type = splice_block %ptr.loc27 [concrete = constants.%ptr.31e] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc27: type = ptr_type %C.ref [concrete = constants.%ptr.31e]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %ptr.loc24: %ptr.31e = value_binding ptr, %ptr.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %C.Member.decl: %C.Member.type = fn_decl @C.Member [concrete = constants.%C.Member] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7c7 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.7c7 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr: type = ptr_type %C.ref [concrete = constants.%ptr.31e]
// CHECK:STDOUT:   %.loc16: %C.elem = field_decl field, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.field [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .Member = %C.Member.decl
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .field = %.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.Member(%self.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%ptr.param: %ptr.31e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ptr.ref.loc32: %ptr.31e = name_ref ptr, %ptr.loc24
// CHECK:STDOUT:   %.loc31.1: ref %C = deref %ptr.ref.loc32
// CHECK:STDOUT:   %Member.ref.loc34: %C.Member.type = name_ref Member, @C.%C.Member.decl [concrete = constants.%C.Member]
// CHECK:STDOUT:   %C.Member.bound.loc34: <bound method> = bound_method %.loc31.1, %Member.ref.loc34
// CHECK:STDOUT:   %.loc31.2: %C = acquire_value %.loc31.1
// CHECK:STDOUT:   %C.Member.call.loc37: init %empty_tuple.type = call %C.Member.bound.loc34(%.loc31.2)
// CHECK:STDOUT:   %ptr.ref.loc39: %ptr.31e = name_ref ptr, %ptr.loc24
// CHECK:STDOUT:   %.loc40.1: ref %C = deref %ptr.ref.loc39
// CHECK:STDOUT:   %Member.ref.loc40: %C.Member.type = name_ref Member, @C.%C.Member.decl [concrete = constants.%C.Member]
// CHECK:STDOUT:   %C.Member.bound.loc40: <bound method> = bound_method %.loc40.1, %Member.ref.loc40
// CHECK:STDOUT:   %.loc40.2: %C = acquire_value %.loc40.1
// CHECK:STDOUT:   %C.Member.call.loc43: init %empty_tuple.type = call %C.Member.bound.loc40(%.loc40.2)
// CHECK:STDOUT:   %ptr.ref.loc47: %ptr.31e = name_ref ptr, %ptr.loc24
// CHECK:STDOUT:   %.loc46: ref %C = deref %ptr.ref.loc47
// CHECK:STDOUT:   %field.ref.loc49: %C.elem = name_ref field, @C.%.loc16 [concrete = @C.%.loc16]
// CHECK:STDOUT:   %.loc49: ref %ptr.31e = class_element_access %.loc46, element0
// CHECK:STDOUT:   %ptr.ref.loc52: %ptr.31e = name_ref ptr, %ptr.loc24
// CHECK:STDOUT:   %.loc53.1: ref %C = deref %ptr.ref.loc52
// CHECK:STDOUT:   %field.ref.loc53: %C.elem = name_ref field, @C.%.loc16 [concrete = @C.%.loc16]
// CHECK:STDOUT:   %.loc53.2: ref %ptr.31e = class_element_access %.loc53.1, element0
// CHECK:STDOUT:   %ptr.ref.loc56: %ptr.31e = name_ref ptr, %ptr.loc24
// CHECK:STDOUT:   %.loc57.1: ref %C = deref %ptr.ref.loc56
// CHECK:STDOUT:   %field.ref.loc57: %C.elem = name_ref field, @C.%.loc16 [concrete = @C.%.loc16]
// CHECK:STDOUT:   %.loc57.2: ref %ptr.31e = class_element_access %.loc57.1, element0
// CHECK:STDOUT:   %.loc57.3: %ptr.31e = acquire_value %.loc57.2
// CHECK:STDOUT:   %.loc59.1: ref %C = deref %.loc57.3
// CHECK:STDOUT:   %field.ref.loc59: %C.elem = name_ref field, @C.%.loc16 [concrete = @C.%.loc16]
// CHECK:STDOUT:   %.loc59.2: ref %ptr.31e = class_element_access %.loc59.1, element0
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
