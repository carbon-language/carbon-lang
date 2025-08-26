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
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.Member.type: type = fn_type @C.Member [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.Member: %C.Member.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %ptr.019 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.field: type = struct_type {.field: %ptr.019} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.field [concrete]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %ptr.patt: %pattern_type.44a = binding_pattern ptr [concrete]
// CHECK:STDOUT:     %ptr.param_patt: %pattern_type.44a = value_param_pattern %ptr.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %ptr.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc20: type = splice_block %ptr.loc20_14 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc20_14: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %ptr.loc20_8: %ptr.019 = bind_name ptr, %ptr.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc15: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %C.Member.decl: %C.Member.type = fn_decl @C.Member [concrete = constants.%C.Member] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:   %.loc17: %C.elem = field_decl field, element0 [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.field [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .Member = %C.Member.decl
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .field = %.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.Member(%self.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%ptr.param: %ptr.019) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ptr.ref.loc21: %ptr.019 = name_ref ptr, %ptr.loc20_8
// CHECK:STDOUT:   %.loc21_4.1: ref %C = deref %ptr.ref.loc21
// CHECK:STDOUT:   %Member.ref.loc21: %C.Member.type = name_ref Member, @C.%C.Member.decl [concrete = constants.%C.Member]
// CHECK:STDOUT:   %C.Member.bound.loc21: <bound method> = bound_method %.loc21_4.1, %Member.ref.loc21
// CHECK:STDOUT:   %.loc21_4.2: %C = bind_value %.loc21_4.1
// CHECK:STDOUT:   %C.Member.call.loc21: init %empty_tuple.type = call %C.Member.bound.loc21(%.loc21_4.2)
// CHECK:STDOUT:   %ptr.ref.loc22: %ptr.019 = name_ref ptr, %ptr.loc20_8
// CHECK:STDOUT:   %.loc22_6.1: ref %C = deref %ptr.ref.loc22
// CHECK:STDOUT:   %Member.ref.loc22: %C.Member.type = name_ref Member, @C.%C.Member.decl [concrete = constants.%C.Member]
// CHECK:STDOUT:   %C.Member.bound.loc22: <bound method> = bound_method %.loc22_6.1, %Member.ref.loc22
// CHECK:STDOUT:   %.loc22_6.2: %C = bind_value %.loc22_6.1
// CHECK:STDOUT:   %C.Member.call.loc22: init %empty_tuple.type = call %C.Member.bound.loc22(%.loc22_6.2)
// CHECK:STDOUT:   %ptr.ref.loc24: %ptr.019 = name_ref ptr, %ptr.loc20_8
// CHECK:STDOUT:   %.loc24_4: ref %C = deref %ptr.ref.loc24
// CHECK:STDOUT:   %field.ref.loc24: %C.elem = name_ref field, @C.%.loc17 [concrete = @C.%.loc17]
// CHECK:STDOUT:   %.loc24_9: ref %ptr.019 = class_element_access %.loc24_4, element0
// CHECK:STDOUT:   %ptr.ref.loc25: %ptr.019 = name_ref ptr, %ptr.loc20_8
// CHECK:STDOUT:   %.loc25_6.1: ref %C = deref %ptr.ref.loc25
// CHECK:STDOUT:   %field.ref.loc25: %C.elem = name_ref field, @C.%.loc17 [concrete = @C.%.loc17]
// CHECK:STDOUT:   %.loc25_6.2: ref %ptr.019 = class_element_access %.loc25_6.1, element0
// CHECK:STDOUT:   %ptr.ref.loc27: %ptr.019 = name_ref ptr, %ptr.loc20_8
// CHECK:STDOUT:   %.loc27_6.1: ref %C = deref %ptr.ref.loc27
// CHECK:STDOUT:   %field.ref.loc27_6: %C.elem = name_ref field, @C.%.loc17 [concrete = @C.%.loc17]
// CHECK:STDOUT:   %.loc27_6.2: ref %ptr.019 = class_element_access %.loc27_6.1, element0
// CHECK:STDOUT:   %.loc27_6.3: %ptr.019 = bind_value %.loc27_6.2
// CHECK:STDOUT:   %.loc27_13.1: ref %C = deref %.loc27_6.3
// CHECK:STDOUT:   %field.ref.loc27_13: %C.elem = name_ref field, @C.%.loc17 [concrete = @C.%.loc17]
// CHECK:STDOUT:   %.loc27_13.2: ref %ptr.019 = class_element_access %.loc27_13.1, element0
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
