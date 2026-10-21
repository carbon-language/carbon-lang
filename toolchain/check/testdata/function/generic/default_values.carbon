// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/default_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/default_values.carbon

// --- generic_fn_specific_param.carbon
library "[[@TEST_NAME]]";

class D {}
impl D as Core.Copy {
  fn Op(unused self: D) -> D { return {} as D; }
}

class E {}
impl E as Core.Copy {
  fn Op(unused self: E) -> E { return {} as E; }
}

class C(T: Core.Copy & Core.Destroy) {
  var t: T;
  var d: D;
  impl as Core.Copy {
    fn Op(self: Self) -> Self {
      let c: Self = {.t = self.t, .d = self.d};
      return c;
    }
  }
}

fn Make(generic T: Core.Copy & Core.Destroy, t: T, d: D = {} as D) -> C(T) {
  let c: C(T) = {.t = t, .d = d};
  return c;
}

fn G() {
  let _: C(E) = Make(E, {} as E, {} as D);
  //@dump-sem-ir-begin
  let _: C(E) = Make(E, {} as E);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- generic_fn_specific_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.e92: <witness> = impl_witness @E.as.Copy.impl.%Copy.impl_witness_table [concrete]
// CHECK:STDOUT:   %E.val: %E = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Copy & @Destroy> [concrete]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc18_46.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc30 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc30 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc30 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.3: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.3, %Destroy.WithSelf.SubobjectDestroy.d01daf.3, %Destroy.WithSelf.SelfDestruct.db3fdb.3), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %E, (%Copy.impl_witness.e92, %custom_witness.f8f19d.3) [concrete]
// CHECK:STDOUT:   %C.14c: type = class_type @C, @C(%facet_value) [concrete]
// CHECK:STDOUT:   %C.elem.54c: type = unbound_element_type %C.14c, %E [concrete]
// CHECK:STDOUT:   %C.elem.4f9: type = unbound_element_type %C.14c, %D [concrete]
// CHECK:STDOUT:   %struct_type.t.d.d81: type = struct_type {.t: %E, .d: %D} [concrete]
// CHECK:STDOUT:   %pattern_type.2cb: type = pattern_type %C.14c [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.2cb = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make, @Make(%facet_value) [concrete]
// CHECK:STDOUT:   %.b7b: ref %E = temporary invalid, %E.val [concrete]
// CHECK:STDOUT:   %pattern_type.b46: type = pattern_type %struct_type.t.d.d81 [concrete]
// CHECK:STDOUT:   %self.param_patt.c2f: %pattern_type.b46 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.430: %pattern_type.b46 = wrapper_binding_pattern self, %self.param_patt.c2f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc32_32.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc32_32.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.049: %pattern_type.2cb = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.769: %pattern_type.2cb = wrapper_binding_pattern self, %self.param_patt.049 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc32_32.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc32_32.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc32_32.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.7e9: <bound method> = bound_method %.b7b, %Destroy.WithSelf.SelfDestruct.db3fdb.3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc32_32.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b46 = ref_param_pattern [concrete = constants.%self.param_patt.c2f]
// CHECK:STDOUT:     %self.patt: %pattern_type.b46 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.430]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.t.d.d81 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.t.d.d81 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc32_32.1 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b46 = ref_param_pattern [concrete = constants.%self.param_patt.c2f]
// CHECK:STDOUT:     %self.patt: %pattern_type.b46 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.430]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.t.d.d81 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.t.d.d81 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc32_32.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2cb = ref_param_pattern [concrete = constants.%self.param_patt.049]
// CHECK:STDOUT:     %self.patt: %pattern_type.2cb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.769]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.14c = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.14c = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc32_32.2 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2cb = ref_param_pattern [concrete = constants.%self.param_patt.049]
// CHECK:STDOUT:     %self.patt: %pattern_type.2cb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.769]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.14c = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.14c = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Make.ref.loc32: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %E.ref.loc32_22: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %.loc32_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %E.ref.loc32_31: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %.loc32_26.2: ref %E = temporary_storage
// CHECK:STDOUT:   %.loc32_26.3: init %E to %.loc32_26.2 = class_init () [concrete = constants.%E.val]
// CHECK:STDOUT:   %.loc32_28.1: init %E = converted %.loc32_26.1, %.loc32_26.3 [concrete = constants.%E.val]
// CHECK:STDOUT:   %facet_value.loc32_32.1: %facet_type = facet_value %E.ref.loc32_22, (constants.%Copy.impl_witness.e92, constants.%custom_witness.f8f19d.3) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc32_32.1: %facet_type = converted %E.ref.loc32_22, %facet_value.loc32_32.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc32_32.2: %facet_type = facet_value constants.%E, (constants.%Copy.impl_witness.e92, constants.%custom_witness.f8f19d.3) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc32_32.2: %facet_type = converted constants.%E, %facet_value.loc32_32.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc32_32.3: %facet_type = facet_value constants.%E, (constants.%Copy.impl_witness.e92, constants.%custom_witness.f8f19d.3) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc32_32.3: %facet_type = converted constants.%E, %facet_value.loc32_32.3 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %Make.specific_fn.loc32: <specific function> = specific_function %Make.ref.loc32, @Make(constants.%facet_value) [concrete = constants.%Make.specific_fn]
// CHECK:STDOUT:   %.loc32_32.4: ref %C.14c = temporary_storage
// CHECK:STDOUT:   %.loc32_28.2: ref %E = temporary %.loc32_26.2, %.loc32_28.1 [concrete = constants.%.b7b]
// CHECK:STDOUT:   %.loc32_28.3: %E = acquire_value %.loc32_28.2 [concrete = constants.%E.val]
// CHECK:STDOUT:   %Make.call.loc32: init %C.14c to %.loc32_32.4 = call %Make.specific_fn.loc32(%.loc32_28.3, @Make.%.loc24_62.3)
// CHECK:STDOUT:   %.loc32_32.5: ref %C.14c = temporary %.loc32_32.4, %Make.call.loc32
// CHECK:STDOUT:   %.loc32_32.6: %C.14c = acquire_value %.loc32_32.5
// CHECK:STDOUT:   %.loc32_13.1: type = splice_block %C.loc32 [concrete = constants.%C.14c] {
// CHECK:STDOUT:     %C.ref.loc32: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %E.ref.loc32_12: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %facet_value.loc32_13: %facet_type = facet_value %E.ref.loc32_12, (constants.%Copy.impl_witness.e92, constants.%custom_witness.f8f19d.3) [concrete = constants.%facet_value]
// CHECK:STDOUT:     %.loc32_13.2: %facet_type = converted %E.ref.loc32_12, %facet_value.loc32_13 [concrete = constants.%facet_value]
// CHECK:STDOUT:     %C.loc32: type = class_type @C, @C(constants.%facet_value) [concrete = constants.%C.14c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.loc32: %C.14c = wrapper_binding _, %.loc32_32.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt.loc32: %pattern_type.2cb = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc32: <bound method> = bound_method %.loc32_32.5, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc32_32: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc32(%.loc32_32.5)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc32_28: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound.7e9(constants.%.b7b)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc32_32.1(%self.param: ref %struct_type.t.d.d81) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc32_32.1: ref %D = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc32_32.1: <bound method> = bound_method %.loc32_32.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc32_32.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc32_32.1(%.loc32_32.1)
// CHECK:STDOUT:   %.loc32_32.2: ref %E = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc32_32.2: <bound method> = bound_method %.loc32_32.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc32_32.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc32_32.2(%.loc32_32.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc32_32.1(%self.param: ref %struct_type.t.d.d81) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc32_32.1(%self.param: ref %struct_type.t.d.d81) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc32_32.2(%self.param: ref %C.14c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %C.elem.4f9 = name_ref d, @C.%field_decl.loc15 [concrete = @C.%field_decl.loc15]
// CHECK:STDOUT:   %.loc32_32.1: ref %D = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc32_32.1: <bound method> = bound_method %.loc32_32.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc32_32.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc32_32.1(%.loc32_32.1)
// CHECK:STDOUT:   %t.ref: %C.elem.54c = name_ref t, @C.%field_decl.loc14 [concrete = @C.%field_decl.loc14]
// CHECK:STDOUT:   %.loc32_32.2: ref %E = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc32_32.2: <bound method> = bound_method %.loc32_32.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc32_32.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc32_32.2(%.loc32_32.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc32_32.2(%self.param: ref %C.14c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc32_32.2(%self.param: ref %C.14c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
