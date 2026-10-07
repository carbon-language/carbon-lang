// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/destroy_format.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/destroy_format.carbon

//@dump-sem-ir-begin
class TrivialDestroy {}

class NonTrivialDestroy {
  impl as Core.Destroy {
    fn Op(unused self) {}
  }
}

class NestedNonTrivialDestroy {
  var x: NonTrivialDestroy;
}

fn CheckIsTrivial[T: Core.TrivialDestroy](ref _: T) {}

fn G() {
  var trivial_destroy: TrivialDestroy = {};
  CheckIsTrivial(ref trivial_destroy);

  var non_trivial_destroy: NonTrivialDestroy = {};
  CheckIsTrivial(ref non_trivial_destroy);

  var nested_non_trivial_destroy: NestedNonTrivialDestroy = {.x = {}};
  CheckIsTrivial(ref nested_non_trivial_destroy);
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- destroy_format.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %TrivialDestroy: type = class_type @TrivialDestroy.loc14 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %NonTrivialDestroy: type = class_type @NonTrivialDestroy [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d3e: type = fn_type @Destroy.WithSelf.Op.1, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.42b: %Destroy.WithSelf.Op.type.d3e = struct_value () [symbolic]
// CHECK:STDOUT:   %.127: <witness> = impl_self_witness %NonTrivialDestroy, @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @NonTrivialDestroy.as.Destroy.impl.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.f6b: type = pattern_type %NonTrivialDestroy [concrete]
// CHECK:STDOUT:   %self.param_patt.231: %pattern_type.f6b = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.d90: %pattern_type.f6b = wrapper_binding_pattern self, %self.param_patt.231 [concrete]
// CHECK:STDOUT:   %NonTrivialDestroy.as.Destroy.impl.Op.type.987e0a.1: type = fn_type @NonTrivialDestroy.as.Destroy.impl.Op.loc18_24.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonTrivialDestroy.as.Destroy.impl.Op.4454e9.1: %NonTrivialDestroy.as.Destroy.impl.Op.type.987e0a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.1cf: %Destroy.type = facet_value %NonTrivialDestroy, (%Destroy.impl_witness) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.342: type = fn_type @Destroy.WithSelf.SubobjectDestroy.1, @Destroy.WithSelf(%Destroy.facet.1cf) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.1fb: %Destroy.WithSelf.SubobjectDestroy.type.342 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.7b2: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.1cf) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.0dc: %Destroy.WithSelf.SelfDestruct.type.7b2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.3ca: %pattern_type.f6b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.41b: %pattern_type.f6b = wrapper_binding_pattern self, %self.param_patt.3ca [concrete]
// CHECK:STDOUT:   %NonTrivialDestroy.as.Destroy.impl.Op.type.987e0a.2: type = fn_type @NonTrivialDestroy.as.Destroy.impl.Op.loc18_24.2 [concrete]
// CHECK:STDOUT:   %NonTrivialDestroy.as.Destroy.impl.Op.4454e9.2: %NonTrivialDestroy.as.Destroy.impl.Op.type.987e0a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NestedNonTrivialDestroy: type = class_type @NestedNonTrivialDestroy [concrete]
// CHECK:STDOUT:   %NestedNonTrivialDestroy.elem: type = unbound_element_type %NestedNonTrivialDestroy, %NonTrivialDestroy [concrete]
// CHECK:STDOUT:   %struct_type.x.4df: type = struct_type {.x: %NonTrivialDestroy} [concrete]
// CHECK:STDOUT:   %complete_type.f9c: <witness> = complete_type_witness %struct_type.x.4df [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %TrivialDestroy.type: type = facet_type <@TrivialDestroy.1> [concrete]
// CHECK:STDOUT:   %pattern_type.1d8: type = pattern_type %TrivialDestroy.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.1d8 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %TrivialDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.2ae: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %_.param_patt.902: %pattern_type.2ae = ref_param_pattern [symbolic]
// CHECK:STDOUT:   %_.patt.e9d: %pattern_type.2ae = wrapper_binding_pattern _, %_.param_patt.902 [symbolic]
// CHECK:STDOUT:   %CheckIsTrivial.type: type = fn_type @CheckIsTrivial [concrete]
// CHECK:STDOUT:   %CheckIsTrivial: %CheckIsTrivial.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.72a: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.ef6: type = pattern_type %TrivialDestroy [concrete]
// CHECK:STDOUT:   %trivial_destroy.patt: %pattern_type.ef6 = ref_binding_pattern trivial_destroy [concrete]
// CHECK:STDOUT:   %trivial_destroy.var_patt: %pattern_type.ef6 = var_pattern %trivial_destroy.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestroy.val: %TrivialDestroy = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc30_37.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc30_37.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.dca: %pattern_type.ef6 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.74a: %pattern_type.ef6 = wrapper_binding_pattern self, %self.param_patt.dca [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc30_37.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc30_37.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc30_37.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.41a: <witness> = custom_witness (), @TrivialDestroy.1 [concrete]
// CHECK:STDOUT:   %TrivialDestroy.facet.236: %TrivialDestroy.type = facet_value %TrivialDestroy, (%custom_witness.41a) [concrete]
// CHECK:STDOUT:   %_.param_patt.997: %pattern_type.ef6 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %_.patt.672: %pattern_type.ef6 = wrapper_binding_pattern _, %_.param_patt.997 [concrete]
// CHECK:STDOUT:   %CheckIsTrivial.specific_fn.42c: <specific function> = specific_function %CheckIsTrivial, @CheckIsTrivial(%TrivialDestroy.facet.236) [concrete]
// CHECK:STDOUT:   %non_trivial_destroy.patt: %pattern_type.f6b = ref_binding_pattern non_trivial_destroy [concrete]
// CHECK:STDOUT:   %non_trivial_destroy.var_patt: %pattern_type.f6b = var_pattern %non_trivial_destroy.patt [concrete]
// CHECK:STDOUT:   %NonTrivialDestroy.val: %NonTrivialDestroy = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc33 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc33 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc33 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestroy.facet.453: %TrivialDestroy.type = facet_value %NonTrivialDestroy, (%custom_witness.41a) [concrete]
// CHECK:STDOUT:   %_.param_patt.4f1: %pattern_type.f6b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %_.patt.829: %pattern_type.f6b = wrapper_binding_pattern _, %_.param_patt.4f1 [concrete]
// CHECK:STDOUT:   %CheckIsTrivial.specific_fn.f23: <specific function> = specific_function %CheckIsTrivial, @CheckIsTrivial(%TrivialDestroy.facet.453) [concrete]
// CHECK:STDOUT:   %pattern_type.b42: type = pattern_type %NestedNonTrivialDestroy [concrete]
// CHECK:STDOUT:   %nested_non_trivial_destroy.patt: %pattern_type.b42 = ref_binding_pattern nested_non_trivial_destroy [concrete]
// CHECK:STDOUT:   %nested_non_trivial_destroy.var_patt: %pattern_type.b42 = var_pattern %nested_non_trivial_destroy.patt [concrete]
// CHECK:STDOUT:   %struct_type.x.3bf: type = struct_type {.x: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.x.3bf = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %NestedNonTrivialDestroy.val: %NestedNonTrivialDestroy = struct_value (%NonTrivialDestroy.val) [concrete]
// CHECK:STDOUT:   %pattern_type.623: type = pattern_type %struct_type.x.4df [concrete]
// CHECK:STDOUT:   %self.param_patt.420: %pattern_type.623 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0e0: %pattern_type.623 = wrapper_binding_pattern self, %self.param_patt.420 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc36_48.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc36_48.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.613: %pattern_type.b42 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.568: %pattern_type.b42 = wrapper_binding_pattern self, %self.param_patt.613 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc36_48.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc36_48.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc36_48.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestroy.facet.e11: %TrivialDestroy.type = facet_value %NestedNonTrivialDestroy, (%custom_witness.41a) [concrete]
// CHECK:STDOUT:   %_.param_patt.2f2: %pattern_type.b42 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %_.patt.96c: %pattern_type.b42 = wrapper_binding_pattern _, %_.param_patt.2f2 [concrete]
// CHECK:STDOUT:   %CheckIsTrivial.specific_fn.eb7: <specific function> = specific_function %CheckIsTrivial, @CheckIsTrivial(%TrivialDestroy.facet.e11) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .TrivialDestroy = %Core.TrivialDestroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.d3e = fn_decl @Destroy.WithSelf.Op.1 [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.Op (constants.%Destroy.WithSelf.Op.42b)] {} {}
// CHECK:STDOUT:   %Core.TrivialDestroy: type = import_ref Core//prelude/parts/destroy, TrivialDestroy, loaded [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc30_37.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc30_37.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc30_37.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ef6 = ref_param_pattern [concrete = constants.%self.param_patt.dca]
// CHECK:STDOUT:     %self.patt: %pattern_type.ef6 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.74a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %TrivialDestroy = ref_param call_param0
// CHECK:STDOUT:     %self: ref %TrivialDestroy = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc30_37.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ef6 = ref_param_pattern [concrete = constants.%self.param_patt.dca]
// CHECK:STDOUT:     %self.patt: %pattern_type.ef6 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.74a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %TrivialDestroy = ref_param call_param0
// CHECK:STDOUT:     %self: ref %TrivialDestroy = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc33 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f6b = ref_param_pattern [concrete = constants.%self.param_patt.3ca]
// CHECK:STDOUT:     %self.patt: %pattern_type.f6b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.41b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %NonTrivialDestroy = ref_param call_param0
// CHECK:STDOUT:     %self: ref %NonTrivialDestroy = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc33 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f6b = ref_param_pattern [concrete = constants.%self.param_patt.3ca]
// CHECK:STDOUT:     %self.patt: %pattern_type.f6b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.41b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %NonTrivialDestroy = ref_param call_param0
// CHECK:STDOUT:     %self: ref %NonTrivialDestroy = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc36_48.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.623 = ref_param_pattern [concrete = constants.%self.param_patt.420]
// CHECK:STDOUT:     %self.patt: %pattern_type.623 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0e0]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.x.4df = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.x.4df = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc36_48.1 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.623 = ref_param_pattern [concrete = constants.%self.param_patt.420]
// CHECK:STDOUT:     %self.patt: %pattern_type.623 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0e0]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.x.4df = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.x.4df = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc36_48.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b42 = ref_param_pattern [concrete = constants.%self.param_patt.613]
// CHECK:STDOUT:     %self.patt: %pattern_type.b42 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.568]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %NestedNonTrivialDestroy = ref_param call_param0
// CHECK:STDOUT:     %self: ref %NestedNonTrivialDestroy = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc36_48.2 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b42 = ref_param_pattern [concrete = constants.%self.param_patt.613]
// CHECK:STDOUT:     %self.patt: %pattern_type.b42 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.568]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %NestedNonTrivialDestroy = ref_param call_param0
// CHECK:STDOUT:     %self: ref %NestedNonTrivialDestroy = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %TrivialDestroy.decl: type = class_decl @TrivialDestroy.loc14 [concrete = constants.%TrivialDestroy] {} {}
// CHECK:STDOUT:   %NonTrivialDestroy.decl: type = class_decl @NonTrivialDestroy [concrete = constants.%NonTrivialDestroy] {} {}
// CHECK:STDOUT:   %NestedNonTrivialDestroy.decl: type = class_decl @NestedNonTrivialDestroy [concrete = constants.%NestedNonTrivialDestroy] {} {}
// CHECK:STDOUT:   %CheckIsTrivial.decl: %CheckIsTrivial.type = fn_decl @CheckIsTrivial [concrete = constants.%CheckIsTrivial] {
// CHECK:STDOUT:     %T.patt.loc26_20.1: %pattern_type.1d8 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc26_20.2 (constants.%T.patt)]
// CHECK:STDOUT:     %_.param_patt.loc26_48.1: @CheckIsTrivial.%pattern_type (%pattern_type.2ae) = ref_param_pattern [symbolic = %_.param_patt.loc26_48.2 (constants.%_.param_patt.902)]
// CHECK:STDOUT:     %_.patt.loc26_48.1: @CheckIsTrivial.%pattern_type (%pattern_type.2ae) = wrapper_binding_pattern _, %_.param_patt.loc26_48.1 [symbolic = %_.patt.loc26_48.2 (constants.%_.patt.e9d)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26_26: type = splice_block %TrivialDestroy.ref [concrete = constants.%TrivialDestroy.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %TrivialDestroy.ref: type = name_ref TrivialDestroy, imports.%Core.TrivialDestroy [concrete = constants.%TrivialDestroy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_20.2: %TrivialDestroy.type = symbolic_binding T, 0 [symbolic = %T.loc26_20.1 (constants.%T)]
// CHECK:STDOUT:     %_.param: ref @CheckIsTrivial.%T.as_type.loc26_50.1 (%T.as_type) = ref_param call_param0
// CHECK:STDOUT:     %.loc26_50.1: type = splice_block %.loc26_50.2 [symbolic = %T.as_type.loc26_50.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %TrivialDestroy.type = name_ref T, %T.loc26_20.2 [symbolic = %T.loc26_20.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc26_50.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc26_50.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc26_50.2: type = converted %T.ref, %T.as_type.loc26_50.2 [symbolic = %T.as_type.loc26_50.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_: ref @CheckIsTrivial.%T.as_type.loc26_50.1 (%T.as_type) = wrapper_binding _, %_.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @NonTrivialDestroy.as.Destroy.impl: %Self.ref as %Destroy.ref {
// CHECK:STDOUT:   %NonTrivialDestroy.as.Destroy.impl.Op.decl.loc18_24.1: %NonTrivialDestroy.as.Destroy.impl.Op.type.987e0a.1 = fn_decl @NonTrivialDestroy.as.Destroy.impl.Op.loc18_24.1 [concrete = constants.%NonTrivialDestroy.as.Destroy.impl.Op.4454e9.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f6b = value_param_pattern [concrete = constants.%self.param_patt.231]
// CHECK:STDOUT:     %self.patt: %pattern_type.f6b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d90]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %NonTrivialDestroy = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NonTrivialDestroy [concrete = constants.%NonTrivialDestroy]
// CHECK:STDOUT:     %self: %NonTrivialDestroy = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonTrivialDestroy.as.Destroy.impl.Op.decl.loc18_24.2: %NonTrivialDestroy.as.Destroy.impl.Op.type.987e0a.2 = fn_decl @NonTrivialDestroy.as.Destroy.impl.Op.loc18_24.2 [concrete = constants.%NonTrivialDestroy.as.Destroy.impl.Op.4454e9.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%NonTrivialDestroy.as.Destroy.impl.Op.decl.loc18_24.2, constants.%Destroy.WithSelf.SubobjectDestroy.1fb, constants.%Destroy.WithSelf.SelfDestruct.0dc), @NonTrivialDestroy.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %NonTrivialDestroy.as.Destroy.impl.Op.decl.loc18_24.1
// CHECK:STDOUT:   .SubobjectDestroy = <poisoned>
// CHECK:STDOUT:   .SelfDestruct = <poisoned>
// CHECK:STDOUT:   extend %Destroy.ref
// CHECK:STDOUT:   witness = %Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @TrivialDestroy.loc14 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%TrivialDestroy
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NonTrivialDestroy {
// CHECK:STDOUT:   impl_decl @NonTrivialDestroy.as.Destroy.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NonTrivialDestroy [concrete = constants.%NonTrivialDestroy]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17: <witness> = impl_self_witness @NonTrivialDestroy.as.Destroy.impl.%Self.ref, @Destroy [concrete = constants.%.127]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NonTrivialDestroy
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NestedNonTrivialDestroy {
// CHECK:STDOUT:   %field_decl: %NestedNonTrivialDestroy.elem = field_decl x, element0, %NonTrivialDestroy.ref in [concrete] {
// CHECK:STDOUT:     %NonTrivialDestroy.ref: type = name_ref NonTrivialDestroy, file.%NonTrivialDestroy.decl [concrete = constants.%NonTrivialDestroy]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x.4df [concrete = constants.%complete_type.f9c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NestedNonTrivialDestroy
// CHECK:STDOUT:   .NonTrivialDestroy = <poisoned>
// CHECK:STDOUT:   .x = %field_decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonTrivialDestroy.as.Destroy.impl.Op.loc18_24.1(%self.param: %NonTrivialDestroy) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonTrivialDestroy.as.Destroy.impl.Op.loc18_24.2(%self.param: ref %NonTrivialDestroy) [thunk @NonTrivialDestroy.as.Destroy.impl.%NonTrivialDestroy.as.Destroy.impl.Op.decl.loc18_24.1 for imports.%Destroy.WithSelf.Op.decl, @Destroy.WithSelf.Op.1(constants.%Destroy.facet.1cf)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Op.ref: %NonTrivialDestroy.as.Destroy.impl.Op.type.987e0a.1 = name_ref Op, @NonTrivialDestroy.as.Destroy.impl.%NonTrivialDestroy.as.Destroy.impl.Op.decl.loc18_24.1 [concrete = constants.%NonTrivialDestroy.as.Destroy.impl.Op.4454e9.1]
// CHECK:STDOUT:   %NonTrivialDestroy.as.Destroy.impl.Op.bound: <bound method> = bound_method %self.param, %Op.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %NonTrivialDestroy.as.Destroy.impl.Op.call: init %empty_tuple.type = call %NonTrivialDestroy.as.Destroy.impl.Op.bound(%.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CheckIsTrivial(%T.loc26_20.2: %TrivialDestroy.type) {
// CHECK:STDOUT:   %T.patt.loc26_20.2: %pattern_type.1d8 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc26_20.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc26_20.1: %TrivialDestroy.type = symbolic_binding T, 0 [symbolic = %T.loc26_20.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc26_50.1: type = facet_access_type %T.loc26_20.1 [symbolic = %T.as_type.loc26_50.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc26_50.1 [symbolic = %pattern_type (constants.%pattern_type.2ae)]
// CHECK:STDOUT:   %_.param_patt.loc26_48.2: @CheckIsTrivial.%pattern_type (%pattern_type.2ae) = ref_param_pattern [symbolic = %_.param_patt.loc26_48.2 (constants.%_.param_patt.902)]
// CHECK:STDOUT:   %_.patt.loc26_48.2: @CheckIsTrivial.%pattern_type (%pattern_type.2ae) = wrapper_binding_pattern _, %_.param_patt.loc26_48.2 [symbolic = %_.patt.loc26_48.2 (constants.%_.patt.e9d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc26_50.1 [symbolic = %require_complete (constants.%require_complete.72a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%_.param: ref @CheckIsTrivial.%T.as_type.loc26_50.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %trivial_destroy.var: ref %TrivialDestroy = var_storage %trivial_destroy.var_patt
// CHECK:STDOUT:   %.loc29_42.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc29_42.2: init %TrivialDestroy to %trivial_destroy.var = class_init () [concrete = constants.%TrivialDestroy.val]
// CHECK:STDOUT:   %.loc29_3: init %TrivialDestroy = converted %.loc29_42.1, %.loc29_42.2 [concrete = constants.%TrivialDestroy.val]
// CHECK:STDOUT:   assign %trivial_destroy.var, %.loc29_3
// CHECK:STDOUT:   %TrivialDestroy.ref: type = name_ref TrivialDestroy, file.%TrivialDestroy.decl [concrete = constants.%TrivialDestroy]
// CHECK:STDOUT:   %trivial_destroy: ref %TrivialDestroy = wrapper_binding trivial_destroy, %trivial_destroy.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %trivial_destroy.patt: %pattern_type.ef6 = ref_binding_pattern trivial_destroy [concrete = constants.%trivial_destroy.patt]
// CHECK:STDOUT:     %trivial_destroy.var_patt: %pattern_type.ef6 = var_pattern %trivial_destroy.patt [concrete = constants.%trivial_destroy.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CheckIsTrivial.ref.loc30: %CheckIsTrivial.type = name_ref CheckIsTrivial, file.%CheckIsTrivial.decl [concrete = constants.%CheckIsTrivial]
// CHECK:STDOUT:   %trivial_destroy.ref: ref %TrivialDestroy = name_ref trivial_destroy, %trivial_destroy
// CHECK:STDOUT:   %.loc30_18: %TrivialDestroy = ref_tag %trivial_destroy.ref
// CHECK:STDOUT:   %TrivialDestroy.facet.loc30_37.1: %TrivialDestroy.type = facet_value constants.%TrivialDestroy, (constants.%custom_witness.41a) [concrete = constants.%TrivialDestroy.facet.236]
// CHECK:STDOUT:   %.loc30_37.1: %TrivialDestroy.type = converted constants.%TrivialDestroy, %TrivialDestroy.facet.loc30_37.1 [concrete = constants.%TrivialDestroy.facet.236]
// CHECK:STDOUT:   %TrivialDestroy.facet.loc30_37.2: %TrivialDestroy.type = facet_value constants.%TrivialDestroy, (constants.%custom_witness.41a) [concrete = constants.%TrivialDestroy.facet.236]
// CHECK:STDOUT:   %.loc30_37.2: %TrivialDestroy.type = converted constants.%TrivialDestroy, %TrivialDestroy.facet.loc30_37.2 [concrete = constants.%TrivialDestroy.facet.236]
// CHECK:STDOUT:   %CheckIsTrivial.specific_fn.loc30: <specific function> = specific_function %CheckIsTrivial.ref.loc30, @CheckIsTrivial(constants.%TrivialDestroy.facet.236) [concrete = constants.%CheckIsTrivial.specific_fn.42c]
// CHECK:STDOUT:   %CheckIsTrivial.call.loc30: init %empty_tuple.type = call %CheckIsTrivial.specific_fn.loc30(%trivial_destroy.ref)
// CHECK:STDOUT:   %non_trivial_destroy.var: ref %NonTrivialDestroy = var_storage %non_trivial_destroy.var_patt
// CHECK:STDOUT:   %.loc32_49.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc32_49.2: init %NonTrivialDestroy to %non_trivial_destroy.var = class_init () [concrete = constants.%NonTrivialDestroy.val]
// CHECK:STDOUT:   %.loc32_3: init %NonTrivialDestroy = converted %.loc32_49.1, %.loc32_49.2 [concrete = constants.%NonTrivialDestroy.val]
// CHECK:STDOUT:   assign %non_trivial_destroy.var, %.loc32_3
// CHECK:STDOUT:   %NonTrivialDestroy.ref: type = name_ref NonTrivialDestroy, file.%NonTrivialDestroy.decl [concrete = constants.%NonTrivialDestroy]
// CHECK:STDOUT:   %non_trivial_destroy: ref %NonTrivialDestroy = wrapper_binding non_trivial_destroy, %non_trivial_destroy.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %non_trivial_destroy.patt: %pattern_type.f6b = ref_binding_pattern non_trivial_destroy [concrete = constants.%non_trivial_destroy.patt]
// CHECK:STDOUT:     %non_trivial_destroy.var_patt: %pattern_type.f6b = var_pattern %non_trivial_destroy.patt [concrete = constants.%non_trivial_destroy.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CheckIsTrivial.ref.loc33: %CheckIsTrivial.type = name_ref CheckIsTrivial, file.%CheckIsTrivial.decl [concrete = constants.%CheckIsTrivial]
// CHECK:STDOUT:   %non_trivial_destroy.ref: ref %NonTrivialDestroy = name_ref non_trivial_destroy, %non_trivial_destroy
// CHECK:STDOUT:   %.loc33_18: %NonTrivialDestroy = ref_tag %non_trivial_destroy.ref
// CHECK:STDOUT:   %TrivialDestroy.facet.loc33_41.1: %TrivialDestroy.type = facet_value constants.%NonTrivialDestroy, (constants.%custom_witness.41a) [concrete = constants.%TrivialDestroy.facet.453]
// CHECK:STDOUT:   %.loc33_41.1: %TrivialDestroy.type = converted constants.%NonTrivialDestroy, %TrivialDestroy.facet.loc33_41.1 [concrete = constants.%TrivialDestroy.facet.453]
// CHECK:STDOUT:   %TrivialDestroy.facet.loc33_41.2: %TrivialDestroy.type = facet_value constants.%NonTrivialDestroy, (constants.%custom_witness.41a) [concrete = constants.%TrivialDestroy.facet.453]
// CHECK:STDOUT:   %.loc33_41.2: %TrivialDestroy.type = converted constants.%NonTrivialDestroy, %TrivialDestroy.facet.loc33_41.2 [concrete = constants.%TrivialDestroy.facet.453]
// CHECK:STDOUT:   %CheckIsTrivial.specific_fn.loc33: <specific function> = specific_function %CheckIsTrivial.ref.loc33, @CheckIsTrivial(constants.%TrivialDestroy.facet.453) [concrete = constants.%CheckIsTrivial.specific_fn.f23]
// CHECK:STDOUT:   %CheckIsTrivial.call.loc33: init %empty_tuple.type = call %CheckIsTrivial.specific_fn.loc33(%non_trivial_destroy.ref)
// CHECK:STDOUT:   %nested_non_trivial_destroy.var: ref %NestedNonTrivialDestroy = var_storage %nested_non_trivial_destroy.var_patt
// CHECK:STDOUT:   %.loc35_68.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc35_69.1: %struct_type.x.3bf = struct_literal (%.loc35_68.1) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc35_69.2: ref %NonTrivialDestroy = class_element_access %nested_non_trivial_destroy.var, element0
// CHECK:STDOUT:   %.loc35_68.2: init %NonTrivialDestroy to %.loc35_69.2 = class_init () [concrete = constants.%NonTrivialDestroy.val]
// CHECK:STDOUT:   %.loc35_69.3: init %NonTrivialDestroy = converted %.loc35_68.1, %.loc35_68.2 [concrete = constants.%NonTrivialDestroy.val]
// CHECK:STDOUT:   %.loc35_69.4: init %NestedNonTrivialDestroy to %nested_non_trivial_destroy.var = class_init (%.loc35_69.3) [concrete = constants.%NestedNonTrivialDestroy.val]
// CHECK:STDOUT:   %.loc35_3: init %NestedNonTrivialDestroy = converted %.loc35_69.1, %.loc35_69.4 [concrete = constants.%NestedNonTrivialDestroy.val]
// CHECK:STDOUT:   assign %nested_non_trivial_destroy.var, %.loc35_3
// CHECK:STDOUT:   %NestedNonTrivialDestroy.ref: type = name_ref NestedNonTrivialDestroy, file.%NestedNonTrivialDestroy.decl [concrete = constants.%NestedNonTrivialDestroy]
// CHECK:STDOUT:   %nested_non_trivial_destroy: ref %NestedNonTrivialDestroy = wrapper_binding nested_non_trivial_destroy, %nested_non_trivial_destroy.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %nested_non_trivial_destroy.patt: %pattern_type.b42 = ref_binding_pattern nested_non_trivial_destroy [concrete = constants.%nested_non_trivial_destroy.patt]
// CHECK:STDOUT:     %nested_non_trivial_destroy.var_patt: %pattern_type.b42 = var_pattern %nested_non_trivial_destroy.patt [concrete = constants.%nested_non_trivial_destroy.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CheckIsTrivial.ref.loc36: %CheckIsTrivial.type = name_ref CheckIsTrivial, file.%CheckIsTrivial.decl [concrete = constants.%CheckIsTrivial]
// CHECK:STDOUT:   %nested_non_trivial_destroy.ref: ref %NestedNonTrivialDestroy = name_ref nested_non_trivial_destroy, %nested_non_trivial_destroy
// CHECK:STDOUT:   %.loc36_18: %NestedNonTrivialDestroy = ref_tag %nested_non_trivial_destroy.ref
// CHECK:STDOUT:   %TrivialDestroy.facet.loc36_48.1: %TrivialDestroy.type = facet_value constants.%NestedNonTrivialDestroy, (constants.%custom_witness.41a) [concrete = constants.%TrivialDestroy.facet.e11]
// CHECK:STDOUT:   %.loc36_48.1: %TrivialDestroy.type = converted constants.%NestedNonTrivialDestroy, %TrivialDestroy.facet.loc36_48.1 [concrete = constants.%TrivialDestroy.facet.e11]
// CHECK:STDOUT:   %TrivialDestroy.facet.loc36_48.2: %TrivialDestroy.type = facet_value constants.%NestedNonTrivialDestroy, (constants.%custom_witness.41a) [concrete = constants.%TrivialDestroy.facet.e11]
// CHECK:STDOUT:   %.loc36_48.2: %TrivialDestroy.type = converted constants.%NestedNonTrivialDestroy, %TrivialDestroy.facet.loc36_48.2 [concrete = constants.%TrivialDestroy.facet.e11]
// CHECK:STDOUT:   %CheckIsTrivial.specific_fn.loc36: <specific function> = specific_function %CheckIsTrivial.ref.loc36, @CheckIsTrivial(constants.%TrivialDestroy.facet.e11) [concrete = constants.%CheckIsTrivial.specific_fn.eb7]
// CHECK:STDOUT:   %CheckIsTrivial.call.loc36: init %empty_tuple.type = call %CheckIsTrivial.specific_fn.loc36(%nested_non_trivial_destroy.ref)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc35: <bound method> = bound_method %nested_non_trivial_destroy.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc35: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc35(%nested_non_trivial_destroy.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc32: <bound method> = bound_method %non_trivial_destroy.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc32: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc32(%non_trivial_destroy.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc29: <bound method> = bound_method %trivial_destroy.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc29: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc29(%trivial_destroy.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc30_37.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc30_37.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc30_37.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc30_37.2(%self.param: ref %TrivialDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc30_37.2(%self.param: ref %TrivialDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc30_37.2(%self.param: ref %TrivialDestroy) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc33(%self.param: ref %NonTrivialDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc33(%self.param: ref %NonTrivialDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc33(%self.param: ref %NonTrivialDestroy) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc36_48.1(%self.param: ref %struct_type.x.4df) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc36_48.1(%self.param: ref %struct_type.x.4df) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc36_48.1(%self.param: ref %struct_type.x.4df) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc36_48.2(%self.param: ref %NestedNonTrivialDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc36_48.2(%self.param: ref %NestedNonTrivialDestroy) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc36_48.2(%self.param: ref %NestedNonTrivialDestroy) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CheckIsTrivial(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc26_20.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc26_20.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc26_50.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2ae
// CHECK:STDOUT:   %_.param_patt.loc26_48.2 => constants.%_.param_patt.902
// CHECK:STDOUT:   %_.patt.loc26_48.2 => constants.%_.patt.e9d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CheckIsTrivial(constants.%TrivialDestroy.facet.236) {
// CHECK:STDOUT:   %T.patt.loc26_20.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc26_20.1 => constants.%TrivialDestroy.facet.236
// CHECK:STDOUT:   %T.as_type.loc26_50.1 => constants.%TrivialDestroy
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ef6
// CHECK:STDOUT:   %_.param_patt.loc26_48.2 => constants.%_.param_patt.997
// CHECK:STDOUT:   %_.patt.loc26_48.2 => constants.%_.patt.672
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CheckIsTrivial(constants.%TrivialDestroy.facet.453) {
// CHECK:STDOUT:   %T.patt.loc26_20.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc26_20.1 => constants.%TrivialDestroy.facet.453
// CHECK:STDOUT:   %T.as_type.loc26_50.1 => constants.%NonTrivialDestroy
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f6b
// CHECK:STDOUT:   %_.param_patt.loc26_48.2 => constants.%_.param_patt.4f1
// CHECK:STDOUT:   %_.patt.loc26_48.2 => constants.%_.patt.829
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CheckIsTrivial(constants.%TrivialDestroy.facet.e11) {
// CHECK:STDOUT:   %T.patt.loc26_20.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc26_20.1 => constants.%TrivialDestroy.facet.e11
// CHECK:STDOUT:   %T.as_type.loc26_50.1 => constants.%NestedNonTrivialDestroy
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b42
// CHECK:STDOUT:   %_.param_patt.loc26_48.2 => constants.%_.param_patt.2f2
// CHECK:STDOUT:   %_.patt.loc26_48.2 => constants.%_.patt.96c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f9c
// CHECK:STDOUT: }
// CHECK:STDOUT:
