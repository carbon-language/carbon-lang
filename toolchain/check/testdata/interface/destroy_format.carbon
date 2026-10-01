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
import Cpp inline '''c++
// struct NoDestroy {
//   ~NoDestroy() = delete;
// };
''';

// class TrivialDestroy {}

class NonTrivialDestroy {
}


  impl as Core.Destroy {
    fn Op(unused self) {}
  }

fn FormatIsNoDestroy[T: Core.DestroyFormat where .value = 0](ref _: T) {}
fn FormatIsTrivial[T: Core.DestroyFormat where .value = 1](ref _: T) {}
fn FormatIsNonTrivial[T: Core.DestroyFormat where .value = 2](ref _: T) {}

// var no_destroy: Cpp.NoDestroy = {};

fn G() {
  // FormatIsNoDestroy(ref no_destroy);

  // CHECK:STDERR: destroy_format.carbon:[[@LINE+4]]:24: error: name `TrivialDestroy` not found [NameNotFound]
  // CHECK:STDERR:   var trivial_destroy: TrivialDestroy = {};
  // CHECK:STDERR:                        ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  // var trivial_destroy: TrivialDestroy = {};
  // FormatIsTrivial(ref trivial_destroy);

  var non_trivial_destroy: NonTrivialDestroy = {};
  FormatIsNonTrivial(ref non_trivial_destroy);
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- destroy_format.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NonTrivialDestroy: type = class_type @NonTrivialDestroy [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d3e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.42b: %Destroy.WithSelf.Op.type.d3e = struct_value () [symbolic]
// CHECK:STDOUT:   %.127: <witness> = impl_self_witness %NonTrivialDestroy, @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @NonTrivialDestroy.as.Destroy.impl.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.f6b: type = pattern_type %NonTrivialDestroy [concrete]
// CHECK:STDOUT:   %self.param_patt.231: %pattern_type.f6b = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.d90: %pattern_type.f6b = wrapper_binding_pattern self, %self.param_patt.231 [concrete]
// CHECK:STDOUT:   %NonTrivialDestroy.as.Destroy.impl.Op.type.987e0a.1: type = fn_type @NonTrivialDestroy.as.Destroy.impl.Op.loc24_24.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NonTrivialDestroy.as.Destroy.impl.Op.4454e9.1: %NonTrivialDestroy.as.Destroy.impl.Op.type.987e0a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %NonTrivialDestroy, (%Destroy.impl_witness) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.342: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Destroy.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.1fb: %Destroy.WithSelf.SubobjectDestroy.type.342 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.7b2: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.0dc: %Destroy.WithSelf.SelfDestruct.type.7b2 = struct_value () [concrete]
// CHECK:STDOUT:   %NonTrivialDestroy.as.Destroy.impl.Op.type.987e0a.2: type = fn_type @NonTrivialDestroy.as.Destroy.impl.Op.loc24_24.2 [concrete]
// CHECK:STDOUT:   %NonTrivialDestroy.as.Destroy.impl.Op.4454e9.2: %NonTrivialDestroy.as.Destroy.impl.Op.type.987e0a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self.frozen.e58: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DestroyFormat.type: type = facet_type <@DestroyFormat> [concrete]
// CHECK:STDOUT:   %.Self.frozen.929: %DestroyFormat.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.929 [symbolic_self]
// CHECK:STDOUT:   %DestroyFormat.assoc_type: type = assoc_entity_type @DestroyFormat [concrete]
// CHECK:STDOUT:   %assoc0: %DestroyFormat.assoc_type = assoc_entity element0, imports.%Core.import_ref.250 [concrete]
// CHECK:STDOUT:   %DestroyFormat.lookup_impl_witness.c60: <witness> = lookup_impl_witness %.Self.frozen.929, @DestroyFormat [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.794: Core.IntLiteral = impl_witness_access %DestroyFormat.lookup_impl_witness.c60, element0 [symbolic_self]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %.Self: %DestroyFormat.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DestroyFormat.lookup_impl_witness.7a8: <witness> = lookup_impl_witness %.Self, @DestroyFormat [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.bfb: Core.IntLiteral = impl_witness_access %DestroyFormat.lookup_impl_witness.7a8, element0 [symbolic_self]
// CHECK:STDOUT:   %DestroyFormat_where.type.ad0: type = facet_type <@DestroyFormat where %impl.elem0.bfb = %int_0> [concrete]
// CHECK:STDOUT:   %pattern_type.474: type = pattern_type %DestroyFormat_where.type.ad0 [concrete]
// CHECK:STDOUT:   %T.patt.d12: %pattern_type.474 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.5df: %DestroyFormat_where.type.ad0 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.725: type = facet_access_type %T.5df [symbolic]
// CHECK:STDOUT:   %pattern_type.1fc: type = pattern_type %T.as_type.725 [symbolic]
// CHECK:STDOUT:   %_.param_patt.6e1: %pattern_type.1fc = ref_param_pattern [symbolic]
// CHECK:STDOUT:   %_.patt.d3b: %pattern_type.1fc = wrapper_binding_pattern _, %_.param_patt.6e1 [symbolic]
// CHECK:STDOUT:   %FormatIsNoDestroy.type: type = fn_type @FormatIsNoDestroy [concrete]
// CHECK:STDOUT:   %FormatIsNoDestroy: %FormatIsNoDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.260: <witness> = require_complete_type %T.as_type.725 [symbolic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %DestroyFormat_where.type.e28: type = facet_type <@DestroyFormat where %impl.elem0.bfb = %int_1> [concrete]
// CHECK:STDOUT:   %pattern_type.5d6: type = pattern_type %DestroyFormat_where.type.e28 [concrete]
// CHECK:STDOUT:   %T.patt.c88: %pattern_type.5d6 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.28c: %DestroyFormat_where.type.e28 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.789: type = facet_access_type %T.28c [symbolic]
// CHECK:STDOUT:   %pattern_type.4fb: type = pattern_type %T.as_type.789 [symbolic]
// CHECK:STDOUT:   %_.param_patt.88c: %pattern_type.4fb = ref_param_pattern [symbolic]
// CHECK:STDOUT:   %_.patt.1ca: %pattern_type.4fb = wrapper_binding_pattern _, %_.param_patt.88c [symbolic]
// CHECK:STDOUT:   %FormatIsTrivial.type: type = fn_type @FormatIsTrivial [concrete]
// CHECK:STDOUT:   %FormatIsTrivial: %FormatIsTrivial.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.b00: <witness> = require_complete_type %T.as_type.789 [symbolic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %DestroyFormat_where.type.0ba: type = facet_type <@DestroyFormat where %impl.elem0.bfb = %int_2> [concrete]
// CHECK:STDOUT:   %pattern_type.cbf: type = pattern_type %DestroyFormat_where.type.0ba [concrete]
// CHECK:STDOUT:   %T.patt.de2: %pattern_type.cbf = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.39d: %DestroyFormat_where.type.0ba = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.6d2: type = facet_access_type %T.39d [symbolic]
// CHECK:STDOUT:   %pattern_type.353: type = pattern_type %T.as_type.6d2 [symbolic]
// CHECK:STDOUT:   %_.param_patt.1bd: %pattern_type.353 = ref_param_pattern [symbolic]
// CHECK:STDOUT:   %_.patt.c81: %pattern_type.353 = wrapper_binding_pattern _, %_.param_patt.1bd [symbolic]
// CHECK:STDOUT:   %FormatIsNonTrivial.type: type = fn_type @FormatIsNonTrivial [concrete]
// CHECK:STDOUT:   %FormatIsNonTrivial: %FormatIsNonTrivial.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.fb5: <witness> = require_complete_type %T.as_type.6d2 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .DestroyFormat = %Core.DestroyFormat
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.d3e = fn_decl @Destroy.WithSelf.Op [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.Op (constants.%Destroy.WithSelf.Op.42b)] {} {}
// CHECK:STDOUT:   %Core.DestroyFormat: type = import_ref Core//prelude/parts/destroy, DestroyFormat, loaded [concrete = constants.%DestroyFormat.type]
// CHECK:STDOUT:   %Core.import_ref.63d: %DestroyFormat.assoc_type = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.import_ref.250: Core.IntLiteral = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [concrete = %value]
// CHECK:STDOUT:   %value: Core.IntLiteral = assoc_const_decl @value [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp inline
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonTrivialDestroy.decl: type = class_decl @NonTrivialDestroy [concrete = constants.%NonTrivialDestroy] {} {}
// CHECK:STDOUT:   %FormatIsNoDestroy.decl: %FormatIsNoDestroy.type = fn_decl @FormatIsNoDestroy [concrete = constants.%FormatIsNoDestroy] {
// CHECK:STDOUT:     %T.patt.loc28_23.1: %pattern_type.474 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc28_23.2 (constants.%T.patt.d12)]
// CHECK:STDOUT:     %_.param_patt.loc28_67.1: @FormatIsNoDestroy.%pattern_type (%pattern_type.1fc) = ref_param_pattern [symbolic = %_.param_patt.loc28_67.2 (constants.%_.param_patt.6e1)]
// CHECK:STDOUT:     %_.patt.loc28_67.1: @FormatIsNoDestroy.%pattern_type (%pattern_type.1fc) = wrapper_binding_pattern _, %_.param_patt.loc28_67.1 [symbolic = %_.patt.loc28_67.2 (constants.%_.patt.d3b)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc28_44.1: type = splice_block %.loc28_44.2 [concrete = constants.%DestroyFormat_where.type.ad0] {
// CHECK:STDOUT:       %.Self.frozen.loc28_23: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %DestroyFormat.ref: type = name_ref DestroyFormat, imports.%Core.DestroyFormat [concrete = constants.%DestroyFormat.type]
// CHECK:STDOUT:       %.Self.frozen.loc28_44: %DestroyFormat.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.929]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %DestroyFormat.ref [concrete]
// CHECK:STDOUT:       %.Self.ref: %DestroyFormat.type = name_ref .Self, %.Self.frozen.loc28_44 [symbolic_self = constants.%.Self.frozen.929]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc28_50: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %value.ref: %DestroyFormat.assoc_type = name_ref value, imports.%Core.import_ref.63d [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc28_50.1: Core.IntLiteral = impl_witness_access constants.%DestroyFormat.lookup_impl_witness.c60, element0 [symbolic_self = constants.%impl.elem0.794]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %rewrite.loc28_57.1 = requirement_rewrite %impl.elem0.loc28_50.1, %int_0 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc28_50.2: Core.IntLiteral = impl_witness_access constants.%DestroyFormat.lookup_impl_witness.7a8, element0 [symbolic_self = constants.%impl.elem0.bfb]
// CHECK:STDOUT:       %rewrite.loc28_57.2 = requirement_rewrite %impl.elem0.loc28_50.2, %int_0 [concrete]
// CHECK:STDOUT:       %.loc28_44.2: type = where_expr [concrete = constants.%DestroyFormat_where.type.ad0] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %DestroyFormat.ref [concrete]
// CHECK:STDOUT:         %rewrite.loc28_57.2 = requirement_rewrite %impl.elem0.loc28_50.2, %int_0 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc28_23.2: %DestroyFormat_where.type.ad0 = symbolic_binding T, 0 [symbolic = %T.loc28_23.1 (constants.%T.5df)]
// CHECK:STDOUT:     %_.param: ref @FormatIsNoDestroy.%T.as_type.loc28_69.1 (%T.as_type.725) = ref_param call_param0
// CHECK:STDOUT:     %.loc28_69.1: type = splice_block %.loc28_69.2 [symbolic = %T.as_type.loc28_69.1 (constants.%T.as_type.725)] {
// CHECK:STDOUT:       %T.ref: %DestroyFormat_where.type.ad0 = name_ref T, %T.loc28_23.2 [symbolic = %T.loc28_23.1 (constants.%T.5df)]
// CHECK:STDOUT:       %T.as_type.loc28_69.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc28_69.1 (constants.%T.as_type.725)]
// CHECK:STDOUT:       %.loc28_69.2: type = converted %T.ref, %T.as_type.loc28_69.2 [symbolic = %T.as_type.loc28_69.1 (constants.%T.as_type.725)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_: ref @FormatIsNoDestroy.%T.as_type.loc28_69.1 (%T.as_type.725) = wrapper_binding _, %_.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FormatIsTrivial.decl: %FormatIsTrivial.type = fn_decl @FormatIsTrivial [concrete = constants.%FormatIsTrivial] {
// CHECK:STDOUT:     %T.patt.loc29_21.1: %pattern_type.5d6 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc29_21.2 (constants.%T.patt.c88)]
// CHECK:STDOUT:     %_.param_patt.loc29_65.1: @FormatIsTrivial.%pattern_type (%pattern_type.4fb) = ref_param_pattern [symbolic = %_.param_patt.loc29_65.2 (constants.%_.param_patt.88c)]
// CHECK:STDOUT:     %_.patt.loc29_65.1: @FormatIsTrivial.%pattern_type (%pattern_type.4fb) = wrapper_binding_pattern _, %_.param_patt.loc29_65.1 [symbolic = %_.patt.loc29_65.2 (constants.%_.patt.1ca)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc29_42.1: type = splice_block %.loc29_42.2 [concrete = constants.%DestroyFormat_where.type.e28] {
// CHECK:STDOUT:       %.Self.frozen.loc29_21: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %DestroyFormat.ref: type = name_ref DestroyFormat, imports.%Core.DestroyFormat [concrete = constants.%DestroyFormat.type]
// CHECK:STDOUT:       %.Self.frozen.loc29_42: %DestroyFormat.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.929]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %DestroyFormat.ref [concrete]
// CHECK:STDOUT:       %.Self.ref: %DestroyFormat.type = name_ref .Self, %.Self.frozen.loc29_42 [symbolic_self = constants.%.Self.frozen.929]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc29_48: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %value.ref: %DestroyFormat.assoc_type = name_ref value, imports.%Core.import_ref.63d [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc29_48.1: Core.IntLiteral = impl_witness_access constants.%DestroyFormat.lookup_impl_witness.c60, element0 [symbolic_self = constants.%impl.elem0.794]
// CHECK:STDOUT:       %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:       %rewrite.loc29_55.1 = requirement_rewrite %impl.elem0.loc29_48.1, %int_1 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc29_48.2: Core.IntLiteral = impl_witness_access constants.%DestroyFormat.lookup_impl_witness.7a8, element0 [symbolic_self = constants.%impl.elem0.bfb]
// CHECK:STDOUT:       %rewrite.loc29_55.2 = requirement_rewrite %impl.elem0.loc29_48.2, %int_1 [concrete]
// CHECK:STDOUT:       %.loc29_42.2: type = where_expr [concrete = constants.%DestroyFormat_where.type.e28] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %DestroyFormat.ref [concrete]
// CHECK:STDOUT:         %rewrite.loc29_55.2 = requirement_rewrite %impl.elem0.loc29_48.2, %int_1 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc29_21.2: %DestroyFormat_where.type.e28 = symbolic_binding T, 0 [symbolic = %T.loc29_21.1 (constants.%T.28c)]
// CHECK:STDOUT:     %_.param: ref @FormatIsTrivial.%T.as_type.loc29_67.1 (%T.as_type.789) = ref_param call_param0
// CHECK:STDOUT:     %.loc29_67.1: type = splice_block %.loc29_67.2 [symbolic = %T.as_type.loc29_67.1 (constants.%T.as_type.789)] {
// CHECK:STDOUT:       %T.ref: %DestroyFormat_where.type.e28 = name_ref T, %T.loc29_21.2 [symbolic = %T.loc29_21.1 (constants.%T.28c)]
// CHECK:STDOUT:       %T.as_type.loc29_67.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc29_67.1 (constants.%T.as_type.789)]
// CHECK:STDOUT:       %.loc29_67.2: type = converted %T.ref, %T.as_type.loc29_67.2 [symbolic = %T.as_type.loc29_67.1 (constants.%T.as_type.789)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_: ref @FormatIsTrivial.%T.as_type.loc29_67.1 (%T.as_type.789) = wrapper_binding _, %_.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FormatIsNonTrivial.decl: %FormatIsNonTrivial.type = fn_decl @FormatIsNonTrivial [concrete = constants.%FormatIsNonTrivial] {
// CHECK:STDOUT:     %T.patt.loc30_24.1: %pattern_type.cbf = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc30_24.2 (constants.%T.patt.de2)]
// CHECK:STDOUT:     %_.param_patt.loc30_68.1: @FormatIsNonTrivial.%pattern_type (%pattern_type.353) = ref_param_pattern [symbolic = %_.param_patt.loc30_68.2 (constants.%_.param_patt.1bd)]
// CHECK:STDOUT:     %_.patt.loc30_68.1: @FormatIsNonTrivial.%pattern_type (%pattern_type.353) = wrapper_binding_pattern _, %_.param_patt.loc30_68.1 [symbolic = %_.patt.loc30_68.2 (constants.%_.patt.c81)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc30_45.1: type = splice_block %.loc30_45.2 [concrete = constants.%DestroyFormat_where.type.0ba] {
// CHECK:STDOUT:       %.Self.frozen.loc30_24: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %DestroyFormat.ref: type = name_ref DestroyFormat, imports.%Core.DestroyFormat [concrete = constants.%DestroyFormat.type]
// CHECK:STDOUT:       %.Self.frozen.loc30_45: %DestroyFormat.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.929]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %DestroyFormat.ref [concrete]
// CHECK:STDOUT:       %.Self.ref: %DestroyFormat.type = name_ref .Self, %.Self.frozen.loc30_45 [symbolic_self = constants.%.Self.frozen.929]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc30_51: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %value.ref: %DestroyFormat.assoc_type = name_ref value, imports.%Core.import_ref.63d [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc30_51.1: Core.IntLiteral = impl_witness_access constants.%DestroyFormat.lookup_impl_witness.c60, element0 [symbolic_self = constants.%impl.elem0.794]
// CHECK:STDOUT:       %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:       %rewrite.loc30_58.1 = requirement_rewrite %impl.elem0.loc30_51.1, %int_2 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc30_51.2: Core.IntLiteral = impl_witness_access constants.%DestroyFormat.lookup_impl_witness.7a8, element0 [symbolic_self = constants.%impl.elem0.bfb]
// CHECK:STDOUT:       %rewrite.loc30_58.2 = requirement_rewrite %impl.elem0.loc30_51.2, %int_2 [concrete]
// CHECK:STDOUT:       %.loc30_45.2: type = where_expr [concrete = constants.%DestroyFormat_where.type.0ba] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %DestroyFormat.ref [concrete]
// CHECK:STDOUT:         %rewrite.loc30_58.2 = requirement_rewrite %impl.elem0.loc30_51.2, %int_2 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc30_24.2: %DestroyFormat_where.type.0ba = symbolic_binding T, 0 [symbolic = %T.loc30_24.1 (constants.%T.39d)]
// CHECK:STDOUT:     %_.param: ref @FormatIsNonTrivial.%T.as_type.loc30_70.1 (%T.as_type.6d2) = ref_param call_param0
// CHECK:STDOUT:     %.loc30_70.1: type = splice_block %.loc30_70.2 [symbolic = %T.as_type.loc30_70.1 (constants.%T.as_type.6d2)] {
// CHECK:STDOUT:       %T.ref: %DestroyFormat_where.type.0ba = name_ref T, %T.loc30_24.2 [symbolic = %T.loc30_24.1 (constants.%T.39d)]
// CHECK:STDOUT:       %T.as_type.loc30_70.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc30_70.1 (constants.%T.as_type.6d2)]
// CHECK:STDOUT:       %.loc30_70.2: type = converted %T.ref, %T.as_type.loc30_70.2 [symbolic = %T.as_type.loc30_70.1 (constants.%T.as_type.6d2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_: ref @FormatIsNonTrivial.%T.as_type.loc30_70.1 (%T.as_type.6d2) = wrapper_binding _, %_.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @NonTrivialDestroy.as.Destroy.impl: %Self.ref as %Destroy.ref {
// CHECK:STDOUT:   %NonTrivialDestroy.as.Destroy.impl.Op.decl.loc24_24.1: %NonTrivialDestroy.as.Destroy.impl.Op.type.987e0a.1 = fn_decl @NonTrivialDestroy.as.Destroy.impl.Op.loc24_24.1 [concrete = constants.%NonTrivialDestroy.as.Destroy.impl.Op.4454e9.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f6b = value_param_pattern [concrete = constants.%self.param_patt.231]
// CHECK:STDOUT:     %self.patt: %pattern_type.f6b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d90]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %NonTrivialDestroy = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NonTrivialDestroy [concrete = constants.%NonTrivialDestroy]
// CHECK:STDOUT:     %self: %NonTrivialDestroy = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonTrivialDestroy.as.Destroy.impl.Op.decl.loc24_24.2: %NonTrivialDestroy.as.Destroy.impl.Op.type.987e0a.2 = fn_decl @NonTrivialDestroy.as.Destroy.impl.Op.loc24_24.2 [concrete = constants.%NonTrivialDestroy.as.Destroy.impl.Op.4454e9.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%NonTrivialDestroy.as.Destroy.impl.Op.decl.loc24_24.2, constants.%Destroy.WithSelf.SubobjectDestroy.1fb, constants.%Destroy.WithSelf.SelfDestruct.0dc), @NonTrivialDestroy.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %NonTrivialDestroy.as.Destroy.impl.Op.decl.loc24_24.1
// CHECK:STDOUT:   .SubobjectDestroy = <poisoned>
// CHECK:STDOUT:   .SelfDestruct = <poisoned>
// CHECK:STDOUT:   extend %Destroy.ref
// CHECK:STDOUT:   witness = %Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NonTrivialDestroy {
// CHECK:STDOUT:   impl_decl @NonTrivialDestroy.as.Destroy.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NonTrivialDestroy [concrete = constants.%NonTrivialDestroy]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23: <witness> = impl_self_witness @NonTrivialDestroy.as.Destroy.impl.%Self.ref, @Destroy [concrete = constants.%.127]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NonTrivialDestroy
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonTrivialDestroy.as.Destroy.impl.Op.loc24_24.1(%self.param: %NonTrivialDestroy) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonTrivialDestroy.as.Destroy.impl.Op.loc24_24.2(%self.param: ref %NonTrivialDestroy) [thunk @NonTrivialDestroy.as.Destroy.impl.%NonTrivialDestroy.as.Destroy.impl.Op.decl.loc24_24.1 for imports.%Destroy.WithSelf.Op.decl, @Destroy.WithSelf.Op(constants.%Destroy.facet)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Op.ref: %NonTrivialDestroy.as.Destroy.impl.Op.type.987e0a.1 = name_ref Op, @NonTrivialDestroy.as.Destroy.impl.%NonTrivialDestroy.as.Destroy.impl.Op.decl.loc24_24.1 [concrete = constants.%NonTrivialDestroy.as.Destroy.impl.Op.4454e9.1]
// CHECK:STDOUT:   %NonTrivialDestroy.as.Destroy.impl.Op.bound: <bound method> = bound_method %self.param, %Op.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %NonTrivialDestroy.as.Destroy.impl.Op.call: init %empty_tuple.type = call %NonTrivialDestroy.as.Destroy.impl.Op.bound(%.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FormatIsNoDestroy(%T.loc28_23.2: %DestroyFormat_where.type.ad0) {
// CHECK:STDOUT:   %T.patt.loc28_23.2: %pattern_type.474 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc28_23.2 (constants.%T.patt.d12)]
// CHECK:STDOUT:   %T.loc28_23.1: %DestroyFormat_where.type.ad0 = symbolic_binding T, 0 [symbolic = %T.loc28_23.1 (constants.%T.5df)]
// CHECK:STDOUT:   %T.as_type.loc28_69.1: type = facet_access_type %T.loc28_23.1 [symbolic = %T.as_type.loc28_69.1 (constants.%T.as_type.725)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc28_69.1 [symbolic = %pattern_type (constants.%pattern_type.1fc)]
// CHECK:STDOUT:   %_.param_patt.loc28_67.2: @FormatIsNoDestroy.%pattern_type (%pattern_type.1fc) = ref_param_pattern [symbolic = %_.param_patt.loc28_67.2 (constants.%_.param_patt.6e1)]
// CHECK:STDOUT:   %_.patt.loc28_67.2: @FormatIsNoDestroy.%pattern_type (%pattern_type.1fc) = wrapper_binding_pattern _, %_.param_patt.loc28_67.2 [symbolic = %_.patt.loc28_67.2 (constants.%_.patt.d3b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc28_69.1 [symbolic = %require_complete (constants.%require_complete.260)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%_.param: ref @FormatIsNoDestroy.%T.as_type.loc28_69.1 (%T.as_type.725)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FormatIsTrivial(%T.loc29_21.2: %DestroyFormat_where.type.e28) {
// CHECK:STDOUT:   %T.patt.loc29_21.2: %pattern_type.5d6 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc29_21.2 (constants.%T.patt.c88)]
// CHECK:STDOUT:   %T.loc29_21.1: %DestroyFormat_where.type.e28 = symbolic_binding T, 0 [symbolic = %T.loc29_21.1 (constants.%T.28c)]
// CHECK:STDOUT:   %T.as_type.loc29_67.1: type = facet_access_type %T.loc29_21.1 [symbolic = %T.as_type.loc29_67.1 (constants.%T.as_type.789)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc29_67.1 [symbolic = %pattern_type (constants.%pattern_type.4fb)]
// CHECK:STDOUT:   %_.param_patt.loc29_65.2: @FormatIsTrivial.%pattern_type (%pattern_type.4fb) = ref_param_pattern [symbolic = %_.param_patt.loc29_65.2 (constants.%_.param_patt.88c)]
// CHECK:STDOUT:   %_.patt.loc29_65.2: @FormatIsTrivial.%pattern_type (%pattern_type.4fb) = wrapper_binding_pattern _, %_.param_patt.loc29_65.2 [symbolic = %_.patt.loc29_65.2 (constants.%_.patt.1ca)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc29_67.1 [symbolic = %require_complete (constants.%require_complete.b00)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%_.param: ref @FormatIsTrivial.%T.as_type.loc29_67.1 (%T.as_type.789)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FormatIsNonTrivial(%T.loc30_24.2: %DestroyFormat_where.type.0ba) {
// CHECK:STDOUT:   %T.patt.loc30_24.2: %pattern_type.cbf = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc30_24.2 (constants.%T.patt.de2)]
// CHECK:STDOUT:   %T.loc30_24.1: %DestroyFormat_where.type.0ba = symbolic_binding T, 0 [symbolic = %T.loc30_24.1 (constants.%T.39d)]
// CHECK:STDOUT:   %T.as_type.loc30_70.1: type = facet_access_type %T.loc30_24.1 [symbolic = %T.as_type.loc30_70.1 (constants.%T.as_type.6d2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc30_70.1 [symbolic = %pattern_type (constants.%pattern_type.353)]
// CHECK:STDOUT:   %_.param_patt.loc30_68.2: @FormatIsNonTrivial.%pattern_type (%pattern_type.353) = ref_param_pattern [symbolic = %_.param_patt.loc30_68.2 (constants.%_.param_patt.1bd)]
// CHECK:STDOUT:   %_.patt.loc30_68.2: @FormatIsNonTrivial.%pattern_type (%pattern_type.353) = wrapper_binding_pattern _, %_.param_patt.loc30_68.2 [symbolic = %_.patt.loc30_68.2 (constants.%_.patt.c81)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc30_70.1 [symbolic = %require_complete (constants.%require_complete.fb5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%_.param: ref @FormatIsNonTrivial.%T.as_type.loc30_70.1 (%T.as_type.6d2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %trivial_destroy.var: ref <error> = var_storage %trivial_destroy.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.loc41: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   assign %trivial_destroy.var, <error>
// CHECK:STDOUT:   %TrivialDestroy.ref: <error> = name_ref TrivialDestroy, <error> [concrete = <error>]
// CHECK:STDOUT:   %trivial_destroy: <error> = wrapper_binding trivial_destroy, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %trivial_destroy.patt: <error> = ref_binding_pattern trivial_destroy [concrete = <error>]
// CHECK:STDOUT:     %trivial_destroy.var_patt: <error> = var_pattern %trivial_destroy.patt [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FormatIsTrivial.ref: %FormatIsTrivial.type = name_ref FormatIsTrivial, file.%FormatIsTrivial.decl [concrete = constants.%FormatIsTrivial]
// CHECK:STDOUT:   %trivial_destroy.ref: <error> = name_ref trivial_destroy, %trivial_destroy [concrete = <error>]
// CHECK:STDOUT:   %.loc42: <error> = ref_tag %trivial_destroy.ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FormatIsNoDestroy(constants.%T.5df) {
// CHECK:STDOUT:   %T.patt.loc28_23.2 => constants.%T.patt.d12
// CHECK:STDOUT:   %T.loc28_23.1 => constants.%T.5df
// CHECK:STDOUT:   %T.as_type.loc28_69.1 => constants.%T.as_type.725
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1fc
// CHECK:STDOUT:   %_.param_patt.loc28_67.2 => constants.%_.param_patt.6e1
// CHECK:STDOUT:   %_.patt.loc28_67.2 => constants.%_.patt.d3b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FormatIsTrivial(constants.%T.28c) {
// CHECK:STDOUT:   %T.patt.loc29_21.2 => constants.%T.patt.c88
// CHECK:STDOUT:   %T.loc29_21.1 => constants.%T.28c
// CHECK:STDOUT:   %T.as_type.loc29_67.1 => constants.%T.as_type.789
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4fb
// CHECK:STDOUT:   %_.param_patt.loc29_65.2 => constants.%_.param_patt.88c
// CHECK:STDOUT:   %_.patt.loc29_65.2 => constants.%_.patt.1ca
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FormatIsNonTrivial(constants.%T.39d) {
// CHECK:STDOUT:   %T.patt.loc30_24.2 => constants.%T.patt.de2
// CHECK:STDOUT:   %T.loc30_24.1 => constants.%T.39d
// CHECK:STDOUT:   %T.as_type.loc30_70.1 => constants.%T.as_type.6d2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.353
// CHECK:STDOUT:   %_.param_patt.loc30_68.2 => constants.%_.param_patt.1bd
// CHECK:STDOUT:   %_.patt.loc30_68.2 => constants.%_.patt.c81
// CHECK:STDOUT: }
// CHECK:STDOUT:
