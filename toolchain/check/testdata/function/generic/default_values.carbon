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

//@dump-sem-ir-begin
fn Make(generic T: Core.Copy & Core.Destroy, t: T, d: D = {} as D) -> C(T) {
  let c: C(T) = {.t = t, .d = d};
  return c;
}

fn G() {
  let _: C(E) = Make(E, {} as E);
}
//@dump-sem-ir-end


// CHECK:STDOUT: --- generic_fn_specific_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.548: <witness> = impl_witness @D.as.Copy.impl.%Copy.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %D.as.Copy.impl.Op.type: type = fn_type @D.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %D.as.Copy.impl.Op: %D.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.48d: %Copy.type = facet_value %D, (%Copy.impl_witness.548) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.5bb: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.48d) [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.e92: <witness> = impl_witness @E.as.Copy.impl.%Copy.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.849: type = pattern_type %E [concrete]
// CHECK:STDOUT:   %E.as.Copy.impl.Op.type: type = fn_type @E.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %E.as.Copy.impl.Op: %E.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.b77: %Copy.type = facet_value %E, (%Copy.impl_witness.e92) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.0e9: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.b77) [concrete]
// CHECK:STDOUT:   %E.val: %E = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.7f7: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.7f7 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.1a6: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.b4e: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.1a6, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %Copy.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type.bd0: type = facet_type <@Copy & @Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81def.1: type = pattern_type %facet_type.bd0 [concrete]
// CHECK:STDOUT:   %T.patt.fe60cd.1: %pattern_type.e81def.1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f84392.1: %facet_type.bd0 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.6fb: type = class_type @C, @C(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.c1cc3e.1: type = facet_access_type %T.f84392.1 [symbolic]
// CHECK:STDOUT:   %require_complete.d83f06.1: <witness> = require_complete_type %T.as_type.c1cc3e.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type.c1cc3e.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.f43: type = pattern_type %C.6fb [symbolic]
// CHECK:STDOUT:   %.9d5: Core.Form = init_form %C.6fb [symbolic]
// CHECK:STDOUT:   %return.param_patt.f63: %pattern_type.f43 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f9d: %pattern_type.f43 = return_slot_pattern %return.param_patt.f63, %C.6fb [symbolic]
// CHECK:STDOUT:   %struct_type.t.d.56a: type = struct_type {.t: %T.as_type.c1cc3e.1, .d: %D} [symbolic]
// CHECK:STDOUT:   %require_complete.e6f: <witness> = require_complete_type %C.6fb [symbolic]
// CHECK:STDOUT:   %c.patt.225: %pattern_type.f43 = value_binding_pattern c [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c332.2: <witness> = lookup_impl_witness %T.f84392.1, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.d94: %Copy.type = facet_value %T.as_type.c1cc3e.1, (%Copy.lookup_impl_witness.d9c332.2) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.00c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.d94) [symbolic]
// CHECK:STDOUT:   %.970: type = fn_type_with_self_type %Copy.WithSelf.Op.type.00c, %Copy.facet.d94 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bec: %.970 = impl_witness_access %Copy.lookup_impl_witness.d9c332.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.32b: <specific function> = specific_impl_function %impl.elem0.bec, @Copy.WithSelf.Op(%Copy.facet.d94) [symbolic]
// CHECK:STDOUT:   %.e9c: type = fn_type_with_self_type %Copy.WithSelf.Op.type.5bb, %Copy.facet.48d [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2cc: <witness> = lookup_impl_witness %C.6fb, @Copy [symbolic]
// CHECK:STDOUT:   %.a18: require_specific_def_type = require_specific_def @C.as.Copy.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %Copy.facet.4b6: %Copy.type = facet_value %C.6fb, (%Copy.lookup_impl_witness.2cc) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.3d8: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.4b6) [symbolic]
// CHECK:STDOUT:   %.bb6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.3d8, %Copy.facet.4b6 [symbolic]
// CHECK:STDOUT:   %impl.elem0.070: %.bb6 = impl_witness_access %Copy.lookup_impl_witness.2cc, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bd1: <specific function> = specific_impl_function %impl.elem0.070, @Copy.WithSelf.Op(%Copy.facet.4b6) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.cfa: <witness> = lookup_impl_witness %C.6fb, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_46.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.08b: <witness> = lookup_impl_witness %C.6fb, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.068: %SubobjectDestroy.type = facet_value %C.6fb, (%SubobjectDestroy.lookup_impl_witness.08b) [symbolic]
// CHECK:STDOUT:   %.f8d: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.068) [symbolic]
// CHECK:STDOUT:   %Destroy.facet.98e: %Destroy.type = facet_value %C.6fb, (%DestroyImpl.lookup_impl_witness.cfa, %SubobjectDestroy.lookup_impl_witness.08b) [symbolic]
// CHECK:STDOUT:   %t.param_patt.adf: %pattern_type.2c66b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt.035: %pattern_type.2c66b4.2 = wrapper_binding_pattern t, %t.param_patt.adf [symbolic]
// CHECK:STDOUT:   %d.param_patt: %pattern_type.d8d = value_param_pattern [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.d8d = wrapper_binding_pattern d, %d.param_patt [concrete]
// CHECK:STDOUT:   %.a37: ref %D = temporary invalid, %D.val [concrete]
// CHECK:STDOUT:   %.ec6: %pattern_type.d8d = default_value_pattern %d.patt, %D.val [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_13 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.612: %SubobjectDestroy.type = facet_value %E, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1c0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.612) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type.bd0 = facet_value %E, (%Copy.impl_witness.e92, %DestroyImpl.impl_witness.1c0, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %C.db0: type = class_type @C, @C(%facet_value) [concrete]
// CHECK:STDOUT:   %C.elem.549: type = unbound_element_type %C.db0, %E [concrete]
// CHECK:STDOUT:   %C.elem.967: type = unbound_element_type %C.db0, %D [concrete]
// CHECK:STDOUT:   %struct_type.t.d.d81: type = struct_type {.t: %E, .d: %D} [concrete]
// CHECK:STDOUT:   %complete_type.031: <witness> = complete_type_witness %struct_type.t.d.d81 [concrete]
// CHECK:STDOUT:   %pattern_type.e5e: type = pattern_type %C.db0 [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.e5e = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %t.param_patt.307: %pattern_type.849 = value_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.404: %pattern_type.849 = wrapper_binding_pattern t, %t.param_patt.307 [concrete]
// CHECK:STDOUT:   %.600: Core.Form = init_form %C.db0 [concrete]
// CHECK:STDOUT:   %return.param_patt.cee: %pattern_type.e5e = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.86e: %pattern_type.e5e = return_slot_pattern %return.param_patt.cee, %C.db0 [concrete]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make, @Make(%facet_value) [concrete]
// CHECK:STDOUT:   %.b7b: ref %E = temporary invalid, %E.val [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f61: %SubobjectDestroy.type = facet_value %D, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.30f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f61) [concrete]
// CHECK:STDOUT:   %Destroy.facet.448: %Destroy.type = facet_value %D, (%DestroyImpl.impl_witness.30f, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %Destroy.facet.2f7: %Destroy.type = facet_value %E, (%DestroyImpl.impl_witness.1c0, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_32.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.3b5: %SubobjectDestroy.type = facet_value %C.db0, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b88: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.3b5) [concrete]
// CHECK:STDOUT:   %.4ec: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.3b5) [concrete]
// CHECK:STDOUT:   %Destroy.facet.455: %Destroy.type = facet_value %C.db0, (%DestroyImpl.impl_witness.b88, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.546: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.455) [concrete]
// CHECK:STDOUT:   %c.patt.292: %pattern_type.e5e = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %.01c: type = fn_type_with_self_type %Copy.WithSelf.Op.type.0e9, %Copy.facet.b77 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.c5f: <witness> = impl_witness @C.as.Copy.impl.%Copy.impl_witness_table, @C.as.Copy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.type.a8b: type = fn_type @C.as.Copy.impl.Op, @C.as.Copy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.46c: %C.as.Copy.impl.Op.type.a8b = struct_value () [concrete]
// CHECK:STDOUT:   %.053: require_specific_def_type = require_specific_def @C.as.Copy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %Copy.facet.e96: %Copy.type = facet_value %C.db0, (%Copy.impl_witness.c5f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.06d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.e96) [concrete]
// CHECK:STDOUT:   %.374: type = fn_type_with_self_type %Copy.WithSelf.Op.type.06d, %Copy.facet.e96 [concrete]
// CHECK:STDOUT:   %C.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %C.as.Copy.impl.Op.46c, @C.as.Copy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.9c3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %T.patt.loc25_18.1: %pattern_type.e81def.1 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_18.2 (constants.%T.patt.fe60cd.1)]
// CHECK:STDOUT:     %t.param_patt.loc25_47.1: @Make.%pattern_type.loc25_47 (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %t.param_patt.loc25_47.2 (constants.%t.param_patt.adf)]
// CHECK:STDOUT:     %t.patt.loc25_47.1: @Make.%pattern_type.loc25_47 (%pattern_type.2c66b4.2) = wrapper_binding_pattern t, %t.param_patt.loc25_47.1 [symbolic = %t.patt.loc25_47.2 (constants.%t.patt.035)]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.d8d = value_param_pattern [concrete = constants.%d.param_patt]
// CHECK:STDOUT:     %d.patt: %pattern_type.d8d = wrapper_binding_pattern d, %d.param_patt [concrete = constants.%d.patt]
// CHECK:STDOUT:     %.loc25_57: %pattern_type.d8d = default_value_pattern %d.patt, %.loc25_62.3 [concrete = constants.%.ec6]
// CHECK:STDOUT:     %return.param_patt.loc25_74.1: @Make.%pattern_type.loc25_74 (%pattern_type.f43) = out_param_pattern [symbolic = %return.param_patt.loc25_74.2 (constants.%return.param_patt.f63)]
// CHECK:STDOUT:     %return.patt.loc25_68.1: @Make.%pattern_type.loc25_74 (%pattern_type.f43) = return_slot_pattern %return.param_patt.loc25_74.1, %C.loc25_74.2 [symbolic = %return.patt.loc25_68.2 (constants.%return.patt.f9d)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc25_60.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %D.ref.loc25_65: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc25_60.2: ref %D = temporary_storage
// CHECK:STDOUT:     %.loc25_60.3: init %D to %.loc25_60.2 = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:     %.loc25_62.1: init %D = converted %.loc25_60.1, %.loc25_60.3 [concrete = constants.%D.val]
// CHECK:STDOUT:     %.loc25_62.2: ref %D = temporary %.loc25_60.2, %.loc25_62.1 [concrete = constants.%.a37]
// CHECK:STDOUT:     %.loc25_62.3: %D = acquire_value %.loc25_62.2 [concrete = constants.%D.val]
// CHECK:STDOUT:     %C.ref.loc25: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref.loc25_73: %facet_type.bd0 = name_ref T, %T.loc25_18.2 [symbolic = %T.loc25_18.1 (constants.%T.f84392.1)]
// CHECK:STDOUT:     %C.loc25_74.2: type = class_type @C, @C(constants.%T.f84392.1) [symbolic = %C.loc25_74.1 (constants.%C.6fb)]
// CHECK:STDOUT:     %.loc25_74.3: Core.Form = init_form %C.loc25_74.2 [symbolic = %.loc25_74.2 (constants.%.9d5)]
// CHECK:STDOUT:     %.loc25_30.1: type = splice_block %.loc25_30.3 [concrete = constants.%facet_type.bd0] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref.loc25_20: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:       %Core.ref.loc25_32: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %impl.elem0.loc25: %.b4e = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc25: <bound method> = bound_method %Copy.ref, %impl.elem0.loc25 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method.loc25(%Copy.ref, %Destroy.ref) [concrete = constants.%facet_type.bd0]
// CHECK:STDOUT:       %.loc25_30.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type.bd0]
// CHECK:STDOUT:       %.loc25_30.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc25_30.2 [concrete = constants.%facet_type.bd0]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc25_18.2: %facet_type.bd0 = symbolic_binding T, 0 [symbolic = %T.loc25_18.1 (constants.%T.f84392.1)]
// CHECK:STDOUT:     %t.param: @Make.%T.as_type.loc25_49.1 (%T.as_type.c1cc3e.1) = value_param call_param0
// CHECK:STDOUT:     %.loc25_49.1: type = splice_block %.loc25_49.2 [symbolic = %T.as_type.loc25_49.1 (constants.%T.as_type.c1cc3e.1)] {
// CHECK:STDOUT:       %T.ref.loc25_49: %facet_type.bd0 = name_ref T, %T.loc25_18.2 [symbolic = %T.loc25_18.1 (constants.%T.f84392.1)]
// CHECK:STDOUT:       %T.as_type.loc25_49.2: type = facet_access_type %T.ref.loc25_49 [symbolic = %T.as_type.loc25_49.1 (constants.%T.as_type.c1cc3e.1)]
// CHECK:STDOUT:       %.loc25_49.2: type = converted %T.ref.loc25_49, %T.as_type.loc25_49.2 [symbolic = %T.as_type.loc25_49.1 (constants.%T.as_type.c1cc3e.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @Make.%T.as_type.loc25_49.1 (%T.as_type.c1cc3e.1) = wrapper_binding t, %t.param
// CHECK:STDOUT:     %d.param: %D = value_param call_param1
// CHECK:STDOUT:     %D.ref.loc25_55: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %d: %D = wrapper_binding d, %d.param
// CHECK:STDOUT:     %return.param: ref @Make.%C.loc25_74.1 (%C.6fb) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Make.%C.loc25_74.1 (%C.6fb) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(%T.loc25_18.2: %facet_type.bd0) {
// CHECK:STDOUT:   %T.patt.loc25_18.2: %pattern_type.e81def.1 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_18.2 (constants.%T.patt.fe60cd.1)]
// CHECK:STDOUT:   %T.loc25_18.1: %facet_type.bd0 = symbolic_binding T, 0 [symbolic = %T.loc25_18.1 (constants.%T.f84392.1)]
// CHECK:STDOUT:   %T.as_type.loc25_49.1: type = facet_access_type %T.loc25_18.1 [symbolic = %T.as_type.loc25_49.1 (constants.%T.as_type.c1cc3e.1)]
// CHECK:STDOUT:   %pattern_type.loc25_47: type = pattern_type %T.as_type.loc25_49.1 [symbolic = %pattern_type.loc25_47 (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %t.param_patt.loc25_47.2: @Make.%pattern_type.loc25_47 (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %t.param_patt.loc25_47.2 (constants.%t.param_patt.adf)]
// CHECK:STDOUT:   %t.patt.loc25_47.2: @Make.%pattern_type.loc25_47 (%pattern_type.2c66b4.2) = wrapper_binding_pattern t, %t.param_patt.loc25_47.2 [symbolic = %t.patt.loc25_47.2 (constants.%t.patt.035)]
// CHECK:STDOUT:   %C.loc25_74.1: type = class_type @C, @C(%T.loc25_18.1) [symbolic = %C.loc25_74.1 (constants.%C.6fb)]
// CHECK:STDOUT:   %.loc25_74.2: Core.Form = init_form %C.loc25_74.1 [symbolic = %.loc25_74.2 (constants.%.9d5)]
// CHECK:STDOUT:   %pattern_type.loc25_74: type = pattern_type %C.loc25_74.1 [symbolic = %pattern_type.loc25_74 (constants.%pattern_type.f43)]
// CHECK:STDOUT:   %return.param_patt.loc25_74.2: @Make.%pattern_type.loc25_74 (%pattern_type.f43) = out_param_pattern [symbolic = %return.param_patt.loc25_74.2 (constants.%return.param_patt.f63)]
// CHECK:STDOUT:   %return.patt.loc25_68.2: @Make.%pattern_type.loc25_74 (%pattern_type.f43) = return_slot_pattern %return.param_patt.loc25_74.2, %C.loc25_74.1 [symbolic = %return.patt.loc25_68.2 (constants.%return.patt.f9d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc25: <witness> = require_complete_type %T.as_type.loc25_49.1 [symbolic = %require_complete.loc25 (constants.%require_complete.d83f06.1)]
// CHECK:STDOUT:   %require_complete.loc26: <witness> = require_complete_type %C.loc25_74.1 [symbolic = %require_complete.loc26 (constants.%require_complete.e6f)]
// CHECK:STDOUT:   %c.patt.loc26_8.2: @Make.%pattern_type.loc25_74 (%pattern_type.f43) = value_binding_pattern c [symbolic = %c.patt.loc26_8.2 (constants.%c.patt.225)]
// CHECK:STDOUT:   %struct_type.t.d: type = struct_type {.t: @Make.%T.as_type.loc25_49.1 (%T.as_type.c1cc3e.1), .d: %D} [symbolic = %struct_type.t.d (constants.%struct_type.t.d.56a)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.loc26: <witness> = lookup_impl_witness %T.loc25_18.1, @Copy [symbolic = %Copy.lookup_impl_witness.loc26 (constants.%Copy.lookup_impl_witness.d9c332.2)]
// CHECK:STDOUT:   %Copy.facet.loc26_23.3: %Copy.type = facet_value %T.as_type.loc25_49.1, (%Copy.lookup_impl_witness.loc26) [symbolic = %Copy.facet.loc26_23.3 (constants.%Copy.facet.d94)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.loc26: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc26_23.3) [symbolic = %Copy.WithSelf.Op.type.loc26 (constants.%Copy.WithSelf.Op.type.00c)]
// CHECK:STDOUT:   %.loc26_23.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type.loc26, %Copy.facet.loc26_23.3 [symbolic = %.loc26_23.3 (constants.%.970)]
// CHECK:STDOUT:   %impl.elem0.loc26_23.2: @Make.%.loc26_23.3 (%.970) = impl_witness_access %Copy.lookup_impl_witness.loc26, element0 [symbolic = %impl.elem0.loc26_23.2 (constants.%impl.elem0.bec)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_23.2: <specific function> = specific_impl_function %impl.elem0.loc26_23.2, @Copy.WithSelf.Op(%Copy.facet.loc26_23.3) [symbolic = %specific_impl_fn.loc26_23.2 (constants.%specific_impl_fn.32b)]
// CHECK:STDOUT:   %.loc27_10.3: require_specific_def_type = require_specific_def @C.as.Copy.impl(%T.loc25_18.1) [symbolic = %.loc27_10.3 (constants.%.a18)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.loc27: <witness> = lookup_impl_witness %C.loc25_74.1, @Copy [symbolic = %Copy.lookup_impl_witness.loc27 (constants.%Copy.lookup_impl_witness.2cc)]
// CHECK:STDOUT:   %Copy.facet.loc27_10.3: %Copy.type = facet_value %C.loc25_74.1, (%Copy.lookup_impl_witness.loc27) [symbolic = %Copy.facet.loc27_10.3 (constants.%Copy.facet.4b6)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.loc27: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc27_10.3) [symbolic = %Copy.WithSelf.Op.type.loc27 (constants.%Copy.WithSelf.Op.type.3d8)]
// CHECK:STDOUT:   %.loc27_10.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type.loc27, %Copy.facet.loc27_10.3 [symbolic = %.loc27_10.4 (constants.%.bb6)]
// CHECK:STDOUT:   %impl.elem0.loc27_10.2: @Make.%.loc27_10.4 (%.bb6) = impl_witness_access %Copy.lookup_impl_witness.loc27, element0 [symbolic = %impl.elem0.loc27_10.2 (constants.%impl.elem0.070)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_10.2: <specific function> = specific_impl_function %impl.elem0.loc27_10.2, @Copy.WithSelf.Op(%Copy.facet.loc27_10.3) [symbolic = %specific_impl_fn.loc27_10.2 (constants.%specific_impl_fn.bd1)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %C.loc25_74.1, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness.08b)]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %C.loc25_74.1, (%SubobjectDestroy.lookup_impl_witness) [symbolic = %SubobjectDestroy.facet (constants.%SubobjectDestroy.facet.068)]
// CHECK:STDOUT:   %.loc26_32.13: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet) [symbolic = %.loc26_32.13 (constants.%.f8d)]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %C.loc25_74.1, @DestroyImpl [symbolic = %DestroyImpl.lookup_impl_witness (constants.%DestroyImpl.lookup_impl_witness.cfa)]
// CHECK:STDOUT:   %Destroy.facet.loc26_32.3: %Destroy.type = facet_value %C.loc25_74.1, (%DestroyImpl.lookup_impl_witness, %SubobjectDestroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc26_32.3 (constants.%Destroy.facet.98e)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @Make.%T.as_type.loc25_49.1 (%T.as_type.c1cc3e.1), %d.param: %D) -> out %return.param: @Make.%C.loc25_74.1 (%C.6fb) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: @Make.%T.as_type.loc25_49.1 (%T.as_type.c1cc3e.1) = name_ref t, %t
// CHECK:STDOUT:     %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:     %.loc26_32.1: @Make.%struct_type.t.d (%struct_type.t.d.56a) = struct_literal (%t.ref, %d.ref)
// CHECK:STDOUT:     %impl.elem0.loc26_23.1: @Make.%.loc26_23.3 (%.970) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c332.2, element0 [symbolic = %impl.elem0.loc26_23.2 (constants.%impl.elem0.bec)]
// CHECK:STDOUT:     %bound_method.loc26_23.1: <bound method> = bound_method %t.ref, %impl.elem0.loc26_23.1
// CHECK:STDOUT:     %Copy.facet.loc26_23.1: %Copy.type = facet_value constants.%T.as_type.c1cc3e.1, (constants.%Copy.lookup_impl_witness.d9c332.2) [symbolic = %Copy.facet.loc26_23.3 (constants.%Copy.facet.d94)]
// CHECK:STDOUT:     %.loc26_23.1: %Copy.type = converted constants.%T.as_type.c1cc3e.1, %Copy.facet.loc26_23.1 [symbolic = %Copy.facet.loc26_23.3 (constants.%Copy.facet.d94)]
// CHECK:STDOUT:     %Copy.facet.loc26_23.2: %Copy.type = facet_value constants.%T.as_type.c1cc3e.1, (constants.%Copy.lookup_impl_witness.d9c332.2) [symbolic = %Copy.facet.loc26_23.3 (constants.%Copy.facet.d94)]
// CHECK:STDOUT:     %.loc26_23.2: %Copy.type = converted constants.%T.as_type.c1cc3e.1, %Copy.facet.loc26_23.2 [symbolic = %Copy.facet.loc26_23.3 (constants.%Copy.facet.d94)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_23.1: <specific function> = specific_impl_function %impl.elem0.loc26_23.1, @Copy.WithSelf.Op(constants.%Copy.facet.d94) [symbolic = %specific_impl_fn.loc26_23.2 (constants.%specific_impl_fn.32b)]
// CHECK:STDOUT:     %bound_method.loc26_23.2: <bound method> = bound_method %t.ref, %specific_impl_fn.loc26_23.1
// CHECK:STDOUT:     %.loc26_32.2: ref @Make.%T.as_type.loc25_49.1 (%T.as_type.c1cc3e.1) = splice_block %.loc26_32.12 {
// CHECK:STDOUT:       %.loc26_32.11: ref @Make.%C.loc25_74.1 (%C.6fb) = temporary_storage
// CHECK:STDOUT:       %.loc26_32.12: ref @Make.%T.as_type.loc25_49.1 (%T.as_type.c1cc3e.1) = class_element_access %.loc26_32.11, element0
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc26: init @Make.%T.as_type.loc25_49.1 (%T.as_type.c1cc3e.1) to %.loc26_32.2 = call %bound_method.loc26_23.2(%t.ref)
// CHECK:STDOUT:     %.loc26_32.3: init @Make.%T.as_type.loc25_49.1 (%T.as_type.c1cc3e.1) to %.loc26_32.12 = in_place_init %Copy.WithSelf.Op.call.loc26
// CHECK:STDOUT:     %impl.elem0.loc26_31: %.e9c = impl_witness_access constants.%Copy.impl_witness.548, element0 [concrete = constants.%D.as.Copy.impl.Op]
// CHECK:STDOUT:     %bound_method.loc26_31: <bound method> = bound_method %d.ref, %impl.elem0.loc26_31
// CHECK:STDOUT:     %.loc26_32.4: ref %D = class_element_access %.loc26_32.11, element1
// CHECK:STDOUT:     %D.as.Copy.impl.Op.call: init %D to %.loc26_32.4 = call %bound_method.loc26_31(%d.ref)
// CHECK:STDOUT:     %.loc26_32.5: init @Make.%C.loc25_74.1 (%C.6fb) to %.loc26_32.11 = class_init (%.loc26_32.3, %D.as.Copy.impl.Op.call)
// CHECK:STDOUT:     %.loc26_32.6: init @Make.%C.loc25_74.1 (%C.6fb) = converted %.loc26_32.1, %.loc26_32.5
// CHECK:STDOUT:     %.loc26_32.7: ref @Make.%C.loc25_74.1 (%C.6fb) = temporary %.loc26_32.11, %.loc26_32.6
// CHECK:STDOUT:     %.loc26_32.8: @Make.%C.loc25_74.1 (%C.6fb) = acquire_value %.loc26_32.7
// CHECK:STDOUT:     %.loc26_13: type = splice_block %C.loc26 [symbolic = %C.loc25_74.1 (constants.%C.6fb)] {
// CHECK:STDOUT:       %C.ref.loc26: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref.loc26: %facet_type.bd0 = name_ref T, %T.loc25_18.2 [symbolic = %T.loc25_18.1 (constants.%T.f84392.1)]
// CHECK:STDOUT:       %C.loc26: type = class_type @C, @C(constants.%T.f84392.1) [symbolic = %C.loc25_74.1 (constants.%C.6fb)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @Make.%C.loc25_74.1 (%C.6fb) = wrapper_binding c, %.loc26_32.8
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %c.patt.loc26_8.1: @Make.%pattern_type.loc25_74 (%pattern_type.f43) = value_binding_pattern c [symbolic = %c.patt.loc26_8.2 (constants.%c.patt.225)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.ref: @Make.%C.loc25_74.1 (%C.6fb) = name_ref c, %c
// CHECK:STDOUT:     %impl.elem0.loc27_10.1: @Make.%.loc27_10.4 (%.bb6) = impl_witness_access constants.%Copy.lookup_impl_witness.2cc, element0 [symbolic = %impl.elem0.loc27_10.2 (constants.%impl.elem0.070)]
// CHECK:STDOUT:     %bound_method.loc27_10.1: <bound method> = bound_method %c.ref, %impl.elem0.loc27_10.1
// CHECK:STDOUT:     %Copy.facet.loc27_10.1: %Copy.type = facet_value constants.%C.6fb, (constants.%Copy.lookup_impl_witness.2cc) [symbolic = %Copy.facet.loc27_10.3 (constants.%Copy.facet.4b6)]
// CHECK:STDOUT:     %.loc27_10.1: %Copy.type = converted constants.%C.6fb, %Copy.facet.loc27_10.1 [symbolic = %Copy.facet.loc27_10.3 (constants.%Copy.facet.4b6)]
// CHECK:STDOUT:     %Copy.facet.loc27_10.2: %Copy.type = facet_value constants.%C.6fb, (constants.%Copy.lookup_impl_witness.2cc) [symbolic = %Copy.facet.loc27_10.3 (constants.%Copy.facet.4b6)]
// CHECK:STDOUT:     %.loc27_10.2: %Copy.type = converted constants.%C.6fb, %Copy.facet.loc27_10.2 [symbolic = %Copy.facet.loc27_10.3 (constants.%Copy.facet.4b6)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_10.1: <specific function> = specific_impl_function %impl.elem0.loc27_10.1, @Copy.WithSelf.Op(constants.%Copy.facet.4b6) [symbolic = %specific_impl_fn.loc27_10.2 (constants.%specific_impl_fn.bd1)]
// CHECK:STDOUT:     %bound_method.loc27_10.2: <bound method> = bound_method %c.ref, %specific_impl_fn.loc27_10.1
// CHECK:STDOUT:     %.loc25_74.1: ref @Make.%C.loc25_74.1 (%C.6fb) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc27: init @Make.%C.loc25_74.1 (%C.6fb) to %.loc25_74.1 = call %bound_method.loc27_10.2(%c.ref)
// CHECK:STDOUT:     %Destroy.facet.loc26_32.1: %Destroy.type = facet_value constants.%C.6fb, (constants.%DestroyImpl.lookup_impl_witness.cfa, constants.%SubobjectDestroy.lookup_impl_witness.08b) [symbolic = %Destroy.facet.loc26_32.3 (constants.%Destroy.facet.98e)]
// CHECK:STDOUT:     %.loc26_32.9: %Destroy.type = converted constants.%C.6fb, %Destroy.facet.loc26_32.1 [symbolic = %Destroy.facet.loc26_32.3 (constants.%Destroy.facet.98e)]
// CHECK:STDOUT:     %Destroy.facet.loc26_32.2: %Destroy.type = facet_value constants.%C.6fb, (constants.%DestroyImpl.lookup_impl_witness.cfa, constants.%SubobjectDestroy.lookup_impl_witness.08b) [symbolic = %Destroy.facet.loc26_32.3 (constants.%Destroy.facet.98e)]
// CHECK:STDOUT:     %.loc26_32.10: %Destroy.type = converted constants.%C.6fb, %Destroy.facet.loc26_32.2 [symbolic = %Destroy.facet.loc26_32.3 (constants.%Destroy.facet.98e)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc26_32.7)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call.loc27 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %E.ref.loc31_22: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %.loc31_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %E.ref.loc31_31: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %.loc31_26.2: ref %E = temporary_storage
// CHECK:STDOUT:   %.loc31_26.3: init %E to %.loc31_26.2 = class_init () [concrete = constants.%E.val]
// CHECK:STDOUT:   %.loc31_28.1: init %E = converted %.loc31_26.1, %.loc31_26.3 [concrete = constants.%E.val]
// CHECK:STDOUT:   %facet_value.loc31_32.1: %facet_type.bd0 = facet_value %E.ref.loc31_22, (constants.%Copy.impl_witness.e92, constants.%DestroyImpl.impl_witness.1c0, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc31_32.1: %facet_type.bd0 = converted %E.ref.loc31_22, %facet_value.loc31_32.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc31_32.2: %facet_type.bd0 = facet_value constants.%E, (constants.%Copy.impl_witness.e92, constants.%DestroyImpl.impl_witness.1c0, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc31_32.2: %facet_type.bd0 = converted constants.%E, %facet_value.loc31_32.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc31_32.3: %facet_type.bd0 = facet_value constants.%E, (constants.%Copy.impl_witness.e92, constants.%DestroyImpl.impl_witness.1c0, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc31_32.3: %facet_type.bd0 = converted constants.%E, %facet_value.loc31_32.3 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make.ref, @Make(constants.%facet_value) [concrete = constants.%Make.specific_fn]
// CHECK:STDOUT:   %.loc31_32.4: ref %C.db0 = temporary_storage
// CHECK:STDOUT:   %.loc31_28.2: ref %E = temporary %.loc31_26.2, %.loc31_28.1 [concrete = constants.%.b7b]
// CHECK:STDOUT:   %.loc31_28.3: %E = acquire_value %.loc31_28.2 [concrete = constants.%E.val]
// CHECK:STDOUT:   %Make.call: init %C.db0 to %.loc31_32.4 = call %Make.specific_fn(%.loc31_28.3, @Make.%.loc25_62.3)
// CHECK:STDOUT:   %.loc31_32.5: ref %C.db0 = temporary %.loc31_32.4, %Make.call
// CHECK:STDOUT:   %.loc31_32.6: %C.db0 = acquire_value %.loc31_32.5
// CHECK:STDOUT:   %.loc31_13.1: type = splice_block %C [concrete = constants.%C.db0] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %E.ref.loc31_12: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %facet_value.loc31_13: %facet_type.bd0 = facet_value %E.ref.loc31_12, (constants.%Copy.impl_witness.e92, constants.%DestroyImpl.impl_witness.1c0, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value]
// CHECK:STDOUT:     %.loc31_13.2: %facet_type.bd0 = converted %E.ref.loc31_12, %facet_value.loc31_13 [concrete = constants.%facet_value]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%facet_value) [concrete = constants.%C.db0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %C.db0 = wrapper_binding _, %.loc31_32.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.e5e = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc31_32.1: %Destroy.type = facet_value constants.%C.db0, (constants.%DestroyImpl.impl_witness.b88, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.455]
// CHECK:STDOUT:   %.loc31_32.7: %Destroy.type = converted constants.%C.db0, %Destroy.facet.loc31_32.1 [concrete = constants.%Destroy.facet.455]
// CHECK:STDOUT:   %Destroy.facet.loc31_32.2: %Destroy.type = facet_value constants.%C.db0, (constants.%DestroyImpl.impl_witness.b88, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.455]
// CHECK:STDOUT:   %.loc31_32.8: %Destroy.type = converted constants.%C.db0, %Destroy.facet.loc31_32.2 [concrete = constants.%Destroy.facet.455]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_32: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_32.5)
// CHECK:STDOUT:   %Destroy.facet.loc31_28.1: %Destroy.type = facet_value constants.%E, (constants.%DestroyImpl.impl_witness.1c0, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.2f7]
// CHECK:STDOUT:   %.loc31_28.4: %Destroy.type = converted constants.%E, %Destroy.facet.loc31_28.1 [concrete = constants.%Destroy.facet.2f7]
// CHECK:STDOUT:   %Destroy.facet.loc31_28.2: %Destroy.type = facet_value constants.%E, (constants.%DestroyImpl.impl_witness.1c0, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.2f7]
// CHECK:STDOUT:   %.loc31_28.5: %Destroy.type = converted constants.%E, %Destroy.facet.loc31_28.2 [concrete = constants.%Destroy.facet.2f7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_28: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_28.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_13(%self.param: ref %E) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_32.1(%self.param: ref %struct_type.t.d.d81) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc31_32.1: ref %D = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc31_32.1: %Destroy.type = facet_value constants.%D, (constants.%DestroyImpl.impl_witness.30f, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.448]
// CHECK:STDOUT:   %.loc31_32.2: %Destroy.type = converted constants.%D, %Destroy.facet.loc31_32.1 [concrete = constants.%Destroy.facet.448]
// CHECK:STDOUT:   %Destroy.facet.loc31_32.2: %Destroy.type = facet_value constants.%D, (constants.%DestroyImpl.impl_witness.30f, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.448]
// CHECK:STDOUT:   %.loc31_32.3: %Destroy.type = converted constants.%D, %Destroy.facet.loc31_32.2 [concrete = constants.%Destroy.facet.448]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_32.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_32.1)
// CHECK:STDOUT:   %.loc31_32.4: ref %E = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc31_32.3: %Destroy.type = facet_value constants.%E, (constants.%DestroyImpl.impl_witness.1c0, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.2f7]
// CHECK:STDOUT:   %.loc31_32.5: %Destroy.type = converted constants.%E, %Destroy.facet.loc31_32.3 [concrete = constants.%Destroy.facet.2f7]
// CHECK:STDOUT:   %Destroy.facet.loc31_32.4: %Destroy.type = facet_value constants.%E, (constants.%DestroyImpl.impl_witness.1c0, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.2f7]
// CHECK:STDOUT:   %.loc31_32.6: %Destroy.type = converted constants.%E, %Destroy.facet.loc31_32.4 [concrete = constants.%Destroy.facet.2f7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_32.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_32.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_32.2(%self.param: ref %C.db0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %C.elem.967 = name_ref d, @C.%field_decl.loc15 [concrete = @C.%field_decl.loc15]
// CHECK:STDOUT:   %.loc31_32.1: ref %D = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc31_32.1: %Destroy.type = facet_value constants.%D, (constants.%DestroyImpl.impl_witness.30f, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.448]
// CHECK:STDOUT:   %.loc31_32.2: %Destroy.type = converted constants.%D, %Destroy.facet.loc31_32.1 [concrete = constants.%Destroy.facet.448]
// CHECK:STDOUT:   %Destroy.facet.loc31_32.2: %Destroy.type = facet_value constants.%D, (constants.%DestroyImpl.impl_witness.30f, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.448]
// CHECK:STDOUT:   %.loc31_32.3: %Destroy.type = converted constants.%D, %Destroy.facet.loc31_32.2 [concrete = constants.%Destroy.facet.448]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_32.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_32.1)
// CHECK:STDOUT:   %t.ref: %C.elem.549 = name_ref t, @C.%field_decl.loc14 [concrete = @C.%field_decl.loc14]
// CHECK:STDOUT:   %.loc31_32.4: ref %E = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc31_32.3: %Destroy.type = facet_value constants.%E, (constants.%DestroyImpl.impl_witness.1c0, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.2f7]
// CHECK:STDOUT:   %.loc31_32.5: %Destroy.type = converted constants.%E, %Destroy.facet.loc31_32.3 [concrete = constants.%Destroy.facet.2f7]
// CHECK:STDOUT:   %Destroy.facet.loc31_32.4: %Destroy.type = facet_value constants.%E, (constants.%DestroyImpl.impl_witness.1c0, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.2f7]
// CHECK:STDOUT:   %.loc31_32.6: %Destroy.type = converted constants.%E, %Destroy.facet.loc31_32.4 [concrete = constants.%Destroy.facet.2f7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_32.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_32.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%T.f84392.1) {
// CHECK:STDOUT:   %T.patt.loc25_18.2 => constants.%T.patt.fe60cd.1
// CHECK:STDOUT:   %T.loc25_18.1 => constants.%T.f84392.1
// CHECK:STDOUT:   %T.as_type.loc25_49.1 => constants.%T.as_type.c1cc3e.1
// CHECK:STDOUT:   %pattern_type.loc25_47 => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %t.param_patt.loc25_47.2 => constants.%t.param_patt.adf
// CHECK:STDOUT:   %t.patt.loc25_47.2 => constants.%t.patt.035
// CHECK:STDOUT:   %C.loc25_74.1 => constants.%C.6fb
// CHECK:STDOUT:   %.loc25_74.2 => constants.%.9d5
// CHECK:STDOUT:   %pattern_type.loc25_74 => constants.%pattern_type.f43
// CHECK:STDOUT:   %return.param_patt.loc25_74.2 => constants.%return.param_patt.f63
// CHECK:STDOUT:   %return.patt.loc25_68.2 => constants.%return.patt.f9d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%facet_value) {
// CHECK:STDOUT:   %T.patt.loc25_18.2 => constants.%T.patt.fe60cd.1
// CHECK:STDOUT:   %T.loc25_18.1 => constants.%facet_value
// CHECK:STDOUT:   %T.as_type.loc25_49.1 => constants.%E
// CHECK:STDOUT:   %pattern_type.loc25_47 => constants.%pattern_type.849
// CHECK:STDOUT:   %t.param_patt.loc25_47.2 => constants.%t.param_patt.307
// CHECK:STDOUT:   %t.patt.loc25_47.2 => constants.%t.patt.404
// CHECK:STDOUT:   %C.loc25_74.1 => constants.%C.db0
// CHECK:STDOUT:   %.loc25_74.2 => constants.%.600
// CHECK:STDOUT:   %pattern_type.loc25_74 => constants.%pattern_type.e5e
// CHECK:STDOUT:   %return.param_patt.loc25_74.2 => constants.%return.param_patt.cee
// CHECK:STDOUT:   %return.patt.loc25_68.2 => constants.%return.patt.86e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc25 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc26 => constants.%complete_type.031
// CHECK:STDOUT:   %c.patt.loc26_8.2 => constants.%c.patt.292
// CHECK:STDOUT:   %struct_type.t.d => constants.%struct_type.t.d.d81
// CHECK:STDOUT:   %Copy.lookup_impl_witness.loc26 => constants.%Copy.impl_witness.e92
// CHECK:STDOUT:   %Copy.facet.loc26_23.3 => constants.%Copy.facet.b77
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.loc26 => constants.%Copy.WithSelf.Op.type.0e9
// CHECK:STDOUT:   %.loc26_23.3 => constants.%.01c
// CHECK:STDOUT:   %impl.elem0.loc26_23.2 => constants.%E.as.Copy.impl.Op
// CHECK:STDOUT:   %specific_impl_fn.loc26_23.2 => constants.%E.as.Copy.impl.Op
// CHECK:STDOUT:   %.loc27_10.3 => constants.%.053
// CHECK:STDOUT:   %Copy.lookup_impl_witness.loc27 => constants.%Copy.impl_witness.c5f
// CHECK:STDOUT:   %Copy.facet.loc27_10.3 => constants.%Copy.facet.e96
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.loc27 => constants.%Copy.WithSelf.Op.type.06d
// CHECK:STDOUT:   %.loc27_10.4 => constants.%.374
// CHECK:STDOUT:   %impl.elem0.loc27_10.2 => constants.%C.as.Copy.impl.Op.46c
// CHECK:STDOUT:   %specific_impl_fn.loc27_10.2 => constants.%C.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness => constants.%custom_witness.b80ef6.5
// CHECK:STDOUT:   %SubobjectDestroy.facet => constants.%SubobjectDestroy.facet.3b5
// CHECK:STDOUT:   %.loc26_32.13 => constants.%.4ec
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness => constants.%DestroyImpl.impl_witness.b88
// CHECK:STDOUT:   %Destroy.facet.loc26_32.3 => constants.%Destroy.facet.455
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.546
// CHECK:STDOUT: }
// CHECK:STDOUT:
