// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/self.carbon

class Class(T: type) {
  // `Self` is the same as `Class(T)` here.
  // TODO: Find a better way to test two types are the same.
  fn MakeSelf() -> Self { return {}; }
  fn MakeClass() -> Class(T) { return {}; }
  fn F() {
    var unused c: Class(T) = MakeSelf();
    var unused s: Self = MakeClass();
  }
}


// CHECK:STDOUT: --- self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = parameterized_class_name_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.67d) [symbolic]
// CHECK:STDOUT:   %.917: Core.Form = init_form %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.28f: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.28f = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt: %pattern_type.28f = return_slot_pattern %return.param_patt, %Class [symbolic]
// CHECK:STDOUT:   %Class.MakeSelf.type: type = fn_type @Class.MakeSelf, @Class(%T.67d) [symbolic]
// CHECK:STDOUT:   %Class.MakeSelf: %Class.MakeSelf.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Class.MakeClass.type: type = fn_type @Class.MakeClass, @Class(%T.67d) [symbolic]
// CHECK:STDOUT:   %Class.MakeClass: %Class.MakeClass.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.67d) [symbolic]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.ac0: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.val: %Class = struct_value () [symbolic]
// CHECK:STDOUT:   %c.patt: %pattern_type.28f = ref_binding_pattern c [symbolic]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.28f = var_pattern %c.patt [symbolic]
// CHECK:STDOUT:   %Class.MakeSelf.specific_fn: <specific function> = specific_function %Class.MakeSelf, @Class.MakeSelf(%T.67d) [symbolic]
// CHECK:STDOUT:   %s.patt: %pattern_type.28f = ref_binding_pattern s [symbolic]
// CHECK:STDOUT:   %s.var_patt: %pattern_type.28f = var_pattern %s.patt [symbolic]
// CHECK:STDOUT:   %Class.MakeClass.specific_fn: <specific function> = specific_function %Class.MakeClass, @Class.MakeClass(%T.67d) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.cc4: <witness> = lookup_impl_witness %Class, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.bb5: <witness> = lookup_impl_witness %Class, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b2a: %SubobjectDestroy.type = facet_value %Class, (%SubobjectDestroy.lookup_impl_witness.bb5) [symbolic]
// CHECK:STDOUT:   %.6e3: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b2a) [symbolic]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Class, (%DestroyImpl.lookup_impl_witness.cc4, %SubobjectDestroy.lookup_impl_witness.bb5) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc15_14.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_14.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_16.1: type = splice_block %.loc15_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc15_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_14.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc15_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc15_14.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_14.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T.loc15_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.MakeSelf.type: type = fn_type @Class.MakeSelf, @Class(%T.loc15_14.1) [symbolic = %Class.MakeSelf.type (constants.%Class.MakeSelf.type)]
// CHECK:STDOUT:   %Class.MakeSelf: @Class.%Class.MakeSelf.type (%Class.MakeSelf.type) = struct_value () [symbolic = %Class.MakeSelf (constants.%Class.MakeSelf)]
// CHECK:STDOUT:   %Class.MakeClass.type: type = fn_type @Class.MakeClass, @Class(%T.loc15_14.1) [symbolic = %Class.MakeClass.type (constants.%Class.MakeClass.type)]
// CHECK:STDOUT:   %Class.MakeClass: @Class.%Class.MakeClass.type (%Class.MakeClass.type) = struct_value () [symbolic = %Class.MakeClass (constants.%Class.MakeClass)]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.loc15_14.1) [symbolic = %Class.F.type (constants.%Class.F.type)]
// CHECK:STDOUT:   %Class.F: @Class.%Class.F.type (%Class.F.type) = struct_value () [symbolic = %Class.F (constants.%Class.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.MakeSelf.decl: @Class.%Class.MakeSelf.type (%Class.MakeSelf.type) = fn_decl @Class.MakeSelf [symbolic = @Class.%Class.MakeSelf (constants.%Class.MakeSelf)] {
// CHECK:STDOUT:       %return.param_patt.loc18_20.1: @Class.MakeSelf.%pattern_type (%pattern_type.28f) = out_param_pattern [symbolic = %return.param_patt.loc18_20.2 (constants.%return.param_patt)]
// CHECK:STDOUT:       %return.patt.loc18_17.1: @Class.MakeSelf.%pattern_type (%pattern_type.28f) = return_slot_pattern %return.param_patt.loc18_20.1, %Self.ref [symbolic = %return.patt.loc18_17.2 (constants.%return.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc18_20.2: type = specific_constant constants.%Class, @Class(constants.%T.67d) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc18_20.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       %.loc18_20.3: Core.Form = init_form %Self.ref [symbolic = %.loc18_20.1 (constants.%.917)]
// CHECK:STDOUT:       %return.param: ref @Class.MakeSelf.%Class (%Class) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Class.MakeSelf.%Class (%Class) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.MakeClass.decl: @Class.%Class.MakeClass.type (%Class.MakeClass.type) = fn_decl @Class.MakeClass [symbolic = @Class.%Class.MakeClass (constants.%Class.MakeClass)] {
// CHECK:STDOUT:       %return.param_patt.loc19_28.1: @Class.MakeClass.%pattern_type (%pattern_type.28f) = out_param_pattern [symbolic = %return.param_patt.loc19_28.2 (constants.%return.param_patt)]
// CHECK:STDOUT:       %return.patt.loc19_18.1: @Class.MakeClass.%pattern_type (%pattern_type.28f) = return_slot_pattern %return.param_patt.loc19_28.1, %Class.loc19_28.2 [symbolic = %return.patt.loc19_18.2 (constants.%return.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc15_14.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %Class.loc19_28.2: type = class_type @Class, @Class(constants.%T.67d) [symbolic = %Class.loc19_28.1 (constants.%Class)]
// CHECK:STDOUT:       %.loc19_28.2: Core.Form = init_form %Class.loc19_28.2 [symbolic = %.loc19_28.1 (constants.%.917)]
// CHECK:STDOUT:       %return.param: ref @Class.MakeClass.%Class.loc19_28.1 (%Class) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Class.MakeClass.%Class.loc19_28.1 (%Class) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.F.decl: @Class.%Class.F.type (%Class.F.type) = fn_decl @Class.F [symbolic = @Class.%Class.F (constants.%Class.F)] {} {}
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .MakeSelf = %Class.MakeSelf.decl
// CHECK:STDOUT:     .Class = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .MakeClass = %Class.MakeClass.decl
// CHECK:STDOUT:     .F = %Class.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.MakeSelf(@Class.%T.loc15_14.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %.loc18_20.1: Core.Form = init_form %Class [symbolic = %.loc18_20.1 (constants.%.917)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Class [symbolic = %pattern_type (constants.%pattern_type.28f)]
// CHECK:STDOUT:   %return.param_patt.loc18_20.2: @Class.MakeSelf.%pattern_type (%pattern_type.28f) = out_param_pattern [symbolic = %return.param_patt.loc18_20.2 (constants.%return.param_patt)]
// CHECK:STDOUT:   %return.patt.loc18_17.2: @Class.MakeSelf.%pattern_type (%pattern_type.28f) = return_slot_pattern %return.param_patt.loc18_20.2, %Class [symbolic = %return.patt.loc18_17.2 (constants.%return.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class [symbolic = %require_complete (constants.%require_complete.ac0)]
// CHECK:STDOUT:   %Class.val: @Class.MakeSelf.%Class (%Class) = struct_value () [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Class.MakeSelf.%Class (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc18_35.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc18_35.2: init @Class.MakeSelf.%Class (%Class) to %return.param = class_init () [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     %.loc18_36: init @Class.MakeSelf.%Class (%Class) = converted %.loc18_35.1, %.loc18_35.2 [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     return %.loc18_36 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.MakeClass(@Class.%T.loc15_14.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %Class.loc19_28.1: type = class_type @Class, @Class(%T) [symbolic = %Class.loc19_28.1 (constants.%Class)]
// CHECK:STDOUT:   %.loc19_28.1: Core.Form = init_form %Class.loc19_28.1 [symbolic = %.loc19_28.1 (constants.%.917)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Class.loc19_28.1 [symbolic = %pattern_type (constants.%pattern_type.28f)]
// CHECK:STDOUT:   %return.param_patt.loc19_28.2: @Class.MakeClass.%pattern_type (%pattern_type.28f) = out_param_pattern [symbolic = %return.param_patt.loc19_28.2 (constants.%return.param_patt)]
// CHECK:STDOUT:   %return.patt.loc19_18.2: @Class.MakeClass.%pattern_type (%pattern_type.28f) = return_slot_pattern %return.param_patt.loc19_28.2, %Class.loc19_28.1 [symbolic = %return.patt.loc19_18.2 (constants.%return.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class.loc19_28.1 [symbolic = %require_complete (constants.%require_complete.ac0)]
// CHECK:STDOUT:   %Class.val: @Class.MakeClass.%Class.loc19_28.1 (%Class) = struct_value () [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Class.MakeClass.%Class.loc19_28.1 (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc19_40.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc19_40.2: init @Class.MakeClass.%Class.loc19_28.1 (%Class) to %return.param = class_init () [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     %.loc19_41: init @Class.MakeClass.%Class.loc19_28.1 (%Class) = converted %.loc19_40.1, %.loc19_40.2 [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     return %.loc19_41 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.F(@Class.%T.loc15_14.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %Class.loc21_26.2: type = class_type @Class, @Class(%T) [symbolic = %Class.loc21_26.2 (constants.%Class)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class.loc21_26.2 [symbolic = %require_complete (constants.%require_complete.ac0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Class.loc21_26.2 [symbolic = %pattern_type (constants.%pattern_type.28f)]
// CHECK:STDOUT:   %c.patt.loc21_17.2: @Class.F.%pattern_type (%pattern_type.28f) = ref_binding_pattern c [symbolic = %c.patt.loc21_17.2 (constants.%c.patt)]
// CHECK:STDOUT:   %c.var_patt.loc21_5.2: @Class.F.%pattern_type (%pattern_type.28f) = var_pattern %c.patt.loc21_17.2 [symbolic = %c.var_patt.loc21_5.2 (constants.%c.var_patt)]
// CHECK:STDOUT:   %Class.MakeSelf.type: type = fn_type @Class.MakeSelf, @Class(%T) [symbolic = %Class.MakeSelf.type (constants.%Class.MakeSelf.type)]
// CHECK:STDOUT:   %Class.MakeSelf: @Class.F.%Class.MakeSelf.type (%Class.MakeSelf.type) = struct_value () [symbolic = %Class.MakeSelf (constants.%Class.MakeSelf)]
// CHECK:STDOUT:   %Class.MakeSelf.specific_fn.loc21_30.2: <specific function> = specific_function %Class.MakeSelf, @Class.MakeSelf(%T) [symbolic = %Class.MakeSelf.specific_fn.loc21_30.2 (constants.%Class.MakeSelf.specific_fn)]
// CHECK:STDOUT:   %s.patt.loc22_17.2: @Class.F.%pattern_type (%pattern_type.28f) = ref_binding_pattern s [symbolic = %s.patt.loc22_17.2 (constants.%s.patt)]
// CHECK:STDOUT:   %s.var_patt.loc22_5.2: @Class.F.%pattern_type (%pattern_type.28f) = var_pattern %s.patt.loc22_17.2 [symbolic = %s.var_patt.loc22_5.2 (constants.%s.var_patt)]
// CHECK:STDOUT:   %Class.MakeClass.type: type = fn_type @Class.MakeClass, @Class(%T) [symbolic = %Class.MakeClass.type (constants.%Class.MakeClass.type)]
// CHECK:STDOUT:   %Class.MakeClass: @Class.F.%Class.MakeClass.type (%Class.MakeClass.type) = struct_value () [symbolic = %Class.MakeClass (constants.%Class.MakeClass)]
// CHECK:STDOUT:   %Class.MakeClass.specific_fn.loc22_26.2: <specific function> = specific_function %Class.MakeClass, @Class.MakeClass(%T) [symbolic = %Class.MakeClass.specific_fn.loc22_26.2 (constants.%Class.MakeClass.specific_fn)]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness: <witness> = lookup_impl_witness %Class.loc21_26.2, @SubobjectDestroy [symbolic = %SubobjectDestroy.lookup_impl_witness (constants.%SubobjectDestroy.lookup_impl_witness.bb5)]
// CHECK:STDOUT:   %SubobjectDestroy.facet: %SubobjectDestroy.type = facet_value %Class.loc21_26.2, (%SubobjectDestroy.lookup_impl_witness) [symbolic = %SubobjectDestroy.facet (constants.%SubobjectDestroy.facet.b2a)]
// CHECK:STDOUT:   %.loc22_5.4: require_specific_def_type = require_specific_def @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet) [symbolic = %.loc22_5.4 (constants.%.6e3)]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness: <witness> = lookup_impl_witness %Class.loc21_26.2, @DestroyImpl [symbolic = %DestroyImpl.lookup_impl_witness (constants.%DestroyImpl.lookup_impl_witness.cc4)]
// CHECK:STDOUT:   %facet_value.loc22_5.3: %facet_type = facet_value %Class.loc21_26.2, (%DestroyImpl.lookup_impl_witness, %SubobjectDestroy.lookup_impl_witness) [symbolic = %facet_value.loc22_5.3 (constants.%facet_value)]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(%facet_value.loc22_5.3) [symbolic = %SelfDestruct.specific_fn.3 (constants.%SelfDestruct.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.var: ref @Class.F.%Class.loc21_26.2 (%Class) = var_storage %c.var_patt.loc21_5.1
// CHECK:STDOUT:     %.loc21_30: @Class.F.%Class.MakeSelf.type (%Class.MakeSelf.type) = specific_constant @Class.%Class.MakeSelf.decl, @Class(constants.%T.67d) [symbolic = %Class.MakeSelf (constants.%Class.MakeSelf)]
// CHECK:STDOUT:     %MakeSelf.ref: @Class.F.%Class.MakeSelf.type (%Class.MakeSelf.type) = name_ref MakeSelf, %.loc21_30 [symbolic = %Class.MakeSelf (constants.%Class.MakeSelf)]
// CHECK:STDOUT:     %Class.MakeSelf.specific_fn.loc21_30.1: <specific function> = specific_function %MakeSelf.ref, @Class.MakeSelf(constants.%T.67d) [symbolic = %Class.MakeSelf.specific_fn.loc21_30.2 (constants.%Class.MakeSelf.specific_fn)]
// CHECK:STDOUT:     %.loc21_5.1: ref @Class.F.%Class.loc21_26.2 (%Class) = splice_block %c.var {}
// CHECK:STDOUT:     %Class.MakeSelf.call: init @Class.F.%Class.loc21_26.2 (%Class) to %.loc21_5.1 = call %Class.MakeSelf.specific_fn.loc21_30.1()
// CHECK:STDOUT:     assign %c.var, %Class.MakeSelf.call
// CHECK:STDOUT:     %.loc21_26: type = splice_block %Class.loc21_26.1 [symbolic = %Class.loc21_26.2 (constants.%Class)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc15_14.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %Class.loc21_26.1: type = class_type @Class, @Class(constants.%T.67d) [symbolic = %Class.loc21_26.2 (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: ref @Class.F.%Class.loc21_26.2 (%Class) = wrapper_binding c, %c.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %c.patt.loc21_17.1: @Class.F.%pattern_type (%pattern_type.28f) = ref_binding_pattern c [symbolic = %c.patt.loc21_17.2 (constants.%c.patt)]
// CHECK:STDOUT:       %c.var_patt.loc21_5.1: @Class.F.%pattern_type (%pattern_type.28f) = var_pattern %c.patt.loc21_17.1 [symbolic = %c.var_patt.loc21_5.2 (constants.%c.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s.var: ref @Class.F.%Class.loc21_26.2 (%Class) = var_storage %s.var_patt.loc22_5.1
// CHECK:STDOUT:     %.loc22_26: @Class.F.%Class.MakeClass.type (%Class.MakeClass.type) = specific_constant @Class.%Class.MakeClass.decl, @Class(constants.%T.67d) [symbolic = %Class.MakeClass (constants.%Class.MakeClass)]
// CHECK:STDOUT:     %MakeClass.ref: @Class.F.%Class.MakeClass.type (%Class.MakeClass.type) = name_ref MakeClass, %.loc22_26 [symbolic = %Class.MakeClass (constants.%Class.MakeClass)]
// CHECK:STDOUT:     %Class.MakeClass.specific_fn.loc22_26.1: <specific function> = specific_function %MakeClass.ref, @Class.MakeClass(constants.%T.67d) [symbolic = %Class.MakeClass.specific_fn.loc22_26.2 (constants.%Class.MakeClass.specific_fn)]
// CHECK:STDOUT:     %.loc22_5.1: ref @Class.F.%Class.loc21_26.2 (%Class) = splice_block %s.var {}
// CHECK:STDOUT:     %Class.MakeClass.call: init @Class.F.%Class.loc21_26.2 (%Class) to %.loc22_5.1 = call %Class.MakeClass.specific_fn.loc22_26.1()
// CHECK:STDOUT:     assign %s.var, %Class.MakeClass.call
// CHECK:STDOUT:     %.loc22_19.1: type = splice_block %Self.ref [symbolic = %Class.loc21_26.2 (constants.%Class)] {
// CHECK:STDOUT:       %.loc22_19.2: type = specific_constant constants.%Class, @Class(constants.%T.67d) [symbolic = %Class.loc21_26.2 (constants.%Class)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc22_19.2 [symbolic = %Class.loc21_26.2 (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s: ref @Class.F.%Class.loc21_26.2 (%Class) = wrapper_binding s, %s.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %s.patt.loc22_17.1: @Class.F.%pattern_type (%pattern_type.28f) = ref_binding_pattern s [symbolic = %s.patt.loc22_17.2 (constants.%s.patt)]
// CHECK:STDOUT:       %s.var_patt.loc22_5.1: @Class.F.%pattern_type (%pattern_type.28f) = var_pattern %s.patt.loc22_17.1 [symbolic = %s.var_patt.loc22_5.2 (constants.%s.var_patt)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %facet_value.loc22_5.1: %facet_type = facet_value constants.%Class, (constants.%DestroyImpl.lookup_impl_witness.cc4, constants.%SubobjectDestroy.lookup_impl_witness.bb5) [symbolic = %facet_value.loc22_5.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc22_5.2: %facet_type = converted constants.%Class, %facet_value.loc22_5.1 [symbolic = %facet_value.loc22_5.3 (constants.%facet_value)]
// CHECK:STDOUT:     %facet_value.loc22_5.2: %facet_type = facet_value constants.%Class, (constants.%DestroyImpl.lookup_impl_witness.cc4, constants.%SubobjectDestroy.lookup_impl_witness.bb5) [symbolic = %facet_value.loc22_5.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc22_5.3: %facet_type = converted constants.%Class, %facet_value.loc22_5.2 [symbolic = %facet_value.loc22_5.3 (constants.%facet_value)]
// CHECK:STDOUT:     %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value) [symbolic = %SelfDestruct.specific_fn.3 (constants.%SelfDestruct.specific_fn)]
// CHECK:STDOUT:     %SelfDestruct.call.loc22: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%s.var)
// CHECK:STDOUT:     %facet_value.loc21_5.1: %facet_type = facet_value constants.%Class, (constants.%DestroyImpl.lookup_impl_witness.cc4, constants.%SubobjectDestroy.lookup_impl_witness.bb5) [symbolic = %facet_value.loc22_5.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc21_5.2: %facet_type = converted constants.%Class, %facet_value.loc21_5.1 [symbolic = %facet_value.loc22_5.3 (constants.%facet_value)]
// CHECK:STDOUT:     %facet_value.loc21_5.2: %facet_type = facet_value constants.%Class, (constants.%DestroyImpl.lookup_impl_witness.cc4, constants.%SubobjectDestroy.lookup_impl_witness.bb5) [symbolic = %facet_value.loc22_5.3 (constants.%facet_value)]
// CHECK:STDOUT:     %.loc21_5.3: %facet_type = converted constants.%Class, %facet_value.loc21_5.2 [symbolic = %facet_value.loc22_5.3 (constants.%facet_value)]
// CHECK:STDOUT:     %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value) [symbolic = %SelfDestruct.specific_fn.3 (constants.%SelfDestruct.specific_fn)]
// CHECK:STDOUT:     %SelfDestruct.call.loc21: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%c.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc22(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc15_14.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc15_14.1 => constants.%T.67d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.MakeSelf.type => constants.%Class.MakeSelf.type
// CHECK:STDOUT:   %Class.MakeSelf => constants.%Class.MakeSelf
// CHECK:STDOUT:   %Class.MakeClass.type => constants.%Class.MakeClass.type
// CHECK:STDOUT:   %Class.MakeClass => constants.%Class.MakeClass
// CHECK:STDOUT:   %Class.F.type => constants.%Class.F.type
// CHECK:STDOUT:   %Class.F => constants.%Class.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.MakeSelf(constants.%T.67d) {
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %.loc18_20.1 => constants.%.917
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.28f
// CHECK:STDOUT:   %return.param_patt.loc18_20.2 => constants.%return.param_patt
// CHECK:STDOUT:   %return.patt.loc18_17.2 => constants.%return.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ac0
// CHECK:STDOUT:   %Class.val => constants.%Class.val
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.MakeClass(constants.%T.67d) {
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %Class.loc19_28.1 => constants.%Class
// CHECK:STDOUT:   %.loc19_28.1 => constants.%.917
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.28f
// CHECK:STDOUT:   %return.param_patt.loc19_28.2 => constants.%return.param_patt
// CHECK:STDOUT:   %return.patt.loc19_18.2 => constants.%return.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ac0
// CHECK:STDOUT:   %Class.val => constants.%Class.val
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F(constants.%T.67d) {}
// CHECK:STDOUT:
