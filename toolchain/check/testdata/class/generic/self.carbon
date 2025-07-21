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

class Class(T:! type) {
  // `Self` is the same as `Class(T)` here.
  // TODO: Find a better way to test two types are the same.
  fn MakeSelf() -> Self { return {}; }
  fn MakeClass() -> Class(T) { return {}; }
  fn F() {
    var c: Class(T) = MakeSelf();
    var s: Self = MakeClass();
  }
}

// CHECK:STDOUT: --- self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.3c1: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %MakeSelf.type: type = fn_type @MakeSelf, @Class(%T) [symbolic]
// CHECK:STDOUT:   %MakeSelf: %MakeSelf.type = struct_value () [symbolic]
// CHECK:STDOUT:   %MakeClass.type: type = fn_type @MakeClass, @Class(%T) [symbolic]
// CHECK:STDOUT:   %MakeClass: %MakeClass.type = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4f8: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %Class.val: %Class = struct_value () [symbolic]
// CHECK:STDOUT:   %MakeSelf.specific_fn: <specific function> = specific_function %MakeSelf, @MakeSelf(%T) [symbolic]
// CHECK:STDOUT:   %MakeClass.specific_fn: <specific function> = specific_function %MakeClass, @MakeClass(%T) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %ptr.955: type = ptr_type %Class [symbolic]
// CHECK:STDOUT:   %require_complete.2ae: <witness> = require_complete_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Class, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Class, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.d76: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.d76 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Op.1(%Destroy.facet) [symbolic]
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
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc15_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T.loc15_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %MakeSelf.type: type = fn_type @MakeSelf, @Class(%T.loc15_13.1) [symbolic = %MakeSelf.type (constants.%MakeSelf.type)]
// CHECK:STDOUT:   %MakeSelf: @Class.%MakeSelf.type (%MakeSelf.type) = struct_value () [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:   %MakeClass.type: type = fn_type @MakeClass, @Class(%T.loc15_13.1) [symbolic = %MakeClass.type (constants.%MakeClass.type)]
// CHECK:STDOUT:   %MakeClass: @Class.%MakeClass.type (%MakeClass.type) = struct_value () [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T.loc15_13.1) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @Class.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %MakeSelf.decl: @Class.%MakeSelf.type (%MakeSelf.type) = fn_decl @MakeSelf [symbolic = @Class.%MakeSelf (constants.%MakeSelf)] {
// CHECK:STDOUT:       %return.patt: @MakeSelf.%pattern_type (%pattern_type.3c1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @MakeSelf.%pattern_type (%pattern_type.3c1) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc18_20: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc18_20 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       %return.param: ref @MakeSelf.%Class (%Class) = out_param call_param0
// CHECK:STDOUT:       %return: ref @MakeSelf.%Class (%Class) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %MakeClass.decl: @Class.%MakeClass.type (%MakeClass.type) = fn_decl @MakeClass [symbolic = @Class.%MakeClass (constants.%MakeClass)] {
// CHECK:STDOUT:       %return.patt: @MakeClass.%pattern_type (%pattern_type.3c1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @MakeClass.%pattern_type (%pattern_type.3c1) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc15_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Class.loc19_28.1: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc19_28.2 (constants.%Class)]
// CHECK:STDOUT:       %return.param: ref @MakeClass.%Class.loc19_28.2 (%Class) = out_param call_param0
// CHECK:STDOUT:       %return: ref @MakeClass.%Class.loc19_28.2 (%Class) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.decl: @Class.%F.type (%F.type) = fn_decl @F [symbolic = @Class.%F (constants.%F)] {} {}
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .MakeSelf = %MakeSelf.decl
// CHECK:STDOUT:     .Class = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .MakeClass = %MakeClass.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MakeSelf(@Class.%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Class [symbolic = %pattern_type (constants.%pattern_type.3c1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class [symbolic = %require_complete (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.val: @MakeSelf.%Class (%Class) = struct_value () [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @MakeSelf.%Class (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc18_35.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc18_35.2: init @MakeSelf.%Class (%Class) = class_init (), %return [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     %.loc18_36: init @MakeSelf.%Class (%Class) = converted %.loc18_35.1, %.loc18_35.2 [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     return %.loc18_36 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MakeClass(@Class.%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class.loc19_28.2: type = class_type @Class, @Class(%T) [symbolic = %Class.loc19_28.2 (constants.%Class)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Class.loc19_28.2 [symbolic = %pattern_type (constants.%pattern_type.3c1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class.loc19_28.2 [symbolic = %require_complete (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.val: @MakeClass.%Class.loc19_28.2 (%Class) = struct_value () [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @MakeClass.%Class.loc19_28.2 (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc19_40.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc19_40.2: init @MakeClass.%Class.loc19_28.2 (%Class) = class_init (), %return [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     %.loc19_41: init @MakeClass.%Class.loc19_28.2 (%Class) = converted %.loc19_40.1, %.loc19_40.2 [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     return %.loc19_41 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Class.%T.loc15_13.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class.loc21_19.2: type = class_type @Class, @Class(%T) [symbolic = %Class.loc21_19.2 (constants.%Class)]
// CHECK:STDOUT:   %require_complete.loc21: <witness> = require_complete_type %Class.loc21_19.2 [symbolic = %require_complete.loc21 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Class.loc21_19.2 [symbolic = %pattern_type (constants.%pattern_type.3c1)]
// CHECK:STDOUT:   %MakeSelf.type: type = fn_type @MakeSelf, @Class(%T) [symbolic = %MakeSelf.type (constants.%MakeSelf.type)]
// CHECK:STDOUT:   %MakeSelf: @F.%MakeSelf.type (%MakeSelf.type) = struct_value () [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:   %MakeSelf.specific_fn.loc21_23.2: <specific function> = specific_function %MakeSelf, @MakeSelf(%T) [symbolic = %MakeSelf.specific_fn.loc21_23.2 (constants.%MakeSelf.specific_fn)]
// CHECK:STDOUT:   %MakeClass.type: type = fn_type @MakeClass, @Class(%T) [symbolic = %MakeClass.type (constants.%MakeClass.type)]
// CHECK:STDOUT:   %MakeClass: @F.%MakeClass.type (%MakeClass.type) = struct_value () [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:   %MakeClass.specific_fn.loc22_19.2: <specific function> = specific_function %MakeClass, @MakeClass(%T) [symbolic = %MakeClass.specific_fn.loc22_19.2 (constants.%MakeClass.specific_fn)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Class.loc21_19.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Class.loc21_19.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %.loc22_5.4: type = fn_type_with_self_type constants.%Op.type.bae, %Destroy.facet [symbolic = %.loc22_5.4 (constants.%.d76)]
// CHECK:STDOUT:   %impl.elem0.loc22_5.3: @F.%.loc22_5.4 (%.d76) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_5.3 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_5.3: <specific function> = specific_impl_function %impl.elem0.loc22_5.3, @Op.1(%Destroy.facet) [symbolic = %specific_impl_fn.loc22_5.3 (constants.%specific_impl_fn)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class.loc21_19.2 [symbolic = %ptr (constants.%ptr.955)]
// CHECK:STDOUT:   %require_complete.loc22: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc22 (constants.%require_complete.2ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %c.patt: @F.%pattern_type (%pattern_type.3c1) = binding_pattern c [concrete]
// CHECK:STDOUT:       %c.var_patt: @F.%pattern_type (%pattern_type.3c1) = var_pattern %c.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.var: ref @F.%Class.loc21_19.2 (%Class) = var %c.var_patt
// CHECK:STDOUT:     %.loc21_23: @F.%MakeSelf.type (%MakeSelf.type) = specific_constant @Class.%MakeSelf.decl, @Class(constants.%T) [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:     %MakeSelf.ref: @F.%MakeSelf.type (%MakeSelf.type) = name_ref MakeSelf, %.loc21_23 [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:     %MakeSelf.specific_fn.loc21_23.1: <specific function> = specific_function %MakeSelf.ref, @MakeSelf(constants.%T) [symbolic = %MakeSelf.specific_fn.loc21_23.2 (constants.%MakeSelf.specific_fn)]
// CHECK:STDOUT:     %.loc21_5.1: ref @F.%Class.loc21_19.2 (%Class) = splice_block %c.var {}
// CHECK:STDOUT:     %MakeSelf.call: init @F.%Class.loc21_19.2 (%Class) = call %MakeSelf.specific_fn.loc21_23.1() to %.loc21_5.1
// CHECK:STDOUT:     assign %c.var, %MakeSelf.call
// CHECK:STDOUT:     %.loc21_19: type = splice_block %Class.loc21_19.1 [symbolic = %Class.loc21_19.2 (constants.%Class)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc15_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Class.loc21_19.1: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc21_19.2 (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: ref @F.%Class.loc21_19.2 (%Class) = bind_name c, %c.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %s.patt: @F.%pattern_type (%pattern_type.3c1) = binding_pattern s [concrete]
// CHECK:STDOUT:       %s.var_patt: @F.%pattern_type (%pattern_type.3c1) = var_pattern %s.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s.var: ref @F.%Class.loc21_19.2 (%Class) = var %s.var_patt
// CHECK:STDOUT:     %.loc22_19: @F.%MakeClass.type (%MakeClass.type) = specific_constant @Class.%MakeClass.decl, @Class(constants.%T) [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:     %MakeClass.ref: @F.%MakeClass.type (%MakeClass.type) = name_ref MakeClass, %.loc22_19 [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:     %MakeClass.specific_fn.loc22_19.1: <specific function> = specific_function %MakeClass.ref, @MakeClass(constants.%T) [symbolic = %MakeClass.specific_fn.loc22_19.2 (constants.%MakeClass.specific_fn)]
// CHECK:STDOUT:     %.loc22_5.1: ref @F.%Class.loc21_19.2 (%Class) = splice_block %s.var {}
// CHECK:STDOUT:     %MakeClass.call: init @F.%Class.loc21_19.2 (%Class) = call %MakeClass.specific_fn.loc22_19.1() to %.loc22_5.1
// CHECK:STDOUT:     assign %s.var, %MakeClass.call
// CHECK:STDOUT:     %.loc22_12.1: type = splice_block %Self.ref [symbolic = %Class.loc21_19.2 (constants.%Class)] {
// CHECK:STDOUT:       %.loc22_12.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class.loc21_19.2 (constants.%Class)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc22_12.2 [symbolic = %Class.loc21_19.2 (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s: ref @F.%Class.loc21_19.2 (%Class) = bind_name s, %s.var
// CHECK:STDOUT:     %impl.elem0.loc22_5.1: @F.%.loc22_5.4 (%.d76) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_5.3 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc22_5.1: <bound method> = bound_method %.loc22_5.1, %impl.elem0.loc22_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc22_5.1: <specific function> = specific_impl_function %impl.elem0.loc22_5.1, @Op.1(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc22_5.3 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc22_5.2: <bound method> = bound_method %.loc22_5.1, %specific_impl_fn.loc22_5.1
// CHECK:STDOUT:     %addr.loc22_5.1: @F.%ptr (%ptr.955) = addr_of %.loc22_5.1
// CHECK:STDOUT:     %.loc22_5.2: init %empty_tuple.type = call %bound_method.loc22_5.2(%addr.loc22_5.1)
// CHECK:STDOUT:     %impl.elem0.loc22_5.2: @F.%.loc22_5.4 (%.d76) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_5.3 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc22_5.3: <bound method> = bound_method %s.var, %impl.elem0.loc22_5.2
// CHECK:STDOUT:     %specific_impl_fn.loc22_5.2: <specific function> = specific_impl_function %impl.elem0.loc22_5.2, @Op.1(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc22_5.3 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc22_5.4: <bound method> = bound_method %s.var, %specific_impl_fn.loc22_5.2
// CHECK:STDOUT:     %addr.loc22_5.2: @F.%ptr (%ptr.955) = addr_of %s.var
// CHECK:STDOUT:     %.loc22_5.3: init %empty_tuple.type = call %bound_method.loc22_5.4(%addr.loc22_5.2)
// CHECK:STDOUT:     %impl.elem0.loc21_5.1: @F.%.loc22_5.4 (%.d76) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_5.3 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc21_5.1: <bound method> = bound_method %.loc21_5.1, %impl.elem0.loc21_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc21_5.1: <specific function> = specific_impl_function %impl.elem0.loc21_5.1, @Op.1(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc22_5.3 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc21_5.2: <bound method> = bound_method %.loc21_5.1, %specific_impl_fn.loc21_5.1
// CHECK:STDOUT:     %addr.loc21_5.1: @F.%ptr (%ptr.955) = addr_of %.loc21_5.1
// CHECK:STDOUT:     %.loc21_5.2: init %empty_tuple.type = call %bound_method.loc21_5.2(%addr.loc21_5.1)
// CHECK:STDOUT:     %impl.elem0.loc21_5.2: @F.%.loc22_5.4 (%.d76) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_5.3 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc21_5.3: <bound method> = bound_method %c.var, %impl.elem0.loc21_5.2
// CHECK:STDOUT:     %specific_impl_fn.loc21_5.2: <specific function> = specific_impl_function %impl.elem0.loc21_5.2, @Op.1(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc22_5.3 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc21_5.4: <bound method> = bound_method %c.var, %specific_impl_fn.loc21_5.2
// CHECK:STDOUT:     %addr.loc21_5.2: @F.%ptr (%ptr.955) = addr_of %c.var
// CHECK:STDOUT:     %.loc21_5.3: init %empty_tuple.type = call %bound_method.loc21_5.4(%addr.loc21_5.2)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %MakeSelf.type => constants.%MakeSelf.type
// CHECK:STDOUT:   %MakeSelf => constants.%MakeSelf
// CHECK:STDOUT:   %MakeClass.type => constants.%MakeClass.type
// CHECK:STDOUT:   %MakeClass => constants.%MakeClass
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeSelf(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3c1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4f8
// CHECK:STDOUT:   %Class.val => constants.%Class.val
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class.loc19_28.2 => constants.%Class
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3c1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4f8
// CHECK:STDOUT:   %Class.val => constants.%Class.val
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {}
// CHECK:STDOUT:
