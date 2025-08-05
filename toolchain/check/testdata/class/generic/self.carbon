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
// CHECK:STDOUT:   %Class.MakeSelf.type: type = fn_type @Class.MakeSelf, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.MakeSelf: %Class.MakeSelf.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Class.MakeClass.type: type = fn_type @Class.MakeClass, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.MakeClass: %Class.MakeClass.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.95a: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.955: type = ptr_type %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.9e0: type = pattern_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: %Class.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Class, (%Destroy.impl_witness.95a) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4f8: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %Class.val: %Class = struct_value () [symbolic]
// CHECK:STDOUT:   %Class.MakeSelf.specific_fn: <specific function> = specific_function %Class.MakeSelf, @Class.MakeSelf(%T) [symbolic]
// CHECK:STDOUT:   %Class.MakeClass.specific_fn: <specific function> = specific_function %Class.MakeClass, @Class.MakeClass(%T) [symbolic]
// CHECK:STDOUT:   %.02c: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %Class.as.Destroy.impl.Op, @Class.as.Destroy.impl.Op(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.2ae: <witness> = require_complete_type %ptr.955 [symbolic]
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
// CHECK:STDOUT: generic impl @Class.as.Destroy.impl(@Class.%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.95a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic = %Class.as.Destroy.impl.Op.type (constants.%Class.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: constants.%Class as constants.%Destroy.type {
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.decl: @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = fn_decl @Class.as.Destroy.impl.Op [symbolic = @Class.as.Destroy.impl.%Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.as.Destroy.impl.Op.%pattern_type (%pattern_type.9e0) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc15_23.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955) = value_param call_param0
// CHECK:STDOUT:       %.loc15_23.2: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc15_23.3: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc15_23.3 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.as.Destroy.impl.Op.%ptr (%ptr.955) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Class.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Class.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T.loc15_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.MakeSelf.type: type = fn_type @Class.MakeSelf, @Class(%T.loc15_13.1) [symbolic = %Class.MakeSelf.type (constants.%Class.MakeSelf.type)]
// CHECK:STDOUT:   %Class.MakeSelf: @Class.%Class.MakeSelf.type (%Class.MakeSelf.type) = struct_value () [symbolic = %Class.MakeSelf (constants.%Class.MakeSelf)]
// CHECK:STDOUT:   %Class.MakeClass.type: type = fn_type @Class.MakeClass, @Class(%T.loc15_13.1) [symbolic = %Class.MakeClass.type (constants.%Class.MakeClass.type)]
// CHECK:STDOUT:   %Class.MakeClass: @Class.%Class.MakeClass.type (%Class.MakeClass.type) = struct_value () [symbolic = %Class.MakeClass (constants.%Class.MakeClass)]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.loc15_13.1) [symbolic = %Class.F.type (constants.%Class.F.type)]
// CHECK:STDOUT:   %Class.F: @Class.%Class.F.type (%Class.F.type) = struct_value () [symbolic = %Class.F (constants.%Class.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.MakeSelf.decl: @Class.%Class.MakeSelf.type (%Class.MakeSelf.type) = fn_decl @Class.MakeSelf [symbolic = @Class.%Class.MakeSelf (constants.%Class.MakeSelf)] {
// CHECK:STDOUT:       %return.patt: @Class.MakeSelf.%pattern_type (%pattern_type.3c1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.MakeSelf.%pattern_type (%pattern_type.3c1) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc18_20: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc18_20 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       %return.param: ref @Class.MakeSelf.%Class (%Class) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Class.MakeSelf.%Class (%Class) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.MakeClass.decl: @Class.%Class.MakeClass.type (%Class.MakeClass.type) = fn_decl @Class.MakeClass [symbolic = @Class.%Class.MakeClass (constants.%Class.MakeClass)] {
// CHECK:STDOUT:       %return.patt: @Class.MakeClass.%pattern_type (%pattern_type.3c1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.MakeClass.%pattern_type (%pattern_type.3c1) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc15_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Class.loc19_28.2: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc19_28.1 (constants.%Class)]
// CHECK:STDOUT:       %return.param: ref @Class.MakeClass.%Class.loc19_28.1 (%Class) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Class.MakeClass.%Class.loc19_28.1 (%Class) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.F.decl: @Class.%Class.F.type (%Class.F.type) = fn_decl @Class.F [symbolic = @Class.%Class.F (constants.%Class.F)] {} {}
// CHECK:STDOUT:     impl_decl @Class.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.decl), @Class.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Class.as.Destroy.impl(constants.%T) [symbolic = @Class.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.95a)]
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT: generic fn @Class.MakeSelf(@Class.%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Class [symbolic = %pattern_type (constants.%pattern_type.3c1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class [symbolic = %require_complete (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.val: @Class.MakeSelf.%Class (%Class) = struct_value () [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @Class.MakeSelf.%Class (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc18_35.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc18_35.2: init @Class.MakeSelf.%Class (%Class) = class_init (), %return [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     %.loc18_36: init @Class.MakeSelf.%Class (%Class) = converted %.loc18_35.1, %.loc18_35.2 [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     return %.loc18_36 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.MakeClass(@Class.%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class.loc19_28.1: type = class_type @Class, @Class(%T) [symbolic = %Class.loc19_28.1 (constants.%Class)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Class.loc19_28.1 [symbolic = %pattern_type (constants.%pattern_type.3c1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class.loc19_28.1 [symbolic = %require_complete (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.val: @Class.MakeClass.%Class.loc19_28.1 (%Class) = struct_value () [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @Class.MakeClass.%Class.loc19_28.1 (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc19_40.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc19_40.2: init @Class.MakeClass.%Class.loc19_28.1 (%Class) = class_init (), %return [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     %.loc19_41: init @Class.MakeClass.%Class.loc19_28.1 (%Class) = converted %.loc19_40.1, %.loc19_40.2 [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     return %.loc19_41 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.F(@Class.%T.loc15_13.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class.loc21_19.2: type = class_type @Class, @Class(%T) [symbolic = %Class.loc21_19.2 (constants.%Class)]
// CHECK:STDOUT:   %require_complete.loc21: <witness> = require_complete_type %Class.loc21_19.2 [symbolic = %require_complete.loc21 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Class.loc21_19.2 [symbolic = %pattern_type (constants.%pattern_type.3c1)]
// CHECK:STDOUT:   %Class.MakeSelf.type: type = fn_type @Class.MakeSelf, @Class(%T) [symbolic = %Class.MakeSelf.type (constants.%Class.MakeSelf.type)]
// CHECK:STDOUT:   %Class.MakeSelf: @Class.F.%Class.MakeSelf.type (%Class.MakeSelf.type) = struct_value () [symbolic = %Class.MakeSelf (constants.%Class.MakeSelf)]
// CHECK:STDOUT:   %Class.MakeSelf.specific_fn.loc21_23.2: <specific function> = specific_function %Class.MakeSelf, @Class.MakeSelf(%T) [symbolic = %Class.MakeSelf.specific_fn.loc21_23.2 (constants.%Class.MakeSelf.specific_fn)]
// CHECK:STDOUT:   %Class.MakeClass.type: type = fn_type @Class.MakeClass, @Class(%T) [symbolic = %Class.MakeClass.type (constants.%Class.MakeClass.type)]
// CHECK:STDOUT:   %Class.MakeClass: @Class.F.%Class.MakeClass.type (%Class.MakeClass.type) = struct_value () [symbolic = %Class.MakeClass (constants.%Class.MakeClass)]
// CHECK:STDOUT:   %Class.MakeClass.specific_fn.loc22_19.2: <specific function> = specific_function %Class.MakeClass, @Class.MakeClass(%T) [symbolic = %Class.MakeClass.specific_fn.loc22_19.2 (constants.%Class.MakeClass.specific_fn)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.95a)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Class.loc21_19.2, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %.loc22_5.2: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc22_5.2 (constants.%.02c)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T) [symbolic = %Class.as.Destroy.impl.Op.type (constants.%Class.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: @Class.F.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %Class.as.Destroy.impl.Op, @Class.as.Destroy.impl.Op(%T) [symbolic = %Class.as.Destroy.impl.Op.specific_fn (constants.%Class.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class.loc21_19.2 [symbolic = %ptr (constants.%ptr.955)]
// CHECK:STDOUT:   %require_complete.loc22: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc22 (constants.%require_complete.2ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %c.patt: @Class.F.%pattern_type (%pattern_type.3c1) = binding_pattern c [concrete]
// CHECK:STDOUT:       %c.var_patt: @Class.F.%pattern_type (%pattern_type.3c1) = var_pattern %c.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.var: ref @Class.F.%Class.loc21_19.2 (%Class) = var %c.var_patt
// CHECK:STDOUT:     %.loc21_23: @Class.F.%Class.MakeSelf.type (%Class.MakeSelf.type) = specific_constant @Class.%Class.MakeSelf.decl, @Class(constants.%T) [symbolic = %Class.MakeSelf (constants.%Class.MakeSelf)]
// CHECK:STDOUT:     %MakeSelf.ref: @Class.F.%Class.MakeSelf.type (%Class.MakeSelf.type) = name_ref MakeSelf, %.loc21_23 [symbolic = %Class.MakeSelf (constants.%Class.MakeSelf)]
// CHECK:STDOUT:     %Class.MakeSelf.specific_fn.loc21_23.1: <specific function> = specific_function %MakeSelf.ref, @Class.MakeSelf(constants.%T) [symbolic = %Class.MakeSelf.specific_fn.loc21_23.2 (constants.%Class.MakeSelf.specific_fn)]
// CHECK:STDOUT:     %.loc21_5: ref @Class.F.%Class.loc21_19.2 (%Class) = splice_block %c.var {}
// CHECK:STDOUT:     %Class.MakeSelf.call: init @Class.F.%Class.loc21_19.2 (%Class) = call %Class.MakeSelf.specific_fn.loc21_23.1() to %.loc21_5
// CHECK:STDOUT:     assign %c.var, %Class.MakeSelf.call
// CHECK:STDOUT:     %.loc21_19: type = splice_block %Class.loc21_19.1 [symbolic = %Class.loc21_19.2 (constants.%Class)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc15_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Class.loc21_19.1: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc21_19.2 (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: ref @Class.F.%Class.loc21_19.2 (%Class) = bind_name c, %c.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %s.patt: @Class.F.%pattern_type (%pattern_type.3c1) = binding_pattern s [concrete]
// CHECK:STDOUT:       %s.var_patt: @Class.F.%pattern_type (%pattern_type.3c1) = var_pattern %s.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s.var: ref @Class.F.%Class.loc21_19.2 (%Class) = var %s.var_patt
// CHECK:STDOUT:     %.loc22_19: @Class.F.%Class.MakeClass.type (%Class.MakeClass.type) = specific_constant @Class.%Class.MakeClass.decl, @Class(constants.%T) [symbolic = %Class.MakeClass (constants.%Class.MakeClass)]
// CHECK:STDOUT:     %MakeClass.ref: @Class.F.%Class.MakeClass.type (%Class.MakeClass.type) = name_ref MakeClass, %.loc22_19 [symbolic = %Class.MakeClass (constants.%Class.MakeClass)]
// CHECK:STDOUT:     %Class.MakeClass.specific_fn.loc22_19.1: <specific function> = specific_function %MakeClass.ref, @Class.MakeClass(constants.%T) [symbolic = %Class.MakeClass.specific_fn.loc22_19.2 (constants.%Class.MakeClass.specific_fn)]
// CHECK:STDOUT:     %.loc22_5.1: ref @Class.F.%Class.loc21_19.2 (%Class) = splice_block %s.var {}
// CHECK:STDOUT:     %Class.MakeClass.call: init @Class.F.%Class.loc21_19.2 (%Class) = call %Class.MakeClass.specific_fn.loc22_19.1() to %.loc22_5.1
// CHECK:STDOUT:     assign %s.var, %Class.MakeClass.call
// CHECK:STDOUT:     %.loc22_12.1: type = splice_block %Self.ref [symbolic = %Class.loc21_19.2 (constants.%Class)] {
// CHECK:STDOUT:       %.loc22_12.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class.loc21_19.2 (constants.%Class)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc22_12.2 [symbolic = %Class.loc21_19.2 (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s: ref @Class.F.%Class.loc21_19.2 (%Class) = bind_name s, %s.var
// CHECK:STDOUT:     %impl.elem0.loc22_5.1: @Class.F.%.loc22_5.2 (%.02c) = impl_witness_access constants.%Destroy.impl_witness.95a, element0 [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:     %bound_method.loc22_5.1: <bound method> = bound_method %.loc22_5.1, %impl.elem0.loc22_5.1
// CHECK:STDOUT:     %specific_fn.loc22_5.1: <specific function> = specific_function %impl.elem0.loc22_5.1, @Class.as.Destroy.impl.Op(constants.%T) [symbolic = %Class.as.Destroy.impl.Op.specific_fn (constants.%Class.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:     %bound_method.loc22_5.2: <bound method> = bound_method %.loc22_5.1, %specific_fn.loc22_5.1
// CHECK:STDOUT:     %addr.loc22_5.1: @Class.F.%ptr (%ptr.955) = addr_of %.loc22_5.1
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.call.loc22_5.1: init %empty_tuple.type = call %bound_method.loc22_5.2(%addr.loc22_5.1)
// CHECK:STDOUT:     %impl.elem0.loc22_5.2: @Class.F.%.loc22_5.2 (%.02c) = impl_witness_access constants.%Destroy.impl_witness.95a, element0 [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:     %bound_method.loc22_5.3: <bound method> = bound_method %s.var, %impl.elem0.loc22_5.2
// CHECK:STDOUT:     %specific_fn.loc22_5.2: <specific function> = specific_function %impl.elem0.loc22_5.2, @Class.as.Destroy.impl.Op(constants.%T) [symbolic = %Class.as.Destroy.impl.Op.specific_fn (constants.%Class.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:     %bound_method.loc22_5.4: <bound method> = bound_method %s.var, %specific_fn.loc22_5.2
// CHECK:STDOUT:     %addr.loc22_5.2: @Class.F.%ptr (%ptr.955) = addr_of %s.var
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.call.loc22_5.2: init %empty_tuple.type = call %bound_method.loc22_5.4(%addr.loc22_5.2)
// CHECK:STDOUT:     %impl.elem0.loc21_5.1: @Class.F.%.loc22_5.2 (%.02c) = impl_witness_access constants.%Destroy.impl_witness.95a, element0 [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:     %bound_method.loc21_5.1: <bound method> = bound_method %.loc21_5, %impl.elem0.loc21_5.1
// CHECK:STDOUT:     %specific_fn.loc21_5.1: <specific function> = specific_function %impl.elem0.loc21_5.1, @Class.as.Destroy.impl.Op(constants.%T) [symbolic = %Class.as.Destroy.impl.Op.specific_fn (constants.%Class.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:     %bound_method.loc21_5.2: <bound method> = bound_method %.loc21_5, %specific_fn.loc21_5.1
// CHECK:STDOUT:     %addr.loc21_5.1: @Class.F.%ptr (%ptr.955) = addr_of %.loc21_5
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.call.loc21_5.1: init %empty_tuple.type = call %bound_method.loc21_5.2(%addr.loc21_5.1)
// CHECK:STDOUT:     %impl.elem0.loc21_5.2: @Class.F.%.loc22_5.2 (%.02c) = impl_witness_access constants.%Destroy.impl_witness.95a, element0 [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op)]
// CHECK:STDOUT:     %bound_method.loc21_5.3: <bound method> = bound_method %c.var, %impl.elem0.loc21_5.2
// CHECK:STDOUT:     %specific_fn.loc21_5.2: <specific function> = specific_function %impl.elem0.loc21_5.2, @Class.as.Destroy.impl.Op(constants.%T) [symbolic = %Class.as.Destroy.impl.Op.specific_fn (constants.%Class.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:     %bound_method.loc21_5.4: <bound method> = bound_method %c.var, %specific_fn.loc21_5.2
// CHECK:STDOUT:     %addr.loc21_5.2: @Class.F.%ptr (%ptr.955) = addr_of %c.var
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.call.loc21_5.2: init %empty_tuple.type = call %bound_method.loc21_5.4(%addr.loc21_5.2)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.as.Destroy.impl.Op(@Class.%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class [symbolic = %ptr (constants.%ptr.955)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.9e0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.as.Destroy.impl.Op.%ptr (%ptr.955)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%T
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
// CHECK:STDOUT: specific @Class.MakeSelf(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3c1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4f8
// CHECK:STDOUT:   %Class.val => constants.%Class.val
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.MakeClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class.loc19_28.1 => constants.%Class
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3c1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4f8
// CHECK:STDOUT:   %Class.val => constants.%Class.val
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.95a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type => constants.%Class.as.Destroy.impl.Op.type
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op => constants.%Class.as.Destroy.impl.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %ptr => constants.%ptr.955
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
