// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %MakeSelf.type: type = fn_type @MakeSelf, @Class(%T) [symbolic]
// CHECK:STDOUT:   %MakeSelf: %MakeSelf.type = struct_value () [symbolic]
// CHECK:STDOUT:   %MakeClass.type: type = fn_type @MakeClass, @Class(%T) [symbolic]
// CHECK:STDOUT:   %MakeClass: %MakeClass.type = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %Class.val: %Class = struct_value () [symbolic]
// CHECK:STDOUT:   %MakeSelf.specific_fn: <specific function> = specific_function %MakeSelf, @MakeSelf(%T) [symbolic]
// CHECK:STDOUT:   %MakeClass.specific_fn: <specific function> = specific_function %MakeClass, @MakeClass(%T) [symbolic]
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
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc11_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc11_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T.loc11_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %MakeSelf.type: type = fn_type @MakeSelf, @Class(%T.loc11_13.2) [symbolic = %MakeSelf.type (constants.%MakeSelf.type)]
// CHECK:STDOUT:   %MakeSelf: @Class.%MakeSelf.type (%MakeSelf.type) = struct_value () [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:   %MakeClass.type: type = fn_type @MakeClass, @Class(%T.loc11_13.2) [symbolic = %MakeClass.type (constants.%MakeClass.type)]
// CHECK:STDOUT:   %MakeClass: @Class.%MakeClass.type (%MakeClass.type) = struct_value () [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T.loc11_13.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @Class.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %MakeSelf.decl: @Class.%MakeSelf.type (%MakeSelf.type) = fn_decl @MakeSelf [symbolic = @Class.%MakeSelf (constants.%MakeSelf)] {
// CHECK:STDOUT:       %return.patt: @MakeSelf.%Class (%Class) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @MakeSelf.%Class (%Class) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc14_20: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc14_20 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       %return.param: ref @MakeSelf.%Class (%Class) = out_param call_param0
// CHECK:STDOUT:       %return: ref @MakeSelf.%Class (%Class) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %MakeClass.decl: @Class.%MakeClass.type (%MakeClass.type) = fn_decl @MakeClass [symbolic = @Class.%MakeClass (constants.%MakeClass)] {
// CHECK:STDOUT:       %return.patt: @MakeClass.%Class.loc15_28.1 (%Class) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @MakeClass.%Class.loc15_28.1 (%Class) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc11_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Class.loc15_28.2: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc15_28.1 (constants.%Class)]
// CHECK:STDOUT:       %return.param: ref @MakeClass.%Class.loc15_28.1 (%Class) = out_param call_param0
// CHECK:STDOUT:       %return: ref @MakeClass.%Class.loc15_28.1 (%Class) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.decl: @Class.%F.type (%F.type) = fn_decl @F [symbolic = @Class.%F (constants.%F)] {} {}
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
// CHECK:STDOUT: generic fn @MakeSelf(@Class.%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @MakeSelf.%Class (%Class) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Class.val: @MakeSelf.%Class (%Class) = struct_value () [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param_patt: @MakeSelf.%Class (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc14_35.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc14_35.2: init @MakeSelf.%Class (%Class) = class_init (), %return [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     %.loc14_36: init @MakeSelf.%Class (%Class) = converted %.loc14_35.1, %.loc14_35.2 [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     return %.loc14_36 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MakeClass(@Class.%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class.loc15_28.1: type = class_type @Class, @Class(%T) [symbolic = %Class.loc15_28.1 (constants.%Class)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @MakeClass.%Class.loc15_28.1 (%Class) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Class.val: @MakeClass.%Class.loc15_28.1 (%Class) = struct_value () [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param_patt: @MakeClass.%Class.loc15_28.1 (%Class) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc15_40.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc15_40.2: init @MakeClass.%Class.loc15_28.1 (%Class) = class_init (), %return [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     %.loc15_41: init @MakeClass.%Class.loc15_28.1 (%Class) = converted %.loc15_40.1, %.loc15_40.2 [symbolic = %Class.val (constants.%Class.val)]
// CHECK:STDOUT:     return %.loc15_41 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Class.%T.loc11_13.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class.loc17_19.2: type = class_type @Class, @Class(%T) [symbolic = %Class.loc17_19.2 (constants.%Class)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%Class.loc17_19.2 (%Class) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %MakeSelf.type: type = fn_type @MakeSelf, @Class(%T) [symbolic = %MakeSelf.type (constants.%MakeSelf.type)]
// CHECK:STDOUT:   %MakeSelf: @F.%MakeSelf.type (%MakeSelf.type) = struct_value () [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:   %MakeSelf.specific_fn.loc17_23.2: <specific function> = specific_function %MakeSelf, @MakeSelf(%T) [symbolic = %MakeSelf.specific_fn.loc17_23.2 (constants.%MakeSelf.specific_fn)]
// CHECK:STDOUT:   %MakeClass.type: type = fn_type @MakeClass, @Class(%T) [symbolic = %MakeClass.type (constants.%MakeClass.type)]
// CHECK:STDOUT:   %MakeClass: @F.%MakeClass.type (%MakeClass.type) = struct_value () [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:   %MakeClass.specific_fn.loc18_19.2: <specific function> = specific_function %MakeClass, @MakeClass(%T) [symbolic = %MakeClass.specific_fn.loc18_19.2 (constants.%MakeClass.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %c.patt: @F.%Class.loc17_19.2 (%Class) = binding_pattern c
// CHECK:STDOUT:       %.loc17_5.1: @F.%Class.loc17_19.2 (%Class) = var_pattern %c.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.var: ref @F.%Class.loc17_19.2 (%Class) = var c
// CHECK:STDOUT:     %.loc17_23: @F.%MakeSelf.type (%MakeSelf.type) = specific_constant @Class.%MakeSelf.decl, @Class(constants.%T) [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:     %MakeSelf.ref: @F.%MakeSelf.type (%MakeSelf.type) = name_ref MakeSelf, %.loc17_23 [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:     %MakeSelf.specific_fn.loc17_23.1: <specific function> = specific_function %MakeSelf.ref, @MakeSelf(constants.%T) [symbolic = %MakeSelf.specific_fn.loc17_23.2 (constants.%MakeSelf.specific_fn)]
// CHECK:STDOUT:     %.loc17_5.2: ref @F.%Class.loc17_19.2 (%Class) = splice_block %c.var {}
// CHECK:STDOUT:     %MakeSelf.call: init @F.%Class.loc17_19.2 (%Class) = call %MakeSelf.specific_fn.loc17_23.1() to %.loc17_5.2
// CHECK:STDOUT:     assign %c.var, %MakeSelf.call
// CHECK:STDOUT:     %.loc17_19: type = splice_block %Class.loc17_19.1 [symbolic = %Class.loc17_19.2 (constants.%Class)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc11_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Class.loc17_19.1: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc17_19.2 (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: ref @F.%Class.loc17_19.2 (%Class) = bind_name c, %c.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %s.patt: @F.%Class.loc17_19.2 (%Class) = binding_pattern s
// CHECK:STDOUT:       %.loc18_5.1: @F.%Class.loc17_19.2 (%Class) = var_pattern %s.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s.var: ref @F.%Class.loc17_19.2 (%Class) = var s
// CHECK:STDOUT:     %.loc18_19: @F.%MakeClass.type (%MakeClass.type) = specific_constant @Class.%MakeClass.decl, @Class(constants.%T) [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:     %MakeClass.ref: @F.%MakeClass.type (%MakeClass.type) = name_ref MakeClass, %.loc18_19 [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:     %MakeClass.specific_fn.loc18_19.1: <specific function> = specific_function %MakeClass.ref, @MakeClass(constants.%T) [symbolic = %MakeClass.specific_fn.loc18_19.2 (constants.%MakeClass.specific_fn)]
// CHECK:STDOUT:     %.loc18_5.2: ref @F.%Class.loc17_19.2 (%Class) = splice_block %s.var {}
// CHECK:STDOUT:     %MakeClass.call: init @F.%Class.loc17_19.2 (%Class) = call %MakeClass.specific_fn.loc18_19.1() to %.loc18_5.2
// CHECK:STDOUT:     assign %s.var, %MakeClass.call
// CHECK:STDOUT:     %.loc18_12.1: type = splice_block %Self.ref [symbolic = %Class.loc17_19.2 (constants.%Class)] {
// CHECK:STDOUT:       %.loc18_12.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class.loc17_19.2 (constants.%Class)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc18_12.2 [symbolic = %Class.loc17_19.2 (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %s: ref @F.%Class.loc17_19.2 (%Class) = bind_name s, %s.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T.patt
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
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %Class.val => constants.%Class.val
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@MakeSelf.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class.loc15_28.1 => constants.%Class
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %Class.val => constants.%Class.val
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@MakeClass.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc11_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@F.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeSelf(@F.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeClass(@F.%T) {}
// CHECK:STDOUT:
