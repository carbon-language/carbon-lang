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
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %MakeSelf.type: type = fn_type @MakeSelf, @Class(%T) [symbolic]
// CHECK:STDOUT:   %MakeSelf: %MakeSelf.type = struct_value () [symbolic]
// CHECK:STDOUT:   %MakeClass.type: type = fn_type @MakeClass, @Class(%T) [symbolic]
// CHECK:STDOUT:   %MakeClass: %MakeClass.type = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %struct: %Class.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: <specific function> = specific_function %MakeSelf, @MakeSelf(%T) [symbolic]
// CHECK:STDOUT:   %.5: <specific function> = specific_function %MakeClass, @MakeClass(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.patt.loc11_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_13.1, runtime_param<invalid> [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_13.2 (constants.%T)]
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
// CHECK:STDOUT:       %return.patt: @MakeSelf.%Class (%Class.2) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @MakeSelf.%Class (%Class.2) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc14_20: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc14_20 [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:       %return.param: ref @MakeSelf.%Class (%Class.2) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @MakeSelf.%Class (%Class.2) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %MakeClass.decl: @Class.%MakeClass.type (%MakeClass.type) = fn_decl @MakeClass [symbolic = @Class.%MakeClass (constants.%MakeClass)] {
// CHECK:STDOUT:       %return.patt: @MakeClass.%Class.loc15_26.1 (%Class.2) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @MakeClass.%Class.loc15_26.1 (%Class.2) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc11_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %Class.loc15_26.2: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc15_26.1 (constants.%Class.2)]
// CHECK:STDOUT:       %return.param: ref @MakeClass.%Class.loc15_26.1 (%Class.2) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @MakeClass.%Class.loc15_26.1 (%Class.2) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.decl: @Class.%F.type (%F.type) = fn_decl @F [symbolic = @Class.%F (constants.%F)] {} {}
// CHECK:STDOUT:     %.loc20: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .MakeSelf = %MakeSelf.decl
// CHECK:STDOUT:     .MakeClass = %MakeClass.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MakeSelf(@Class.%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %struct: @MakeSelf.%Class (%Class.2) = struct_value () [symbolic = %struct (constants.%struct)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return: @MakeSelf.%Class (%Class.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc14_35.1: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc14_35.2: init @MakeSelf.%Class (%Class.2) = class_init (), %return [symbolic = %struct (constants.%struct)]
// CHECK:STDOUT:     %.loc14_36: init @MakeSelf.%Class (%Class.2) = converted %.loc14_35.1, %.loc14_35.2 [symbolic = %struct (constants.%struct)]
// CHECK:STDOUT:     return %.loc14_36 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MakeClass(@Class.%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class.loc15_26.1: type = class_type @Class, @Class(%T) [symbolic = %Class.loc15_26.1 (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %struct: @MakeClass.%Class.loc15_26.1 (%Class.2) = struct_value () [symbolic = %struct (constants.%struct)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return: @MakeClass.%Class.loc15_26.1 (%Class.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc15_40.1: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc15_40.2: init @MakeClass.%Class.loc15_26.1 (%Class.2) = class_init (), %return [symbolic = %struct (constants.%struct)]
// CHECK:STDOUT:     %.loc15_41: init @MakeClass.%Class.loc15_26.1 (%Class.2) = converted %.loc15_40.1, %.loc15_40.2 [symbolic = %struct (constants.%struct)]
// CHECK:STDOUT:     return %.loc15_41 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Class.%T.loc11_13.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class.loc17_17.2: type = class_type @Class, @Class(%T) [symbolic = %Class.loc17_17.2 (constants.%Class.2)]
// CHECK:STDOUT:   %MakeSelf.type: type = fn_type @MakeSelf, @Class(%T) [symbolic = %MakeSelf.type (constants.%MakeSelf.type)]
// CHECK:STDOUT:   %MakeSelf: @F.%MakeSelf.type (%MakeSelf.type) = struct_value () [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:   %.loc17_23.3: <specific function> = specific_function %MakeSelf, @MakeSelf(%T) [symbolic = %.loc17_23.3 (constants.%.4)]
// CHECK:STDOUT:   %MakeClass.type: type = fn_type @MakeClass, @Class(%T) [symbolic = %MakeClass.type (constants.%MakeClass.type)]
// CHECK:STDOUT:   %MakeClass: @F.%MakeClass.type (%MakeClass.type) = struct_value () [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:   %.loc18_19.3: <specific function> = specific_function %MakeClass, @MakeClass(%T) [symbolic = %.loc18_19.3 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, @Class.%T.loc11_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %Class.loc17_17.1: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.loc17_17.2 (constants.%Class.2)]
// CHECK:STDOUT:     %c.var: ref @F.%Class.loc17_17.2 (%Class.2) = var c
// CHECK:STDOUT:     %c: ref @F.%Class.loc17_17.2 (%Class.2) = bind_name c, %c.var
// CHECK:STDOUT:     %.loc17_23.1: @F.%MakeSelf.type (%MakeSelf.type) = specific_constant @Class.%MakeSelf.decl, @Class(constants.%T) [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:     %MakeSelf.ref: @F.%MakeSelf.type (%MakeSelf.type) = name_ref MakeSelf, %.loc17_23.1 [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:     %.loc17_23.2: <specific function> = specific_function %MakeSelf.ref, @MakeSelf(constants.%T) [symbolic = %.loc17_23.3 (constants.%.4)]
// CHECK:STDOUT:     %.loc17_9: ref @F.%Class.loc17_17.2 (%Class.2) = splice_block %c.var {}
// CHECK:STDOUT:     %MakeSelf.call: init @F.%Class.loc17_17.2 (%Class.2) = call %.loc17_23.2() to %.loc17_9
// CHECK:STDOUT:     assign %c.var, %MakeSelf.call
// CHECK:STDOUT:     %.loc18_12: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = %Class.loc17_17.2 (constants.%Class.2)]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, %.loc18_12 [symbolic = %Class.loc17_17.2 (constants.%Class.2)]
// CHECK:STDOUT:     %s.var: ref @F.%Class.loc17_17.2 (%Class.2) = var s
// CHECK:STDOUT:     %s: ref @F.%Class.loc17_17.2 (%Class.2) = bind_name s, %s.var
// CHECK:STDOUT:     %.loc18_19.1: @F.%MakeClass.type (%MakeClass.type) = specific_constant @Class.%MakeClass.decl, @Class(constants.%T) [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:     %MakeClass.ref: @F.%MakeClass.type (%MakeClass.type) = name_ref MakeClass, %.loc18_19.1 [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:     %.loc18_19.2: <specific function> = specific_function %MakeClass.ref, @MakeClass(constants.%T) [symbolic = %.loc18_19.3 (constants.%.5)]
// CHECK:STDOUT:     %.loc18_9: ref @F.%Class.loc17_17.2 (%Class.2) = splice_block %s.var {}
// CHECK:STDOUT:     %MakeClass.call: init @F.%Class.loc17_17.2 (%Class.2) = call %.loc18_19.2() to %.loc18_9
// CHECK:STDOUT:     assign %s.var, %MakeClass.call
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
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
// CHECK:STDOUT: specific @Class(@MakeSelf.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeSelf(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %struct => constants.%struct
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@MakeClass.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class.loc15_26.1 => constants.%Class.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %struct => constants.%struct
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc11_13.2) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@F.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeSelf(@F.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeClass(@F.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class.loc15_26.1 => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
