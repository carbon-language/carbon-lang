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
  fn MakeSelf() -> Self;
  fn MakeClass() -> Class(T);
  fn F() {
    var c: Class(T) = MakeSelf();
    var s: Self = MakeClass();
  }
}

// CHECK:STDOUT: --- self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %MakeSelf.type: type = fn_type @MakeSelf, @Class(%T) [symbolic]
// CHECK:STDOUT:   %MakeSelf: %MakeSelf.type = struct_value () [symbolic]
// CHECK:STDOUT:   %MakeClass.type: type = fn_type @MakeClass, @Class(%T) [symbolic]
// CHECK:STDOUT:   %MakeClass: %MakeClass.type = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
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
// CHECK:STDOUT:     %T.loc11_13.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_13.2: type = bind_symbolic_name T 0, %T.loc11_13.1 [symbolic = @Class.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(file.%T.loc11_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %MakeSelf.type: type = fn_type @MakeSelf, @Class(%T) [symbolic = %MakeSelf.type (constants.%MakeSelf.type)]
// CHECK:STDOUT:   %MakeSelf: @Class.%MakeSelf.type (%MakeSelf.type) = struct_value () [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:   %MakeClass.type: type = fn_type @MakeClass, @Class(%T) [symbolic = %MakeClass.type (constants.%MakeClass.type)]
// CHECK:STDOUT:   %MakeClass: @Class.%MakeClass.type (%MakeClass.type) = struct_value () [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @Class.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %MakeSelf.decl: @Class.%MakeSelf.type (%MakeSelf.type) = fn_decl @MakeSelf [symbolic = %MakeSelf (constants.%MakeSelf)] {
// CHECK:STDOUT:       %.loc14: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = @MakeSelf.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc14 [symbolic = @MakeSelf.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %return.var.loc14: ref @MakeSelf.%Class (%Class.2) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %MakeClass.decl: @Class.%MakeClass.type (%MakeClass.type) = fn_decl @MakeClass [symbolic = %MakeClass (constants.%MakeClass)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:       %T.ref: type = name_ref T, file.%T.loc11_13.2 [symbolic = @MakeClass.%T (constants.%T)]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%T) [symbolic = @MakeClass.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %return.var.loc15: ref @MakeClass.%Class (%Class.2) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.decl: @Class.%F.type (%F.type) = fn_decl @F [symbolic = %F (constants.%F)] {}
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .MakeSelf = %MakeSelf.decl
// CHECK:STDOUT:     .MakeClass = %MakeClass.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MakeSelf(file.%T.loc11_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @MakeSelf.%Class (%Class.2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MakeClass(file.%T.loc11_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @MakeClass.%Class (%Class.2);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(file.%T.loc11_13.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class.1: type = class_type @Class, @Class(%T) [symbolic = %Class.1 (constants.%Class.2)]
// CHECK:STDOUT:   %MakeSelf.type: type = fn_type @MakeSelf, @Class(%T) [symbolic = %MakeSelf.type (constants.%MakeSelf.type)]
// CHECK:STDOUT:   %MakeSelf: @F.%MakeSelf.type (%MakeSelf.type) = struct_value () [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:   %MakeClass.type: type = fn_type @MakeClass, @Class(%T) [symbolic = %MakeClass.type (constants.%MakeClass.type)]
// CHECK:STDOUT:   %MakeClass: @F.%MakeClass.type (%MakeClass.type) = struct_value () [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, file.%T.loc11_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %Class.loc17: type = class_type @Class, @Class(constants.%T) [symbolic = %Class.1 (constants.%Class.2)]
// CHECK:STDOUT:     %c.var: ref @F.%Class.1 (%Class.2) = var c
// CHECK:STDOUT:     %c: ref @F.%Class.1 (%Class.2) = bind_name c, %c.var
// CHECK:STDOUT:     %.loc17_23: @F.%MakeSelf.type (%MakeSelf.type) = specific_constant @Class.%MakeSelf.decl, @Class(constants.%T) [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:     %MakeSelf.ref: @F.%MakeSelf.type (%MakeSelf.type) = name_ref MakeSelf, %.loc17_23 [symbolic = %MakeSelf (constants.%MakeSelf)]
// CHECK:STDOUT:     %.loc17_9: ref @F.%Class.1 (%Class.2) = splice_block %c.var {}
// CHECK:STDOUT:     %MakeSelf.call: init @F.%Class.1 (%Class.2) = call %MakeSelf.ref() to %.loc17_9
// CHECK:STDOUT:     assign %c.var, %MakeSelf.call
// CHECK:STDOUT:     %.loc18_12: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = %Class.1 (constants.%Class.2)]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, %.loc18_12 [symbolic = %Class.1 (constants.%Class.2)]
// CHECK:STDOUT:     %s.var: ref @F.%Class.1 (%Class.2) = var s
// CHECK:STDOUT:     %s: ref @F.%Class.1 (%Class.2) = bind_name s, %s.var
// CHECK:STDOUT:     %.loc18_19: @F.%MakeClass.type (%MakeClass.type) = specific_constant @Class.%MakeClass.decl, @Class(constants.%T) [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:     %MakeClass.ref: @F.%MakeClass.type (%MakeClass.type) = name_ref MakeClass, %.loc18_19 [symbolic = %MakeClass (constants.%MakeClass)]
// CHECK:STDOUT:     %.loc18_9: ref @F.%Class.1 (%Class.2) = splice_block %s.var {}
// CHECK:STDOUT:     %MakeClass.call: init @F.%Class.1 (%Class.2) = call %MakeClass.ref() to %.loc18_9
// CHECK:STDOUT:     assign %s.var, %MakeClass.call
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
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
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeSelf(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@MakeClass.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeClass(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Class.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@F.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
