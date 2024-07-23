// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/subst_into_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/subst_into_decl.carbon

class Class(T:! type) {
  var x: T;

  fn Get[self: Self]() -> T { return self.x; }
}

fn F(x: Class(i32)) -> i32 {
  return x.x;
}

fn G(x: Class(i32)) -> i32 {
  return x.Get();
}

// CHECK:STDOUT: --- subst_into_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %Get.type.1: type = fn_type @Get, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Get.1: %Get.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %Get.2: %Get.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, @Class(i32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %Class.3, i32 [template]
// CHECK:STDOUT:   %Get.type.2: type = fn_type @Get, @Class(i32) [template]
// CHECK:STDOUT:   %Get.3: %Get.type.2 = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.3: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.4: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Core: <namespace> = namespace %Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc11_13.1: type = param T
// CHECK:STDOUT:     %T.loc11_13.2: type = bind_symbolic_name T 0, %T.loc11_13.1 [symbolic = @Class.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Class.ref.loc17: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %int.make_type_32.loc17_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc17_14.1: type = value_of_initializer %int.make_type_32.loc17_15 [template = i32]
// CHECK:STDOUT:     %.loc17_14.2: type = converted %int.make_type_32.loc17_15, %.loc17_14.1 [template = i32]
// CHECK:STDOUT:     %.loc17_14.3: init type = call %Class.ref.loc17(%.loc17_14.2) [template = constants.%Class.3]
// CHECK:STDOUT:     %.loc17_18.1: type = value_of_initializer %.loc17_14.3 [template = constants.%Class.3]
// CHECK:STDOUT:     %.loc17_18.2: type = converted %.loc17_14.3, %.loc17_18.1 [template = constants.%Class.3]
// CHECK:STDOUT:     %x.loc17_6.1: %Class.3 = param x
// CHECK:STDOUT:     @F.%x: %Class.3 = bind_name x, %x.loc17_6.1
// CHECK:STDOUT:     %int.make_type_32.loc17_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc17_24.1: type = value_of_initializer %int.make_type_32.loc17_24 [template = i32]
// CHECK:STDOUT:     %.loc17_24.2: type = converted %int.make_type_32.loc17_24, %.loc17_24.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %Class.ref.loc21: %Class.type = name_ref Class, %Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:     %int.make_type_32.loc21_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_14.1: type = value_of_initializer %int.make_type_32.loc21_15 [template = i32]
// CHECK:STDOUT:     %.loc21_14.2: type = converted %int.make_type_32.loc21_15, %.loc21_14.1 [template = i32]
// CHECK:STDOUT:     %.loc21_14.3: init type = call %Class.ref.loc21(%.loc21_14.2) [template = constants.%Class.3]
// CHECK:STDOUT:     %.loc21_18.1: type = value_of_initializer %.loc21_14.3 [template = constants.%Class.3]
// CHECK:STDOUT:     %.loc21_18.2: type = converted %.loc21_14.3, %.loc21_18.1 [template = constants.%Class.3]
// CHECK:STDOUT:     %x.loc21_6.1: %Class.3 = param x
// CHECK:STDOUT:     @G.%x: %Class.3 = bind_name x, %x.loc21_6.1
// CHECK:STDOUT:     %int.make_type_32.loc21_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_24.1: type = value_of_initializer %int.make_type_32.loc21_24 [template = i32]
// CHECK:STDOUT:     %.loc21_24.2: type = converted %int.make_type_32.loc21_24, %.loc21_24.1 [template = i32]
// CHECK:STDOUT:     @G.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(file.%T.loc11_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %Get.type: type = fn_type @Get, @Class(%T) [symbolic = %Get.type (constants.%Get.type.1)]
// CHECK:STDOUT:   %Get: @Class.%Get.type (%Get.type.1) = struct_value () [symbolic = %Get (constants.%Get.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref.loc12: type = name_ref T, file.%T.loc11_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %.loc12: @Class.%.1 (%.2) = field_decl x, element0 [template]
// CHECK:STDOUT:     %Get.decl: @Class.%Get.type (%Get.type.1) = fn_decl @Get [symbolic = %Get (constants.%Get.1)] {
// CHECK:STDOUT:       %.loc14: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = @Get.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc14 [symbolic = @Get.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %self.loc14_10.1: @Get.%Class (%Class.2) = param self
// CHECK:STDOUT:       %self.loc14_10.2: @Get.%Class (%Class.2) = bind_name self, %self.loc14_10.1
// CHECK:STDOUT:       %T.ref.loc14: type = name_ref T, file.%T.loc11_13.2 [symbolic = @Get.%T (constants.%T)]
// CHECK:STDOUT:       %return.var: ref @Get.%T (%T) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .x = %.loc12
// CHECK:STDOUT:     .Get = %Get.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Get(file.%T.loc11_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = unbound_element_type @Get.%Class (%Class.2), @Get.%T (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[@Class.%self.loc14_10.2: @Get.%Class (%Class.2)]() -> @Get.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Get.%Class (%Class.2) = name_ref self, @Class.%self.loc14_10.2
// CHECK:STDOUT:     %x.ref: @Get.%.1 (%.2) = name_ref x, @Class.%.loc12 [template = @Class.%.loc12]
// CHECK:STDOUT:     %.loc14_42.1: ref @Get.%T (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc14_42.2: @Get.%T (%T) = bind_value %.loc14_42.1
// CHECK:STDOUT:     return %.loc14_42.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x: %Class.3) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc18_10: %Class.3 = name_ref x, %x
// CHECK:STDOUT:   %x.ref.loc18_11: %.5 = name_ref x, @Class.%.loc12 [template = @Class.%.loc12]
// CHECK:STDOUT:   %.loc18_11.1: ref i32 = class_element_access %x.ref.loc18_10, element0
// CHECK:STDOUT:   %.loc18_11.2: i32 = bind_value %.loc18_11.1
// CHECK:STDOUT:   return %.loc18_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x: %Class.3) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %Class.3 = name_ref x, %x
// CHECK:STDOUT:   %.loc22_11.1: %Get.type.2 = specific_constant @Class.%Get.decl, @Class(i32) [template = constants.%Get.3]
// CHECK:STDOUT:   %Get.ref: %Get.type.2 = name_ref Get, %.loc22_11.1 [template = constants.%Get.3]
// CHECK:STDOUT:   %.loc22_11.2: <bound method> = bound_method %x.ref, %Get.ref
// CHECK:STDOUT:   %Get.call: init i32 = call %.loc22_11.2(%x.ref)
// CHECK:STDOUT:   %.loc22_17.1: i32 = value_of_initializer %Get.call
// CHECK:STDOUT:   %.loc22_17.2: i32 = converted %Get.call, %.loc22_17.1
// CHECK:STDOUT:   return %.loc22_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT:   %Get.type => constants.%Get.type.1
// CHECK:STDOUT:   %Get => constants.%Get.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Get.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Class.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(i32) {
// CHECK:STDOUT:   %T => i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.3
// CHECK:STDOUT:   %.1 => constants.%.5
// CHECK:STDOUT:   %Get.type => constants.%Get.type.2
// CHECK:STDOUT:   %Get => constants.%Get.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(i32) {
// CHECK:STDOUT:   %T => i32
// CHECK:STDOUT:   %Class => constants.%Class.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
