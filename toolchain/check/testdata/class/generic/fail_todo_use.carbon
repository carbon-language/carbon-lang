// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/fail_todo_use.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/fail_todo_use.carbon

class Class(T:! type) {
  fn Get[addr self: Self*]() -> T* {
    return &self->k;
  }

  var k: T;
}

// TODO: The following should work.
fn Run() -> i32 {
  // CHECK:STDERR: fail_todo_use.carbon:[[@LINE+4]]:23: ERROR: Cannot implicitly convert from `i32` to `T`.
  // CHECK:STDERR:   var v: Class(i32) = {.k = 0};
  // CHECK:STDERR:                       ^~~~~~~~
  // CHECK:STDERR:
  var v: Class(i32) = {.k = 0};
  // CHECK:STDERR: fail_todo_use.carbon:[[@LINE+10]]:3: ERROR: Cannot implicitly convert from `T*` to `i32`.
  // CHECK:STDERR:   return v.Get();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_use.carbon:[[@LINE+6]]:10: ERROR: Cannot implicitly convert from `Class*` to `Class*`.
  // CHECK:STDERR:   return v.Get();
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR: fail_todo_use.carbon:[[@LINE-21]]:15: Initializing `addr self` parameter of method declared here.
  // CHECK:STDERR:   fn Get[addr self: Self*]() -> T* {
  // CHECK:STDERR:               ^~~~
  return v.Get();
}

// CHECK:STDOUT: --- fail_todo_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = ptr_type %Class.2 [symbolic]
// CHECK:STDOUT:   %.3: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %Get.type: type = fn_type @Get [template]
// CHECK:STDOUT:   %Get: %Get.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %.5: type = struct_type {.k: %T} [symbolic]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %Class.3: type = class_type @Class, @Class(i32) [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type %Class.3, i32 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.9: type = struct_type {.k: i32} [template]
// CHECK:STDOUT:   %.10: type = ptr_type %Class.3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Run = %Run.decl
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
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc20_13.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc20_13.2: type = converted %int.make_type_32, %.loc20_13.1 [template = i32]
// CHECK:STDOUT:     @Run.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(file.%T.loc11_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T (%T) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Get.decl: %Get.type = fn_decl @Get [template = constants.%Get] {
// CHECK:STDOUT:       %.loc12_21: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = @Get.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc12_21 [symbolic = @Get.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %.loc12_25: type = ptr_type %Class.2 [symbolic = @Get.%.1 (constants.%.2)]
// CHECK:STDOUT:       %self.loc12_15.1: @Get.%.1 (%.2) = param self
// CHECK:STDOUT:       %self.loc12_15.3: @Get.%.1 (%.2) = bind_name self, %self.loc12_15.1
// CHECK:STDOUT:       %.loc12_10: @Get.%.1 (%.2) = addr_pattern %self.loc12_15.3
// CHECK:STDOUT:       %T.ref.loc12: type = name_ref T, file.%T.loc11_13.2 [symbolic = @Get.%T (constants.%T)]
// CHECK:STDOUT:       %.loc12_34: type = ptr_type %T [symbolic = @Get.%.2 (constants.%.3)]
// CHECK:STDOUT:       %return.var: ref %.3 = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref.loc16: type = name_ref T, file.%T.loc11_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %.loc16: @Class.%.1 (%.4) = field_decl k, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .Get = %Get.decl
// CHECK:STDOUT:     .k = %.loc16
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Get(file.%T.loc11_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = ptr_type @Get.%Class (%Class.2) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %.2: type = ptr_type @Get.%T (%T) [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.3: type = unbound_element_type @Get.%Class (%Class.2), @Get.%T (%T) [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr @Class.%self.loc12_15.3: @Get.%.1 (%.2)]() -> %.3 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Get.%.1 (%.2) = name_ref self, @Class.%self.loc12_15.3
// CHECK:STDOUT:     %.loc13_17.1: ref @Get.%Class (%Class.2) = deref %self.ref
// CHECK:STDOUT:     %k.ref: @Get.%.3 (%.4) = name_ref k, @Class.%.loc16 [template = @Class.%.loc16]
// CHECK:STDOUT:     %.loc13_17.2: ref @Get.%T (%T) = class_element_access %.loc13_17.1, element0
// CHECK:STDOUT:     %.loc13_12: @Get.%.2 (%.3) = addr_of %.loc13_17.2
// CHECK:STDOUT:     return %.loc13_12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: %Class.type = name_ref Class, file.%Class.decl [template = constants.%Class.1]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc25_15.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc25_15.2: type = converted %int.make_type_32, %.loc25_15.1 [template = i32]
// CHECK:STDOUT:   %.loc25_15.3: init type = call %Class.ref(%.loc25_15.2) [template = constants.%Class.3]
// CHECK:STDOUT:   %.loc25_19.1: type = value_of_initializer %.loc25_15.3 [template = constants.%Class.3]
// CHECK:STDOUT:   %.loc25_19.2: type = converted %.loc25_15.3, %.loc25_19.1 [template = constants.%Class.3]
// CHECK:STDOUT:   %v.var: ref %Class.3 = var v
// CHECK:STDOUT:   %v: ref %Class.3 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc25_29: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc25_30: %.9 = struct_literal (%.loc25_29)
// CHECK:STDOUT:   assign %v.var, <error>
// CHECK:STDOUT:   %v.ref: ref %Class.3 = name_ref v, %v
// CHECK:STDOUT:   %Get.ref: %Get.type = name_ref Get, @Class.%Get.decl [template = constants.%Get]
// CHECK:STDOUT:   %.loc36_11: <bound method> = bound_method %v.ref, %Get.ref
// CHECK:STDOUT:   %.loc36_10: %.10 = addr_of %v.ref
// CHECK:STDOUT:   %Get.call: init %.3 = call %.loc36_11(<invalid>) [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Get.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Get(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT:   %.2 => constants.%.3
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
// CHECK:STDOUT:   %.1 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
