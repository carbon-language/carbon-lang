// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/access_modifers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/access_modifers.carbon

// --- fail_private_field_access.carbon

library "[[@TEST_NAME]]";

class Circle {
  private var radius: i32;
  private let SOME_INTERNAL_CONSTANT: i32 = 5;

  private fn SomeInternalFunction() -> i32 {
    return 0;
  }

  fn Make() -> Self {
    return {.radius = 5};
  }
}

fn Run() {
  let circle: Circle = Circle.Make();
  // CHECK:STDERR: fail_private_field_access.carbon:[[@LINE+7]]:21: error: cannot access private member `radius` of type `Circle` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let radius: i32 = circle.radius;
  // CHECK:STDERR:                     ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_private_field_access.carbon:[[@LINE-17]]:15: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   private var radius: i32;
  // CHECK:STDERR:               ^~~~~~~~~~~
  // CHECK:STDERR:
  let radius: i32 = circle.radius;
  // CHECK:STDERR: fail_private_field_access.carbon:[[@LINE+7]]:3: error: cannot access private member `radius` of type `Circle` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   circle.radius = 5;
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_private_field_access.carbon:[[@LINE-25]]:15: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   private var radius: i32;
  // CHECK:STDERR:               ^~~~~~~~~~~
  // CHECK:STDERR:
  circle.radius = 5;
  // CHECK:STDERR: fail_private_field_access.carbon:[[@LINE+7]]:3: error: cannot access private member `SOME_INTERNAL_CONSTANT` of type `Circle` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   circle.SOME_INTERNAL_CONSTANT;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_private_field_access.carbon:[[@LINE-32]]:15: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   private let SOME_INTERNAL_CONSTANT: i32 = 5;
  // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  circle.SOME_INTERNAL_CONSTANT;

  // CHECK:STDERR: fail_private_field_access.carbon:[[@LINE+7]]:3: error: cannot access private member `SomeInternalFunction` of type `Circle` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   circle.SomeInternalFunction();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_private_field_access.carbon:[[@LINE-39]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   private fn SomeInternalFunction() -> i32 {
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  circle.SomeInternalFunction();
}

// --- fail_protected_field_access.carbon

library "[[@TEST_NAME]]";

class A {
  protected var x: i32;
}

fn Run() {
  // CHECK:STDERR: fail_protected_field_access.carbon:[[@LINE+7]]:16: error: cannot access protected member `x` of type `A` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let x: i32 = A.x;
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR: fail_protected_field_access.carbon:[[@LINE-7]]:17: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   protected var x: i32;
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR:
  let x: i32 = A.x;
}

// --- instance_private_field_access_on_self.carbon

library "[[@TEST_NAME]]";

class Circle {
  private var radius: i32;

  fn GetRadius[self: Self]() -> i32 {
    return self.radius;
  }

  private fn SomeInternalFunction() -> i32 {
    return 0;
  }

  fn Compute[self: Self]() -> i32 {
    return self.SomeInternalFunction();
  }
}

// --- public_global_access.carbon

library "[[@TEST_NAME]]";

class A {
  let x: i32 = 5;
}

let x: i32 = A.x;

// --- fail_global_access.carbon

library "[[@TEST_NAME]]";

class A {
  protected let x: i32 = 5;
  private let y: i32 = 5;
}

// CHECK:STDERR: fail_global_access.carbon:[[@LINE+7]]:14: error: cannot access protected member `x` of type `A` [ClassInvalidMemberAccess]
// CHECK:STDERR: let x: i32 = A.x;
// CHECK:STDERR:              ^~~
// CHECK:STDERR: fail_global_access.carbon:[[@LINE-7]]:17: note: declared here [ClassMemberDeclaration]
// CHECK:STDERR:   protected let x: i32 = 5;
// CHECK:STDERR:                 ^
// CHECK:STDERR:
let x: i32 = A.x;
// CHECK:STDERR: fail_global_access.carbon:[[@LINE+6]]:14: error: cannot access private member `y` of type `A` [ClassInvalidMemberAccess]
// CHECK:STDERR: let y: i32 = A.y;
// CHECK:STDERR:              ^~~
// CHECK:STDERR: fail_global_access.carbon:[[@LINE-14]]:15: note: declared here [ClassMemberDeclaration]
// CHECK:STDERR:   private let y: i32 = 5;
// CHECK:STDERR:               ^
let y: i32 = A.y;

// --- self_access.carbon

library "[[@TEST_NAME]]";

class A {
  private fn F() {}
  private fn G() { Self.F(); }
}

// CHECK:STDOUT: --- fail_private_field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Circle: type = class_type @Circle [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Circle, i32 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %SomeInternalFunction.type: type = fn_type @SomeInternalFunction [template]
// CHECK:STDOUT:   %SomeInternalFunction: %SomeInternalFunction.type = struct_value () [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = struct_type {.radius: i32} [template]
// CHECK:STDOUT:   %.5: <witness> = complete_type_witness %.4 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %struct: %Circle = struct_value (%.3) [template]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Circle = %Circle.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Circle.decl: type = class_decl @Circle [template = constants.%Circle] {} {}
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Circle {
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_23.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_23.2: type = converted %int.make_type_32.loc5, %.loc5_23.1 [template = i32]
// CHECK:STDOUT:   %.loc5_21: %.2 = field_decl radius, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_39.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_39.2: type = converted %int.make_type_32.loc6, %.loc6_39.1 [template = i32]
// CHECK:STDOUT:   %.loc6_45: i32 = int_literal 5 [template = constants.%.3]
// CHECK:STDOUT:   %SOME_INTERNAL_CONSTANT: i32 = bind_name SOME_INTERNAL_CONSTANT, %.loc6_45
// CHECK:STDOUT:   %SomeInternalFunction.decl: %SomeInternalFunction.type = fn_decl @SomeInternalFunction [template = constants.%SomeInternalFunction] {
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_40.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc8_40.2: type = converted %int.make_type_32, %.loc8_40.1 [template = i32]
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %Circle = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Circle = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Circle [template = constants.%Circle]
// CHECK:STDOUT:     %return.param: ref %Circle = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %Circle = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Circle
// CHECK:STDOUT:   .radius [private] = %.loc5_21
// CHECK:STDOUT:   .SOME_INTERNAL_CONSTANT [private] = %SOME_INTERNAL_CONSTANT
// CHECK:STDOUT:   .SomeInternalFunction [private] = %SomeInternalFunction.decl
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SomeInternalFunction() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   return %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: %Circle {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_23: i32 = int_literal 5 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_24.1: %.4 = struct_literal (%.loc13_23)
// CHECK:STDOUT:   %.loc13_24.2: ref i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc13_24.3: init i32 = initialize_from %.loc13_23 to %.loc13_24.2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_24.4: init %Circle = class_init (%.loc13_24.3), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_25: init %Circle = converted %.loc13_24.1, %.loc13_24.4 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc13_25 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Circle.ref.loc18_15: type = name_ref Circle, file.%Circle.decl [template = constants.%Circle]
// CHECK:STDOUT:   %Circle.ref.loc18_24: type = name_ref Circle, file.%Circle.decl [template = constants.%Circle]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, @Circle.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc18_35.1: ref %Circle = temporary_storage
// CHECK:STDOUT:   %Make.call: init %Circle = call %Make.ref() to %.loc18_35.1
// CHECK:STDOUT:   %.loc18_35.2: ref %Circle = temporary %.loc18_35.1, %Make.call
// CHECK:STDOUT:   %.loc18_35.3: %Circle = bind_value %.loc18_35.2
// CHECK:STDOUT:   %circle: %Circle = bind_name circle, %.loc18_35.3
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc26_15.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc26_15.2: type = converted %int.make_type_32, %.loc26_15.1 [template = i32]
// CHECK:STDOUT:   %circle.ref.loc26: %Circle = name_ref circle, %circle
// CHECK:STDOUT:   %radius.ref.loc26: <error> = name_ref radius, <error> [template = <error>]
// CHECK:STDOUT:   %radius: i32 = bind_name radius, <error>
// CHECK:STDOUT:   %circle.ref.loc34: %Circle = name_ref circle, %circle
// CHECK:STDOUT:   %radius.ref.loc34: <error> = name_ref radius, <error> [template = <error>]
// CHECK:STDOUT:   %.loc34: i32 = int_literal 5 [template = constants.%.3]
// CHECK:STDOUT:   assign %radius.ref.loc34, <error>
// CHECK:STDOUT:   %circle.ref.loc42: %Circle = name_ref circle, %circle
// CHECK:STDOUT:   %SOME_INTERNAL_CONSTANT.ref: <error> = name_ref SOME_INTERNAL_CONSTANT, <error> [template = <error>]
// CHECK:STDOUT:   %circle.ref.loc51: %Circle = name_ref circle, %circle
// CHECK:STDOUT:   %SomeInternalFunction.ref: <error> = name_ref SomeInternalFunction, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_protected_field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %A, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.x: i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [template = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_20.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc5_20.2: type = converted %int.make_type_32, %.loc5_20.1 [template = i32]
// CHECK:STDOUT:   %.loc5_18: %.2 = field_decl x, element0 [template]
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x [protected] = %.loc5_18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc16_10.2: type = converted %int.make_type_32, %.loc16_10.1 [template = i32]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %x.ref: <error> = name_ref x, <error> [template = <error>]
// CHECK:STDOUT:   %x: i32 = bind_name x, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- instance_private_field_access_on_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Circle: type = class_type @Circle [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Circle, i32 [template]
// CHECK:STDOUT:   %GetRadius.type: type = fn_type @GetRadius [template]
// CHECK:STDOUT:   %GetRadius: %GetRadius.type = struct_value () [template]
// CHECK:STDOUT:   %SomeInternalFunction.type: type = fn_type @SomeInternalFunction [template]
// CHECK:STDOUT:   %SomeInternalFunction: %SomeInternalFunction.type = struct_value () [template]
// CHECK:STDOUT:   %Compute.type: type = fn_type @Compute [template]
// CHECK:STDOUT:   %Compute: %Compute.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.radius: i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Circle = %Circle.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Circle.decl: type = class_decl @Circle [template = constants.%Circle] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Circle {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_23.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc5_23.2: type = converted %int.make_type_32, %.loc5_23.1 [template = i32]
// CHECK:STDOUT:   %.loc5_21: %.2 = field_decl radius, element0 [template]
// CHECK:STDOUT:   %GetRadius.decl: %GetRadius.type = fn_decl @GetRadius [template = constants.%GetRadius] {
// CHECK:STDOUT:     %self.patt: %Circle = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Circle = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Circle [template = constants.%Circle]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_33.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc7_33.2: type = converted %int.make_type_32, %.loc7_33.1 [template = i32]
// CHECK:STDOUT:     %self.param: %Circle = value_param runtime_param0
// CHECK:STDOUT:     %self: %Circle = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SomeInternalFunction.decl: %SomeInternalFunction.type = fn_decl @SomeInternalFunction [template = constants.%SomeInternalFunction] {
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_40.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc11_40.2: type = converted %int.make_type_32, %.loc11_40.1 [template = i32]
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compute.decl: %Compute.type = fn_decl @Compute [template = constants.%Compute] {
// CHECK:STDOUT:     %self.patt: %Circle = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Circle = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Circle [template = constants.%Circle]
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_31.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc15_31.2: type = converted %int.make_type_32, %.loc15_31.1 [template = i32]
// CHECK:STDOUT:     %self.param: %Circle = value_param runtime_param0
// CHECK:STDOUT:     %self: %Circle = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Circle
// CHECK:STDOUT:   .radius [private] = %.loc5_21
// CHECK:STDOUT:   .GetRadius = %GetRadius.decl
// CHECK:STDOUT:   .SomeInternalFunction [private] = %SomeInternalFunction.decl
// CHECK:STDOUT:   .Compute = %Compute.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetRadius[%self.param_patt: %Circle]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Circle = name_ref self, %self
// CHECK:STDOUT:   %radius.ref: %.2 = name_ref radius, @Circle.%.loc5_21 [template = @Circle.%.loc5_21]
// CHECK:STDOUT:   %.loc8_16.1: ref i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc8_16.2: i32 = bind_value %.loc8_16.1
// CHECK:STDOUT:   return %.loc8_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SomeInternalFunction() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   return %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Compute[%self.param_patt: %Circle]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Circle = name_ref self, %self
// CHECK:STDOUT:   %SomeInternalFunction.ref: %SomeInternalFunction.type = name_ref SomeInternalFunction, @Circle.%SomeInternalFunction.decl [template = constants.%SomeInternalFunction]
// CHECK:STDOUT:   %SomeInternalFunction.call: init i32 = call %SomeInternalFunction.ref()
// CHECK:STDOUT:   %.loc16_39.1: i32 = value_of_initializer %SomeInternalFunction.call
// CHECK:STDOUT:   %.loc16_39.2: i32 = converted %SomeInternalFunction.call, %.loc16_39.1
// CHECK:STDOUT:   return %.loc16_39.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- public_global_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .x = @__global_init.%x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_8.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc8_8.2: type = converted %int.make_type_32, %.loc8_8.1 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc5_10.2: type = converted %int.make_type_32, %.loc5_10.1 [template = i32]
// CHECK:STDOUT:   %.loc5_16: i32 = int_literal 5 [template = constants.%.2]
// CHECK:STDOUT:   %x: i32 = bind_name x, %.loc5_16
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %x
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %x.ref: i32 = name_ref x, @A.%x
// CHECK:STDOUT:   %x: i32 = bind_name x, %x.ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_global_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .x = @__global_init.%x
// CHECK:STDOUT:     .y = @__global_init.%y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %int.make_type_32.loc16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_8.1: type = value_of_initializer %int.make_type_32.loc16 [template = i32]
// CHECK:STDOUT:   %.loc16_8.2: type = converted %int.make_type_32.loc16, %.loc16_8.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_8.1: type = value_of_initializer %int.make_type_32.loc23 [template = i32]
// CHECK:STDOUT:   %.loc23_8.2: type = converted %int.make_type_32.loc23, %.loc23_8.1 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_20.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_20.2: type = converted %int.make_type_32.loc5, %.loc5_20.1 [template = i32]
// CHECK:STDOUT:   %.loc5_26: i32 = int_literal 5 [template = constants.%.2]
// CHECK:STDOUT:   %x: i32 = bind_name x, %.loc5_26
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_18.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_18.2: type = converted %int.make_type_32.loc6, %.loc6_18.1 [template = i32]
// CHECK:STDOUT:   %.loc6_24: i32 = int_literal 5 [template = constants.%.2]
// CHECK:STDOUT:   %y: i32 = bind_name y, %.loc6_24
// CHECK:STDOUT:   %.loc7: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x [protected] = %x
// CHECK:STDOUT:   .y [private] = %y
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref.loc16: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %x.ref: <error> = name_ref x, <error> [template = <error>]
// CHECK:STDOUT:   %x: i32 = bind_name x, <error>
// CHECK:STDOUT:   %A.ref.loc23: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %y.ref: <error> = name_ref y, <error> [template = <error>]
// CHECK:STDOUT:   %y: i32 = bind_name y, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- self_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.2 [template]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT:   %.loc7: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F [private] = %F.decl
// CHECK:STDOUT:   .G [private] = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%A [template = constants.%A]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @A.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init %.1 = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
