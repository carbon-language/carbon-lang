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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Circle.elem: type = unbound_element_type %Circle, %i32 [template]
// CHECK:STDOUT:   %int_5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_5.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %SomeInternalFunction.type: type = fn_type @SomeInternalFunction [template]
// CHECK:STDOUT:   %SomeInternalFunction: %SomeInternalFunction.type = struct_value () [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.radius.1: type = struct_type {.radius: %i32} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.radius.1 [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_0.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %struct_type.radius.2: type = struct_type {.radius: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Circle.val: %Circle = struct_value (%int_5.2) [template]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%int_32.loc5) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_23.1: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_23.2: type = converted %int.make_type_signed.loc5, %.loc5_23.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_21: %Circle.elem = field_decl radius, element0 [template]
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%int_32.loc6) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_39.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_39.2: type = converted %int.make_type_signed.loc6, %.loc6_39.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_5, %impl.elem0 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_5) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc6_46.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc6_46.2: %i32 = converted %int_5, %.loc6_46.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %SOME_INTERNAL_CONSTANT: %i32 = bind_name SOME_INTERNAL_CONSTANT, %.loc6_46.2
// CHECK:STDOUT:   %SomeInternalFunction.decl: %SomeInternalFunction.type = fn_decl @SomeInternalFunction [template = constants.%SomeInternalFunction] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_40.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_40.2: type = converted %int.make_type_signed, %.loc8_40.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %Circle = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Circle = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Circle [template = constants.%Circle]
// CHECK:STDOUT:     %return.param: ref %Circle = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %Circle = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.radius.1 [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Circle
// CHECK:STDOUT:   .radius [private] = %.loc5_21
// CHECK:STDOUT:   .SOME_INTERNAL_CONSTANT [private] = %SOME_INTERNAL_CONSTANT
// CHECK:STDOUT:   .SomeInternalFunction [private] = %SomeInternalFunction.decl
// CHECK:STDOUT:   .Make = %Make.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SomeInternalFunction() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0, %impl.elem0 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_0) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc9_13.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc9_13.2: %i32 = converted %int_0, %.loc9_13.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   return %.loc9_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param_patt: %Circle {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %.loc13_24.1: %struct_type.radius.2 = struct_literal (%int_5)
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_5, %impl.elem0 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_5) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc13_24.2: init %i32 = converted %int_5, %int.convert_checked [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc13_24.3: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc13_24.4: init %i32 = initialize_from %.loc13_24.2 to %.loc13_24.3 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc13_24.5: init %Circle = class_init (%.loc13_24.4), %return [template = constants.%Circle.val]
// CHECK:STDOUT:   %.loc13_25: init %Circle = converted %.loc13_24.1, %.loc13_24.5 [template = constants.%Circle.val]
// CHECK:STDOUT:   return %.loc13_25 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Circle.ref.loc18_15: type = name_ref Circle, file.%Circle.decl [template = constants.%Circle]
// CHECK:STDOUT:   %Circle.ref.loc18_24: type = name_ref Circle, file.%Circle.decl [template = constants.%Circle]
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, @Circle.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc18_36.1: ref %Circle = temporary_storage
// CHECK:STDOUT:   %Make.call: init %Circle = call %Make.ref() to %.loc18_36.1
// CHECK:STDOUT:   %.loc18_36.2: ref %Circle = temporary %.loc18_36.1, %Make.call
// CHECK:STDOUT:   %.loc18_36.3: %Circle = bind_value %.loc18_36.2
// CHECK:STDOUT:   %circle: %Circle = bind_name circle, %.loc18_36.3
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc26_15.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc26_15.2: type = converted %int.make_type_signed, %.loc26_15.1 [template = constants.%i32]
// CHECK:STDOUT:   %circle.ref.loc26: %Circle = name_ref circle, %circle
// CHECK:STDOUT:   %radius.ref.loc26: <error> = name_ref radius, <error> [template = <error>]
// CHECK:STDOUT:   %radius: %i32 = bind_name radius, <error>
// CHECK:STDOUT:   %circle.ref.loc34: %Circle = name_ref circle, %circle
// CHECK:STDOUT:   %radius.ref.loc34: <error> = name_ref radius, <error> [template = <error>]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [template]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [template]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_20.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_20.2: type = converted %int.make_type_signed, %.loc5_20.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_18: %A.elem = field_decl x, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x [protected] = %.loc5_18
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_10.2: type = converted %int.make_type_signed, %.loc16_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %x.ref: <error> = name_ref x, <error> [template = <error>]
// CHECK:STDOUT:   %x: %i32 = bind_name x, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- instance_private_field_access_on_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Circle: type = class_type @Circle [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Circle.elem: type = unbound_element_type %Circle, %i32 [template]
// CHECK:STDOUT:   %GetRadius.type: type = fn_type @GetRadius [template]
// CHECK:STDOUT:   %GetRadius: %GetRadius.type = struct_value () [template]
// CHECK:STDOUT:   %SomeInternalFunction.type: type = fn_type @SomeInternalFunction [template]
// CHECK:STDOUT:   %SomeInternalFunction: %SomeInternalFunction.type = struct_value () [template]
// CHECK:STDOUT:   %Compute.type: type = fn_type @Compute [template]
// CHECK:STDOUT:   %Compute: %Compute.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.radius: type = struct_type {.radius: %i32} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.radius [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_23.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_23.2: type = converted %int.make_type_signed, %.loc5_23.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_21: %Circle.elem = field_decl radius, element0 [template]
// CHECK:STDOUT:   %GetRadius.decl: %GetRadius.type = fn_decl @GetRadius [template = constants.%GetRadius] {
// CHECK:STDOUT:     %self.patt: %Circle = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Circle = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Circle [template = constants.%Circle]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_33.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_33.2: type = converted %int.make_type_signed, %.loc7_33.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: %Circle = value_param runtime_param0
// CHECK:STDOUT:     %self: %Circle = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SomeInternalFunction.decl: %SomeInternalFunction.type = fn_decl @SomeInternalFunction [template = constants.%SomeInternalFunction] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_40.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_40.2: type = converted %int.make_type_signed, %.loc11_40.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Compute.decl: %Compute.type = fn_decl @Compute [template = constants.%Compute] {
// CHECK:STDOUT:     %self.patt: %Circle = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Circle = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Circle [template = constants.%Circle]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_31.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_31.2: type = converted %int.make_type_signed, %.loc15_31.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: %Circle = value_param runtime_param0
// CHECK:STDOUT:     %self: %Circle = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.radius [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Circle
// CHECK:STDOUT:   .radius [private] = %.loc5_21
// CHECK:STDOUT:   .GetRadius = %GetRadius.decl
// CHECK:STDOUT:   .SomeInternalFunction [private] = %SomeInternalFunction.decl
// CHECK:STDOUT:   .Compute = %Compute.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetRadius[%self.param_patt: %Circle]() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Circle = name_ref self, %self
// CHECK:STDOUT:   %radius.ref: %Circle.elem = name_ref radius, @Circle.%.loc5_21 [template = @Circle.%.loc5_21]
// CHECK:STDOUT:   %.loc8_16.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc8_16.2: %i32 = bind_value %.loc8_16.1
// CHECK:STDOUT:   return %.loc8_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SomeInternalFunction() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_0) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_13.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc12_13.2: %i32 = converted %int_0, %.loc12_13.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   return %.loc12_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Compute[%self.param_patt: %Circle]() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Circle = name_ref self, %self
// CHECK:STDOUT:   %SomeInternalFunction.ref: %SomeInternalFunction.type = name_ref SomeInternalFunction, @Circle.%SomeInternalFunction.decl [template = constants.%SomeInternalFunction]
// CHECK:STDOUT:   %SomeInternalFunction.call: init %i32 = call %SomeInternalFunction.ref()
// CHECK:STDOUT:   %.loc16_39.1: %i32 = value_of_initializer %SomeInternalFunction.call
// CHECK:STDOUT:   %.loc16_39.2: %i32 = converted %SomeInternalFunction.call, %.loc16_39.1
// CHECK:STDOUT:   return %.loc16_39.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- public_global_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_5.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8.2: type = converted %int.make_type_signed, %.loc8_8.1 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_10.2: type = converted %int.make_type_signed, %.loc5_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_5, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_5) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc5_17.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc5_17.2: %i32 = converted %int_5, %.loc5_17.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %x: %i32 = bind_name x, %.loc5_17.2
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %x
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, @A.%x
// CHECK:STDOUT:   %x: %i32 = bind_name x, %x.ref
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_global_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_5.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %int_32.loc16: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc16: init type = call constants.%Int(%int_32.loc16) [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_8.1: type = value_of_initializer %int.make_type_signed.loc16 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_8.2: type = converted %int.make_type_signed.loc16, %.loc16_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc23: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc23: init type = call constants.%Int(%int_32.loc23) [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_8.1: type = value_of_initializer %int.make_type_signed.loc23 [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_8.2: type = converted %int.make_type_signed.loc23, %.loc23_8.1 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%int_32.loc5) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_20.1: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_20.2: type = converted %int.make_type_signed.loc5, %.loc5_20.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_5.loc5: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %impl.elem0.loc5: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc5: <bound method> = bound_method %int_5.loc5, %impl.elem0.loc5 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc5: <specific function> = specific_function %Convert.bound.loc5, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc5: init %i32 = call %Convert.specific_fn.loc5(%int_5.loc5) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc5_27.1: %i32 = value_of_initializer %int.convert_checked.loc5 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc5_27.2: %i32 = converted %int_5.loc5, %.loc5_27.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %x: %i32 = bind_name x, %.loc5_27.2
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%int_32.loc6) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_18.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_18.2: type = converted %int.make_type_signed.loc6, %.loc6_18.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_5.loc6: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %impl.elem0.loc6: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc6: <bound method> = bound_method %int_5.loc6, %impl.elem0.loc6 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc6: <specific function> = specific_function %Convert.bound.loc6, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc6: init %i32 = call %Convert.specific_fn.loc6(%int_5.loc6) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc6_25.1: %i32 = value_of_initializer %int.convert_checked.loc6 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc6_25.2: %i32 = converted %int_5.loc6, %.loc6_25.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %y: %i32 = bind_name y, %.loc6_25.2
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x [protected] = %x
// CHECK:STDOUT:   .y [private] = %y
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref.loc16: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %x.ref: <error> = name_ref x, <error> [template = <error>]
// CHECK:STDOUT:   %x: %i32 = bind_name x, <error>
// CHECK:STDOUT:   %A.ref.loc23: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %y.ref: <error> = name_ref y, <error> [template = <error>]
// CHECK:STDOUT:   %y: %i32 = bind_name y, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- self_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F [private] = %F.decl
// CHECK:STDOUT:   .G [private] = %G.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
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
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
