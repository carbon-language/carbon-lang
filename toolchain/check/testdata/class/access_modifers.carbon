// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDERR: fail_global_access.carbon:[[@LINE+7]]:14: error: cannot access private member `y` of type `A` [ClassInvalidMemberAccess]
// CHECK:STDERR: let y: i32 = A.y;
// CHECK:STDERR:              ^~~
// CHECK:STDERR: fail_global_access.carbon:[[@LINE-14]]:15: note: declared here [ClassMemberDeclaration]
// CHECK:STDERR:   private let y: i32 = 5;
// CHECK:STDERR:               ^
// CHECK:STDERR:
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
// CHECK:STDOUT:   %Circle: type = class_type @Circle [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Circle.elem: type = unbound_element_type %Circle, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e6: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.a25: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Circle.SomeInternalFunction.type: type = fn_type @Circle.SomeInternalFunction [concrete]
// CHECK:STDOUT:   %Circle.SomeInternalFunction: %Circle.SomeInternalFunction.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Circle [concrete]
// CHECK:STDOUT:   %Circle.Make.type: type = fn_type @Circle.Make [concrete]
// CHECK:STDOUT:   %Circle.Make: %Circle.Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.radius.251: type = struct_type {.radius: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.5a5: <witness> = complete_type_witness %struct_type.radius.251 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %struct_type.radius.f47: type = struct_type {.radius: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %Circle.val: %Circle = struct_value (%int_5.0f6) [concrete]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.6cf: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%Circle) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.e38: %T.as.Destroy.impl.Op.type.6cf = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.54d: type = ptr_type %Circle [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %T.as.Destroy.impl.Op.e38, @T.as.Destroy.impl.Op(%Circle) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Circle = %Circle.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Circle.decl: type = class_decl @Circle [concrete = constants.%Circle] {} {}
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Circle {
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5: %Circle.elem = field_decl radius, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %SOME_INTERNAL_CONSTANT.patt: %pattern_type.7ce = binding_pattern SOME_INTERNAL_CONSTANT [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc6_39: type = splice_block %i32.loc6 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc6_45.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e6]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_45.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method.a25]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc6_45.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc6_45.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc6_45.2: %i32 = converted %int_5, %.loc6_45.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %SOME_INTERNAL_CONSTANT: %i32 = bind_name SOME_INTERNAL_CONSTANT, %.loc6_45.2
// CHECK:STDOUT:   %Circle.SomeInternalFunction.decl: %Circle.SomeInternalFunction.type = fn_decl @Circle.SomeInternalFunction [concrete = constants.%Circle.SomeInternalFunction] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Circle.Make.decl: %Circle.Make.type = fn_decl @Circle.Make [concrete = constants.%Circle.Make] {
// CHECK:STDOUT:     %return.patt: %pattern_type.ce2 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.ce2 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Circle [concrete = constants.%Circle]
// CHECK:STDOUT:     %return.param: ref %Circle = out_param call_param0
// CHECK:STDOUT:     %return: ref %Circle = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_type.radius: type = struct_type {.radius: %i32} [concrete = constants.%struct_type.radius.251]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.radius [concrete = constants.%complete_type.5a5]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Circle
// CHECK:STDOUT:   .radius [private] = %.loc5
// CHECK:STDOUT:   .SOME_INTERNAL_CONSTANT [private] = %SOME_INTERNAL_CONSTANT
// CHECK:STDOUT:   .SomeInternalFunction [private] = %Circle.SomeInternalFunction.decl
// CHECK:STDOUT:   .Make = %Circle.Make.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Circle.SomeInternalFunction() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc9_13.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_13.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_13.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc9_13.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc9_13.2: %i32 = converted %int_0, %.loc9_13.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   return %.loc9_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Circle.Make() -> %return.param: %Circle {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc13_24.1: %struct_type.radius.f47 = struct_literal (%int_5)
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_24.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e6]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_24.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method.a25]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_24.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc13_24.2: init %i32 = converted %int_5, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc13_24.3: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc13_24.4: init %i32 = initialize_from %.loc13_24.2 to %.loc13_24.3 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc13_24.5: init %Circle = class_init (%.loc13_24.4), %return [concrete = constants.%Circle.val]
// CHECK:STDOUT:   %.loc13_25: init %Circle = converted %.loc13_24.1, %.loc13_24.5 [concrete = constants.%Circle.val]
// CHECK:STDOUT:   return %.loc13_25 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %circle.patt: %pattern_type.ce2 = binding_pattern circle [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Circle.ref.loc18_24: type = name_ref Circle, file.%Circle.decl [concrete = constants.%Circle]
// CHECK:STDOUT:   %Make.ref: %Circle.Make.type = name_ref Make, @Circle.%Circle.Make.decl [concrete = constants.%Circle.Make]
// CHECK:STDOUT:   %.loc18_36.1: ref %Circle = temporary_storage
// CHECK:STDOUT:   %Circle.Make.call: init %Circle = call %Make.ref() to %.loc18_36.1
// CHECK:STDOUT:   %Circle.ref.loc18_15: type = name_ref Circle, file.%Circle.decl [concrete = constants.%Circle]
// CHECK:STDOUT:   %.loc18_36.2: ref %Circle = temporary %.loc18_36.1, %Circle.Make.call
// CHECK:STDOUT:   %.loc18_36.3: %Circle = bind_value %.loc18_36.2
// CHECK:STDOUT:   %circle: %Circle = bind_name circle, %.loc18_36.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %radius.patt: %pattern_type.7ce = binding_pattern radius [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %circle.ref.loc26: %Circle = name_ref circle, %circle
// CHECK:STDOUT:   %radius.ref.loc26: <error> = name_ref radius, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc26: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %radius: %i32 = bind_name radius, <error> [concrete = <error>]
// CHECK:STDOUT:   %circle.ref.loc34: %Circle = name_ref circle, %circle
// CHECK:STDOUT:   %radius.ref.loc34: <error> = name_ref radius, <error> [concrete = <error>]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   assign %radius.ref.loc34, <error>
// CHECK:STDOUT:   %circle.ref.loc42: %Circle = name_ref circle, %circle
// CHECK:STDOUT:   %SOME_INTERNAL_CONSTANT.ref: <error> = name_ref SOME_INTERNAL_CONSTANT, <error> [concrete = <error>]
// CHECK:STDOUT:   %circle.ref.loc51: %Circle = name_ref circle, %circle
// CHECK:STDOUT:   %SomeInternalFunction.ref: <error> = name_ref SomeInternalFunction, <error> [concrete = <error>]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc18_36.1, constants.%T.as.Destroy.impl.Op.e38
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.e38, @T.as.Destroy.impl.Op(constants.%Circle) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc18_36.1, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.54d = addr_of %.loc18_36.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_protected_field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5: %A.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %i32} [concrete = constants.%struct_type.x]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [concrete = constants.%complete_type.1ec]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x [protected] = %.loc5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %x.ref: <error> = name_ref x, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc16: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: %i32 = bind_name x, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- instance_private_field_access_on_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Circle: type = class_type @Circle [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Circle.elem: type = unbound_element_type %Circle, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Circle [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Circle.GetRadius.type: type = fn_type @Circle.GetRadius [concrete]
// CHECK:STDOUT:   %Circle.GetRadius: %Circle.GetRadius.type = struct_value () [concrete]
// CHECK:STDOUT:   %Circle.SomeInternalFunction.type: type = fn_type @Circle.SomeInternalFunction [concrete]
// CHECK:STDOUT:   %Circle.SomeInternalFunction: %Circle.SomeInternalFunction.type = struct_value () [concrete]
// CHECK:STDOUT:   %Circle.Compute.type: type = fn_type @Circle.Compute [concrete]
// CHECK:STDOUT:   %Circle.Compute: %Circle.Compute.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.radius: type = struct_type {.radius: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.5a5: <witness> = complete_type_witness %struct_type.radius [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Circle = %Circle.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Circle.decl: type = class_decl @Circle [concrete = constants.%Circle] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Circle {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5: %Circle.elem = field_decl radius, element0 [concrete]
// CHECK:STDOUT:   %Circle.GetRadius.decl: %Circle.GetRadius.type = fn_decl @Circle.GetRadius [concrete = constants.%Circle.GetRadius] {
// CHECK:STDOUT:     %self.patt: %pattern_type.ce2 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ce2 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %Circle = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Circle [concrete = constants.%Circle]
// CHECK:STDOUT:     %self: %Circle = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Circle.SomeInternalFunction.decl: %Circle.SomeInternalFunction.type = fn_decl @Circle.SomeInternalFunction [concrete = constants.%Circle.SomeInternalFunction] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Circle.Compute.decl: %Circle.Compute.type = fn_decl @Circle.Compute [concrete = constants.%Circle.Compute] {
// CHECK:STDOUT:     %self.patt: %pattern_type.ce2 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ce2 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %Circle = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Circle [concrete = constants.%Circle]
// CHECK:STDOUT:     %self: %Circle = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_type.radius: type = struct_type {.radius: %i32} [concrete = constants.%struct_type.radius]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.radius [concrete = constants.%complete_type.5a5]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Circle
// CHECK:STDOUT:   .radius [private] = %.loc5
// CHECK:STDOUT:   .GetRadius = %Circle.GetRadius.decl
// CHECK:STDOUT:   .SomeInternalFunction [private] = %Circle.SomeInternalFunction.decl
// CHECK:STDOUT:   .Compute = %Circle.Compute.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Circle.GetRadius(%self.param: %Circle) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Circle = name_ref self, %self
// CHECK:STDOUT:   %radius.ref: %Circle.elem = name_ref radius, @Circle.%.loc5 [concrete = @Circle.%.loc5]
// CHECK:STDOUT:   %.loc8_16.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc8_16.2: %i32 = bind_value %.loc8_16.1
// CHECK:STDOUT:   return %.loc8_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Circle.SomeInternalFunction() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc12_13.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_13.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_13.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc12_13.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc12_13.2: %i32 = converted %int_0, %.loc12_13.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   return %.loc12_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Circle.Compute(%self.param: %Circle) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Circle = name_ref self, %self
// CHECK:STDOUT:   %SomeInternalFunction.ref: %Circle.SomeInternalFunction.type = name_ref SomeInternalFunction, @Circle.%Circle.SomeInternalFunction.decl [concrete = constants.%Circle.SomeInternalFunction]
// CHECK:STDOUT:   %Circle.SomeInternalFunction.call: init %i32 = call %SomeInternalFunction.ref()
// CHECK:STDOUT:   %.loc16_39.1: %i32 = value_of_initializer %Circle.SomeInternalFunction.call
// CHECK:STDOUT:   %.loc16_39.2: %i32 = converted %Circle.SomeInternalFunction.call, %.loc16_39.1
// CHECK:STDOUT:   return %.loc16_39.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- public_global_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: %i32 = bind_name x, @__global_init.%x.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc5_10: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc5_16.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_16.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc5_16.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc5_16.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc5_16.2: %i32 = converted %int_5, %.loc5_16.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %x: %i32 = bind_name x, %.loc5_16.2
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x = %x
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, @A.%x
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_global_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16: type = splice_block %i32.loc16 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: %i32 = bind_name x, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.7ce = binding_pattern y [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc24: type = splice_block %i32.loc24 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: %i32 = bind_name y, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_5.loc5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc5_20: type = splice_block %i32.loc5 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc5: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc5_26.1: <bound method> = bound_method %int_5.loc5, %impl.elem0.loc5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_26.2: <bound method> = bound_method %int_5.loc5, %specific_fn.loc5 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5: init %i32 = call %bound_method.loc5_26.2(%int_5.loc5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc5_26.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc5_26.2: %i32 = converted %int_5.loc5, %.loc5_26.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %x: %i32 = bind_name x, %.loc5_26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.7ce = binding_pattern y [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_5.loc6: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc6_18: type = splice_block %i32.loc6 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc6: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc6_24.1: <bound method> = bound_method %int_5.loc6, %impl.elem0.loc6 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_24.2: <bound method> = bound_method %int_5.loc6, %specific_fn.loc6 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %i32 = call %bound_method.loc6_24.2(%int_5.loc6) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc6_24.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc6_24.2: %i32 = converted %int_5.loc6, %.loc6_24.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %y: %i32 = bind_name y, %.loc6_24.2
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x [protected] = %x
// CHECK:STDOUT:   .y [private] = %y
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref.loc16: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %x.ref: <error> = name_ref x, <error> [concrete = <error>]
// CHECK:STDOUT:   %A.ref.loc24: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %y.ref: <error> = name_ref y, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- self_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.F: %A.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.G.type: type = fn_type @A.G [concrete]
// CHECK:STDOUT:   %A.G: %A.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %A.F.decl: %A.F.type = fn_decl @A.F [concrete = constants.%A.F] {} {}
// CHECK:STDOUT:   %A.G.decl: %A.G.type = fn_decl @A.G [concrete = constants.%A.G] {} {}
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F [private] = %A.F.decl
// CHECK:STDOUT:   .G [private] = %A.G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:   %F.ref: %A.F.type = name_ref F, @A.%A.F.decl [concrete = constants.%A.F]
// CHECK:STDOUT:   %A.F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
