// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/inheritance_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/inheritance_access.carbon


// --- instance_protected_field_access.carbon

library "[[@TEST_NAME]]";

base class Shape {
  protected var x: i32;
  protected var y: i32;
}

class Circle {
  extend base: Shape;

  fn GetPosition[self: Self]() -> (i32, i32) {
    return (self.x, self.y);
  }
}

// --- shadowing_access.carbon

library "[[@TEST_NAME]]";

base class A {
  fn F();
}

base class B {
  extend base: A;
  private fn F() -> i32;
}

base class C {
  extend base: B;
  fn G[self: Self]() -> () { return self.F(); }
}

// --- inherited_member_access.carbon

library "[[@TEST_NAME]]";

base class A {
  protected let SOME_CONSTANT: i32 = 5;
  protected fn SomeProtectedFunction() -> i32 {
    return 5;
  }
}

class B {
  extend base: A;

  fn G() -> i32 {
    return A.SOME_CONSTANT;
  }

  fn H() -> i32 {
    return A.SomeProtectedFunction();
  }
}

// --- fail_inherited_private_field_access.carbon

library "[[@TEST_NAME]]";

base class Shape {
  private var y: i32;
}

class Square {
  extend base: Shape;

  fn GetPosition[self: Self]() -> i32 {
    // CHECK:STDERR: fail_inherited_private_field_access.carbon:[[@LINE+7]]:12: error: cannot access private member `y` of type `Shape` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     return self.y;
    // CHECK:STDERR:            ^~~~~~
    // CHECK:STDERR: fail_inherited_private_field_access.carbon:[[@LINE-10]]:15: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:   private var y: i32;
    // CHECK:STDERR:               ^~~~~~
    // CHECK:STDERR:
    return self.y;
  }
}

// --- noninstance_private_on_self.carbon

library "[[@TEST_NAME]]";

class C {
  private fn F() {}
  fn G() { Self.F(); }
}

// --- noninstance_protected_on_self.carbon

library "[[@TEST_NAME]]";

class C {
  protected fn F() {}
  fn G() { Self.F(); }
}

// --- fail_noninstance_private_on_parent.carbon

library "[[@TEST_NAME]]";

base class B {
  private fn F() {}
}

class C {
  extend base: B;
  // CHECK:STDERR: fail_noninstance_private_on_parent.carbon:[[@LINE+7]]:12: error: cannot access private member `F` of type `B` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   fn G() { Self.F(); }
  // CHECK:STDERR:            ^~~~~~
  // CHECK:STDERR: fail_noninstance_private_on_parent.carbon:[[@LINE-8]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   private fn F() {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn G() { Self.F(); }
}

// --- noninstance_protected_on_parent.carbon

library "[[@TEST_NAME]]";

base class B {
  protected fn F() {}
}

class C {
  extend base: B;
  fn G() { Self.F(); }
}

// --- fail_non_inherited_access.carbon

library "[[@TEST_NAME]]";

base class A {
  protected let SOME_PROTECTED_CONSTANT: i32 = 5;
  private let SOME_PRIVATE_CONSTANT: i32 = 5;
}

base class Internal {
  protected let INTERNAL_CONSTANT: i32 = 5;
}

class B {
  private var internal: Internal;

  fn G() -> i32 {
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE+7]]:5: error: cannot access private member `SOME_PRIVATE_CONSTANT` of type `A` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     A.SOME_PRIVATE_CONSTANT;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE-14]]:15: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:   private let SOME_PRIVATE_CONSTANT: i32 = 5;
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    A.SOME_PRIVATE_CONSTANT;

    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE+7]]:12: error: cannot access protected member `SOME_PROTECTED_CONSTANT` of type `A` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     return A.SOME_PROTECTED_CONSTANT;
    // CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE-24]]:17: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:   protected let SOME_PROTECTED_CONSTANT: i32 = 5;
    // CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    return A.SOME_PROTECTED_CONSTANT;
  }

  fn SomeFunc[self: Self]() -> i32{
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE+7]]:12: error: cannot access protected member `INTERNAL_CONSTANT` of type `Internal` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     return self.internal.INTERNAL_CONSTANT;
    // CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE-30]]:17: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:   protected let INTERNAL_CONSTANT: i32 = 5;
    // CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    return self.internal.INTERNAL_CONSTANT;
  }
}

// --- fail_compound_member_access.carbon

library "[[@TEST_NAME]]";

base class A {
  private var x: i32;
}

class B {
  extend base: A;
  fn F[self: Self]() {
    // CHECK:STDERR: fail_compound_member_access.carbon:[[@LINE+6]]:11: error: cannot access private member `x` of type `A` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     self.(A.x);
    // CHECK:STDERR:           ^~~
    // CHECK:STDERR: fail_compound_member_access.carbon:[[@LINE-9]]:15: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:   private var x: i32;
    // CHECK:STDERR:               ^~~~~~
    self.(A.x);
  }
}

// --- inherited_compound_member_access.carbon

library "[[@TEST_NAME]]";

base class A {
  protected var x: i32;
}

class B {
  extend base: A;

  fn F[self: Self]() {
    self.(A.x);
  }
}

// CHECK:STDOUT: --- instance_protected_field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Shape: type = class_type @Shape [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Shape.elem: type = unbound_element_type %Shape, %i32 [template]
// CHECK:STDOUT:   %struct_type.x.y: type = struct_type {.x: %i32, .y: %i32} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.x.y [template]
// CHECK:STDOUT:   %Circle: type = class_type @Circle [template]
// CHECK:STDOUT:   %Circle.elem: type = unbound_element_type %Circle, %Shape [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %GetPosition.type: type = fn_type @GetPosition [template]
// CHECK:STDOUT:   %GetPosition: %GetPosition.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.base.1: type = struct_type {.base: %Shape} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.base.1 [template]
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
// CHECK:STDOUT:     .Shape = %Shape.decl
// CHECK:STDOUT:     .Circle = %Circle.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Shape.decl: type = class_decl @Shape [template = constants.%Shape] {} {}
// CHECK:STDOUT:   %Circle.decl: type = class_decl @Circle [template = constants.%Circle] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Shape {
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%int_32.loc5) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_20.1: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_20.2: type = converted %int.make_type_signed.loc5, %.loc5_20.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_18: %Shape.elem = field_decl x, element0 [template]
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%int_32.loc6) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_20.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_20.2: type = converted %int.make_type_signed.loc6, %.loc6_20.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_18: %Shape.elem = field_decl y, element1 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.y [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Shape
// CHECK:STDOUT:   .x [protected] = %.loc5_18
// CHECK:STDOUT:   .y [protected] = %.loc6_18
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Circle {
// CHECK:STDOUT:   %Shape.ref: type = name_ref Shape, file.%Shape.decl [template = constants.%Shape]
// CHECK:STDOUT:   %.loc10: %Circle.elem = base_decl %Shape.ref, element0 [template]
// CHECK:STDOUT:   %GetPosition.decl: %GetPosition.type = fn_decl @GetPosition [template = constants.%GetPosition] {
// CHECK:STDOUT:     %self.patt: %Circle = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Circle = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %tuple.type.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Circle [template = constants.%Circle]
// CHECK:STDOUT:     %int_32.loc12_36: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc12_36: init type = call constants.%Int(%int_32.loc12_36) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc12_41: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc12_41: init type = call constants.%Int(%int_32.loc12_41) [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_44.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc12_36, %int.make_type_signed.loc12_41)
// CHECK:STDOUT:     %.loc12_44.2: type = value_of_initializer %int.make_type_signed.loc12_36 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_44.3: type = converted %int.make_type_signed.loc12_36, %.loc12_44.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_44.4: type = value_of_initializer %int.make_type_signed.loc12_41 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_44.5: type = converted %int.make_type_signed.loc12_41, %.loc12_44.4 [template = constants.%i32]
// CHECK:STDOUT:     %.loc12_44.6: type = converted %.loc12_44.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %self.param: %Circle = value_param runtime_param0
// CHECK:STDOUT:     %self: %Circle = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %tuple.type.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.1 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Circle
// CHECK:STDOUT:   .base = %.loc10
// CHECK:STDOUT:   .GetPosition = %GetPosition.decl
// CHECK:STDOUT:   extend %Shape.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetPosition[%self.param_patt: %Circle]() -> %return.param_patt: %tuple.type.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc13_13: %Circle = name_ref self, %self
// CHECK:STDOUT:   %x.ref: %Shape.elem = name_ref x, @Shape.%.loc5_18 [template = @Shape.%.loc5_18]
// CHECK:STDOUT:   %.loc13_17.1: ref %Shape = class_element_access %self.ref.loc13_13, element0
// CHECK:STDOUT:   %.loc13_17.2: ref %Shape = converted %self.ref.loc13_13, %.loc13_17.1
// CHECK:STDOUT:   %.loc13_17.3: ref %i32 = class_element_access %.loc13_17.2, element0
// CHECK:STDOUT:   %self.ref.loc13_21: %Circle = name_ref self, %self
// CHECK:STDOUT:   %y.ref: %Shape.elem = name_ref y, @Shape.%.loc6_18 [template = @Shape.%.loc6_18]
// CHECK:STDOUT:   %.loc13_25.1: ref %Shape = class_element_access %self.ref.loc13_21, element0
// CHECK:STDOUT:   %.loc13_25.2: ref %Shape = converted %self.ref.loc13_21, %.loc13_25.1
// CHECK:STDOUT:   %.loc13_25.3: ref %i32 = class_element_access %.loc13_25.2, element1
// CHECK:STDOUT:   %.loc13_27.1: %tuple.type.2 = tuple_literal (%.loc13_17.3, %.loc13_25.3)
// CHECK:STDOUT:   %.loc13_17.4: %i32 = bind_value %.loc13_17.3
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc13_27.2: init %i32 = initialize_from %.loc13_17.4 to %tuple.elem0
// CHECK:STDOUT:   %.loc13_25.4: %i32 = bind_value %.loc13_25.3
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc13_27.3: init %i32 = initialize_from %.loc13_25.4 to %tuple.elem1
// CHECK:STDOUT:   %.loc13_27.4: init %tuple.type.2 = tuple_init (%.loc13_27.2, %.loc13_27.3) to %return
// CHECK:STDOUT:   %.loc13_28: init %tuple.type.2 = converted %.loc13_27.1, %.loc13_27.4
// CHECK:STDOUT:   return %.loc13_28 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- shadowing_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %struct_type.base.1: type = struct_type {.base: %A} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.base.1 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.base.3: type = struct_type {.base: %B} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.base.3 [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
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
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc9: %B.elem = base_decl %A.ref, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_21.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_21.2: type = converted %int.make_type_signed, %.loc10_21.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.1 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F [private] = %F.decl
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc14: %C.elem = base_decl %B.ref, element0 [template]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %.loc15_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_26.2: type = converted %.loc15_26.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.3 [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc14
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[%self.param_patt: %C]() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C = name_ref self, %self
// CHECK:STDOUT:   %F.ref: %F.type.1 = name_ref F, @A.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   %.loc15_44.1: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:   %.loc15_44.2: ref %empty_tuple.type = temporary %.loc15_44.1, %F.call
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_45: %empty_tuple.type = converted %F.call, %tuple [template = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc15_45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inherited_member_access.carbon
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
// CHECK:STDOUT:   %SomeProtectedFunction.type: type = fn_type @SomeProtectedFunction [template]
// CHECK:STDOUT:   %SomeProtectedFunction: %SomeProtectedFunction.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %A} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.base [template]
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
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_32.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_32.2: type = converted %int.make_type_signed, %.loc5_32.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_5, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_5) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc5_39.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc5_39.2: %i32 = converted %int_5, %.loc5_39.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %SOME_CONSTANT: %i32 = bind_name SOME_CONSTANT, %.loc5_39.2
// CHECK:STDOUT:   %SomeProtectedFunction.decl: %SomeProtectedFunction.type = fn_decl @SomeProtectedFunction [template = constants.%SomeProtectedFunction] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_43.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_43.2: type = converted %int.make_type_signed, %.loc6_43.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .SOME_CONSTANT [protected] = %SOME_CONSTANT
// CHECK:STDOUT:   .SomeProtectedFunction [protected] = %SomeProtectedFunction.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc12: %B.elem = base_decl %A.ref, element0 [template]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_13.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_13.2: type = converted %int.make_type_signed, %.loc14_13.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_13.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc18_13.2: type = converted %int.make_type_signed, %.loc18_13.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc12
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SomeProtectedFunction() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_5, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_5) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc7_13.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc7_13.2: %i32 = converted %int_5, %.loc7_13.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   return %.loc7_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %SOME_CONSTANT.ref: %i32 = name_ref SOME_CONSTANT, @A.%SOME_CONSTANT
// CHECK:STDOUT:   return %SOME_CONSTANT.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %SomeProtectedFunction.ref: %SomeProtectedFunction.type = name_ref SomeProtectedFunction, @A.%SomeProtectedFunction.decl [template = constants.%SomeProtectedFunction]
// CHECK:STDOUT:   %SomeProtectedFunction.call: init %i32 = call %SomeProtectedFunction.ref()
// CHECK:STDOUT:   %.loc19_37.1: %i32 = value_of_initializer %SomeProtectedFunction.call
// CHECK:STDOUT:   %.loc19_37.2: %i32 = converted %SomeProtectedFunction.call, %.loc19_37.1
// CHECK:STDOUT:   return %.loc19_37.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inherited_private_field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Shape: type = class_type @Shape [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Shape.elem: type = unbound_element_type %Shape, %i32 [template]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %i32} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.y [template]
// CHECK:STDOUT:   %Square: type = class_type @Square [template]
// CHECK:STDOUT:   %Square.elem: type = unbound_element_type %Square, %Shape [template]
// CHECK:STDOUT:   %GetPosition.type: type = fn_type @GetPosition [template]
// CHECK:STDOUT:   %GetPosition: %GetPosition.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.base.1: type = struct_type {.base: %Shape} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.base.1 [template]
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
// CHECK:STDOUT:     .Shape = %Shape.decl
// CHECK:STDOUT:     .Square = %Square.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Shape.decl: type = class_decl @Shape [template = constants.%Shape] {} {}
// CHECK:STDOUT:   %Square.decl: type = class_decl @Square [template = constants.%Square] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Shape {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_18.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_18.2: type = converted %int.make_type_signed, %.loc5_18.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_16: %Shape.elem = field_decl y, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.y [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Shape
// CHECK:STDOUT:   .y [private] = %.loc5_16
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Square {
// CHECK:STDOUT:   %Shape.ref: type = name_ref Shape, file.%Shape.decl [template = constants.%Shape]
// CHECK:STDOUT:   %.loc9: %Square.elem = base_decl %Shape.ref, element0 [template]
// CHECK:STDOUT:   %GetPosition.decl: %GetPosition.type = fn_decl @GetPosition [template = constants.%GetPosition] {
// CHECK:STDOUT:     %self.patt: %Square = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Square = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Square [template = constants.%Square]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_35.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_35.2: type = converted %int.make_type_signed, %.loc11_35.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: %Square = value_param runtime_param0
// CHECK:STDOUT:     %self: %Square = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.1 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Square
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .GetPosition = %GetPosition.decl
// CHECK:STDOUT:   extend %Shape.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetPosition[%self.param_patt: %Square]() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Square = name_ref self, %self
// CHECK:STDOUT:   %y.ref: <error> = name_ref y, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- noninstance_private_on_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F [private] = %F.decl
// CHECK:STDOUT:   .G = %G.decl
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
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @C.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- noninstance_protected_on_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F [protected] = %F.decl
// CHECK:STDOUT:   .G = %G.decl
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
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @C.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_noninstance_private_on_parent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.base.1: type = struct_type {.base: %B} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.base.1 [template]
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
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .F [private] = %F.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc9: %C.elem = base_decl %B.ref, element0 [template]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.1 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   extend %B.ref
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
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- noninstance_protected_on_parent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.base.1: type = struct_type {.base: %B} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.base.1 [template]
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
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .F [protected] = %F.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc9: %C.elem = base_decl %B.ref, element0 [template]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.1 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   extend %B.ref
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
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @B.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_inherited_access.carbon
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
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Internal: type = class_type @Internal [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %Internal [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %SomeFunc.type: type = fn_type @SomeFunc [template]
// CHECK:STDOUT:   %SomeFunc: %SomeFunc.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.internal.1: type = struct_type {.internal: %Internal} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.internal.1 [template]
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
// CHECK:STDOUT:     .Internal = %Internal.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %Internal.decl: type = class_decl @Internal [template = constants.%Internal] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%int_32.loc5) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_42.1: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_42.2: type = converted %int.make_type_signed.loc5, %.loc5_42.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_5.loc5: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %impl.elem0.loc5: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc5: <bound method> = bound_method %int_5.loc5, %impl.elem0.loc5 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc5: <specific function> = specific_function %Convert.bound.loc5, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc5: init %i32 = call %Convert.specific_fn.loc5(%int_5.loc5) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc5_49.1: %i32 = value_of_initializer %int.convert_checked.loc5 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc5_49.2: %i32 = converted %int_5.loc5, %.loc5_49.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %SOME_PROTECTED_CONSTANT: %i32 = bind_name SOME_PROTECTED_CONSTANT, %.loc5_49.2
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%int_32.loc6) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_38.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_38.2: type = converted %int.make_type_signed.loc6, %.loc6_38.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_5.loc6: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %impl.elem0.loc6: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc6: <bound method> = bound_method %int_5.loc6, %impl.elem0.loc6 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc6: <specific function> = specific_function %Convert.bound.loc6, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc6: init %i32 = call %Convert.specific_fn.loc6(%int_5.loc6) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc6_45.1: %i32 = value_of_initializer %int.convert_checked.loc6 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc6_45.2: %i32 = converted %int_5.loc6, %.loc6_45.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %SOME_PRIVATE_CONSTANT: %i32 = bind_name SOME_PRIVATE_CONSTANT, %.loc6_45.2
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .SOME_PROTECTED_CONSTANT [protected] = %SOME_PROTECTED_CONSTANT
// CHECK:STDOUT:   .SOME_PRIVATE_CONSTANT [private] = %SOME_PRIVATE_CONSTANT
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Internal {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_36.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc10_36.2: type = converted %int.make_type_signed, %.loc10_36.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_5, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_5) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc10_43.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc10_43.2: %i32 = converted %int_5, %.loc10_43.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %INTERNAL_CONSTANT: %i32 = bind_name INTERNAL_CONSTANT, %.loc10_43.2
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Internal
// CHECK:STDOUT:   .INTERNAL_CONSTANT [protected] = %INTERNAL_CONSTANT
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %Internal.ref: type = name_ref Internal, file.%Internal.decl [template = constants.%Internal]
// CHECK:STDOUT:   %.loc14: %B.elem = field_decl internal, element0 [template]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_13.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_13.2: type = converted %int.make_type_signed, %.loc16_13.1 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SomeFunc.decl: %SomeFunc.type = fn_decl @SomeFunc [template = constants.%SomeFunc] {
// CHECK:STDOUT:     %self.patt: %B = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %B = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [template = constants.%B]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc36_32.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc36_32.2: type = converted %int.make_type_signed, %.loc36_32.1 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: %B = value_param runtime_param0
// CHECK:STDOUT:     %self: %B = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.internal.1 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .internal [private] = %.loc14
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .SomeFunc = %SomeFunc.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref.loc24: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %SOME_PRIVATE_CONSTANT.ref: <error> = name_ref SOME_PRIVATE_CONSTANT, <error> [template = <error>]
// CHECK:STDOUT:   %A.ref.loc33: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %SOME_PROTECTED_CONSTANT.ref: <error> = name_ref SOME_PROTECTED_CONSTANT, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SomeFunc[%self.param_patt: %B]() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %B = name_ref self, %self
// CHECK:STDOUT:   %internal.ref: %B.elem = name_ref internal, @B.%.loc14 [template = @B.%.loc14]
// CHECK:STDOUT:   %.loc44_16.1: ref %Internal = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc44_16.2: %Internal = bind_value %.loc44_16.1
// CHECK:STDOUT:   %INTERNAL_CONSTANT.ref: <error> = name_ref INTERNAL_CONSTANT, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_compound_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [template]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.x [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.base.1: type = struct_type {.base: %A} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.base.1 [template]
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
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_18.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_18.2: type = converted %int.make_type_signed, %.loc5_18.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_16: %A.elem = field_decl x, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x [private] = %.loc5_16
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc9: %B.elem = base_decl %A.ref, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %B = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %B = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [template = constants.%B]
// CHECK:STDOUT:     %self.param: %B = value_param runtime_param0
// CHECK:STDOUT:     %self: %B = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.1 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.param_patt: %B]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %B = name_ref self, %self
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %x.ref: <error> = name_ref x, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inherited_compound_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [template]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %i32} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.x [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.base.1: type = struct_type {.base: %A} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.base.1 [template]
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
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_20.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_20.2: type = converted %int.make_type_signed, %.loc5_20.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_18: %A.elem = field_decl x, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x [protected] = %.loc5_18
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc9: %B.elem = base_decl %A.ref, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %self.patt: %B = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %B = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [template = constants.%B]
// CHECK:STDOUT:     %self.param: %B = value_param runtime_param0
// CHECK:STDOUT:     %self: %B = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.1 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self.param_patt: %B]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %B = name_ref self, %self
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %x.ref: %A.elem = name_ref x, @A.%.loc5_18 [template = @A.%.loc5_18]
// CHECK:STDOUT:   %.loc12_9.1: ref %A = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc12_9.2: ref %A = converted %self.ref, %.loc12_9.1
// CHECK:STDOUT:   %.loc12_9.3: ref %i32 = class_element_access %.loc12_9.2, element0
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
