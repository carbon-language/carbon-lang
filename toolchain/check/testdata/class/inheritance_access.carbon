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
    // CHECK:STDERR: fail_inherited_private_field_access.carbon:[[@LINE+7]]:12: ERROR: Cannot access private member `y` of type `Shape`.
    // CHECK:STDERR:     return self.y;
    // CHECK:STDERR:            ^~~~~~
    // CHECK:STDERR: fail_inherited_private_field_access.carbon:[[@LINE-10]]:15: The private member `y` is defined here.
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
  // CHECK:STDERR: fail_noninstance_private_on_parent.carbon:[[@LINE+7]]:12: ERROR: Cannot access private member `F` of type `B`.
  // CHECK:STDERR:   fn G() { Self.F(); }
  // CHECK:STDERR:            ^~~~~~
  // CHECK:STDERR: fail_noninstance_private_on_parent.carbon:[[@LINE-8]]:3: The private member `F` is defined here.
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
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE+7]]:5: ERROR: Cannot access private member `SOME_PRIVATE_CONSTANT` of type `A`.
    // CHECK:STDERR:     A.SOME_PRIVATE_CONSTANT;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE-14]]:15: The private member `SOME_PRIVATE_CONSTANT` is defined here.
    // CHECK:STDERR:   private let SOME_PRIVATE_CONSTANT: i32 = 5;
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    A.SOME_PRIVATE_CONSTANT;

    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE+7]]:12: ERROR: Cannot access protected member `SOME_PROTECTED_CONSTANT` of type `A`.
    // CHECK:STDERR:     return A.SOME_PROTECTED_CONSTANT;
    // CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE-24]]:17: The protected member `SOME_PROTECTED_CONSTANT` is defined here.
    // CHECK:STDERR:   protected let SOME_PROTECTED_CONSTANT: i32 = 5;
    // CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    return A.SOME_PROTECTED_CONSTANT;
  }

  fn SomeFunc[self: Self]() -> i32{
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE+6]]:12: ERROR: Cannot access protected member `INTERNAL_CONSTANT` of type `Internal`.
    // CHECK:STDERR:     return self.internal.INTERNAL_CONSTANT;
    // CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE-30]]:17: The protected member `INTERNAL_CONSTANT` is defined here.
    // CHECK:STDERR:   protected let INTERNAL_CONSTANT: i32 = 5;
    // CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~
    return self.internal.INTERNAL_CONSTANT;
  }
}

// CHECK:STDOUT: --- instance_protected_field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Shape: type = class_type @Shape [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Shape, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.x: i32, .y: i32} [template]
// CHECK:STDOUT:   %Circle: type = class_type @Circle [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %Circle, %Shape [template]
// CHECK:STDOUT:   %.6: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %GetPosition.type: type = fn_type @GetPosition [template]
// CHECK:STDOUT:   %GetPosition: %GetPosition.type = struct_value () [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: %Shape} [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %.10: type = struct_type {.base: %.4} [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.8 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Shape = %Shape.decl
// CHECK:STDOUT:     .Circle = %Circle.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Shape.decl: type = class_decl @Shape [template = constants.%Shape] {}
// CHECK:STDOUT:   %Circle.decl: type = class_decl @Circle [template = constants.%Circle] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Shape {
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_20.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_20.2: type = converted %int.make_type_32.loc5, %.loc5_20.1 [template = i32]
// CHECK:STDOUT:   %.loc5_18: %.2 = field_decl x, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_20.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_20.2: type = converted %int.make_type_32.loc6, %.loc6_20.1 [template = i32]
// CHECK:STDOUT:   %.loc6_18: %.2 = field_decl y, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Shape
// CHECK:STDOUT:   .x [protected] = %.loc5_18
// CHECK:STDOUT:   .y [protected] = %.loc6_18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Circle {
// CHECK:STDOUT:   %Shape.ref: type = name_ref Shape, file.%Shape.decl [template = constants.%Shape]
// CHECK:STDOUT:   %.loc10: %.5 = base_decl %Shape, element0 [template]
// CHECK:STDOUT:   %GetPosition.decl: %GetPosition.type = fn_decl @GetPosition [template = constants.%GetPosition] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Circle [template = constants.%Circle]
// CHECK:STDOUT:     %self.loc12_18.1: %Circle = param self
// CHECK:STDOUT:     %self.loc12_18.2: %Circle = bind_name self, %self.loc12_18.1
// CHECK:STDOUT:     %int.make_type_32.loc12_36: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc12_41: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_44.1: %.6 = tuple_literal (%int.make_type_32.loc12_36, %int.make_type_32.loc12_41)
// CHECK:STDOUT:     %.loc12_44.2: type = value_of_initializer %int.make_type_32.loc12_36 [template = i32]
// CHECK:STDOUT:     %.loc12_44.3: type = converted %int.make_type_32.loc12_36, %.loc12_44.2 [template = i32]
// CHECK:STDOUT:     %.loc12_44.4: type = value_of_initializer %int.make_type_32.loc12_41 [template = i32]
// CHECK:STDOUT:     %.loc12_44.5: type = converted %int.make_type_32.loc12_41, %.loc12_44.4 [template = i32]
// CHECK:STDOUT:     %.loc12_44.6: type = converted %.loc12_44.1, constants.%.7 [template = constants.%.7]
// CHECK:STDOUT:     %return.var: ref %.7 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Circle
// CHECK:STDOUT:   .base = %.loc10
// CHECK:STDOUT:   .GetPosition = %GetPosition.decl
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetPosition[@Circle.%self.loc12_18.2: %Circle]() -> @Circle.%return.var: %.7 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc13_13: %Circle = name_ref self, @Circle.%self.loc12_18.2
// CHECK:STDOUT:   %x.ref: %.2 = name_ref x, @Shape.%.loc5_18 [template = @Shape.%.loc5_18]
// CHECK:STDOUT:   %.loc13_17.1: ref %Shape = class_element_access %self.ref.loc13_13, element0
// CHECK:STDOUT:   %.loc13_17.2: ref %Shape = converted %self.ref.loc13_13, %.loc13_17.1
// CHECK:STDOUT:   %.loc13_17.3: ref i32 = class_element_access %.loc13_17.2, element0
// CHECK:STDOUT:   %self.ref.loc13_21: %Circle = name_ref self, @Circle.%self.loc12_18.2
// CHECK:STDOUT:   %y.ref: %.2 = name_ref y, @Shape.%.loc6_18 [template = @Shape.%.loc6_18]
// CHECK:STDOUT:   %.loc13_25.1: ref %Shape = class_element_access %self.ref.loc13_21, element0
// CHECK:STDOUT:   %.loc13_25.2: ref %Shape = converted %self.ref.loc13_21, %.loc13_25.1
// CHECK:STDOUT:   %.loc13_25.3: ref i32 = class_element_access %.loc13_25.2, element1
// CHECK:STDOUT:   %.loc13_27.1: %.7 = tuple_literal (%.loc13_17.3, %.loc13_25.3)
// CHECK:STDOUT:   %.loc13_17.4: i32 = bind_value %.loc13_17.3
// CHECK:STDOUT:   %.loc13_27.2: ref i32 = tuple_access @Circle.%return.var, element0
// CHECK:STDOUT:   %.loc13_27.3: init i32 = initialize_from %.loc13_17.4 to %.loc13_27.2
// CHECK:STDOUT:   %.loc13_25.4: i32 = bind_value %.loc13_25.3
// CHECK:STDOUT:   %.loc13_27.4: ref i32 = tuple_access @Circle.%return.var, element1
// CHECK:STDOUT:   %.loc13_27.5: init i32 = initialize_from %.loc13_25.4 to %.loc13_27.4
// CHECK:STDOUT:   %.loc13_27.6: init %.7 = tuple_init (%.loc13_27.3, %.loc13_27.5) to @Circle.%return.var
// CHECK:STDOUT:   %.loc13_28: init %.7 = converted %.loc13_27.1, %.loc13_27.6
// CHECK:STDOUT:   return %.loc13_28 to @Circle.%return.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- shadowing_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = struct_type {.base: %A} [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: %.3} [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %.8: type = unbound_element_type %C, %B [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.9: type = struct_type {.base: %B} [template]
// CHECK:STDOUT:   %.10: type = struct_type {.base: %.7} [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.9 [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc9: %.4 = base_decl %A, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc10_21.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc10_21.2: type = converted %int.make_type_32, %.loc10_21.1 [template = i32]
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F [private] = %F.decl
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc14: %.8 = base_decl %B, element0 [template]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %self.loc15_8.1: %C = param self
// CHECK:STDOUT:     %self.loc15_8.2: %C = bind_name self, %self.loc15_8.1
// CHECK:STDOUT:     %.loc15_26.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc15_26.2: type = converted %.loc15_26.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return.var: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc14
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   extend name_scope3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[@C.%self.loc15_8.2: %C]() -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C = name_ref self, @C.%self.loc15_8.2
// CHECK:STDOUT:   %F.ref: %F.type.1 = name_ref F, @A.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %F.call: init %.1 = call %F.ref()
// CHECK:STDOUT:   %.loc15_43.1: ref %.1 = temporary_storage
// CHECK:STDOUT:   %.loc15_43.2: ref %.1 = temporary %.loc15_43.1, %F.call
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc15_45: %.1 = converted %F.call, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc15_45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inherited_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %SomeProtectedFunction.type: type = fn_type @SomeProtectedFunction [template]
// CHECK:STDOUT:   %SomeProtectedFunction: %SomeProtectedFunction.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: %A} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_32.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_32.2: type = converted %int.make_type_32.loc5, %.loc5_32.1 [template = i32]
// CHECK:STDOUT:   %.loc5_38: i32 = int_literal 5 [template = constants.%.2]
// CHECK:STDOUT:   %SOME_CONSTANT: i32 = bind_name SOME_CONSTANT, %.loc5_38
// CHECK:STDOUT:   %SomeProtectedFunction.decl: %SomeProtectedFunction.type = fn_decl @SomeProtectedFunction [template = constants.%SomeProtectedFunction] {
// CHECK:STDOUT:     %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_43.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:     %.loc6_43.2: type = converted %int.make_type_32.loc6, %.loc6_43.1 [template = i32]
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .SOME_CONSTANT [protected] = %SOME_CONSTANT
// CHECK:STDOUT:   .SomeProtectedFunction [protected] = %SomeProtectedFunction.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc12: %.5 = base_decl %A, element0 [template]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc14_13.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:     %.loc14_13.2: type = converted %int.make_type_32.loc14, %.loc14_13.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc14: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %int.make_type_32.loc18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_13.1: type = value_of_initializer %int.make_type_32.loc18 [template = i32]
// CHECK:STDOUT:     %.loc18_13.2: type = converted %int.make_type_32.loc18, %.loc18_13.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc18: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc12
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SomeProtectedFunction() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: i32 = int_literal 5 [template = constants.%.2]
// CHECK:STDOUT:   return %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %SOME_CONSTANT.ref: i32 = name_ref SOME_CONSTANT, @A.%SOME_CONSTANT
// CHECK:STDOUT:   return %SOME_CONSTANT.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %SomeProtectedFunction.ref: %SomeProtectedFunction.type = name_ref SomeProtectedFunction, @A.%SomeProtectedFunction.decl [template = constants.%SomeProtectedFunction]
// CHECK:STDOUT:   %SomeProtectedFunction.call: init i32 = call %SomeProtectedFunction.ref()
// CHECK:STDOUT:   %.loc19_37.1: i32 = value_of_initializer %SomeProtectedFunction.call
// CHECK:STDOUT:   %.loc19_37.2: i32 = converted %SomeProtectedFunction.call, %.loc19_37.1
// CHECK:STDOUT:   return %.loc19_37.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inherited_private_field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Shape: type = class_type @Shape [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Shape, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.y: i32} [template]
// CHECK:STDOUT:   %Square: type = class_type @Square [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %Square, %Shape [template]
// CHECK:STDOUT:   %GetPosition.type: type = fn_type @GetPosition [template]
// CHECK:STDOUT:   %GetPosition: %GetPosition.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: %Shape} [template]
// CHECK:STDOUT:   %.7: type = struct_type {.base: %.4} [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Shape = %Shape.decl
// CHECK:STDOUT:     .Square = %Square.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Shape.decl: type = class_decl @Shape [template = constants.%Shape] {}
// CHECK:STDOUT:   %Square.decl: type = class_decl @Square [template = constants.%Square] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Shape {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_18.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc5_18.2: type = converted %int.make_type_32, %.loc5_18.1 [template = i32]
// CHECK:STDOUT:   %.loc5_16: %.2 = field_decl y, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Shape
// CHECK:STDOUT:   .y [private] = %.loc5_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Square {
// CHECK:STDOUT:   %Shape.ref: type = name_ref Shape, file.%Shape.decl [template = constants.%Shape]
// CHECK:STDOUT:   %.loc9: %.5 = base_decl %Shape, element0 [template]
// CHECK:STDOUT:   %GetPosition.decl: %GetPosition.type = fn_decl @GetPosition [template = constants.%GetPosition] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Square [template = constants.%Square]
// CHECK:STDOUT:     %self.loc11_18.1: %Square = param self
// CHECK:STDOUT:     %self.loc11_18.2: %Square = bind_name self, %self.loc11_18.1
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_35.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc11_35.2: type = converted %int.make_type_32, %.loc11_35.1 [template = i32]
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Square
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .GetPosition = %GetPosition.decl
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetPosition[@Square.%self.loc11_18.2: %Square]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Square = name_ref self, @Square.%self.loc11_18.2
// CHECK:STDOUT:   %y.ref: <error> = name_ref y, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- noninstance_private_on_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F [private] = %F.decl
// CHECK:STDOUT:   .G = %G.decl
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
// CHECK:STDOUT:   %F.call: init %.1 = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- noninstance_protected_on_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F [protected] = %F.decl
// CHECK:STDOUT:   .G = %G.decl
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
// CHECK:STDOUT:   %F.call: init %.1 = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_noninstance_private_on_parent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %C, %B [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = struct_type {.base: %B} [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: %.3} [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.5 [template]
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
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .F [private] = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc9: %.4 = base_decl %B, element0 [template]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   extend name_scope2
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
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %C, %B [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = struct_type {.base: %B} [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: %.3} [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.5 [template]
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
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .F [protected] = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc9: %.4 = base_decl %B, element0 [template]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   extend name_scope2
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
// CHECK:STDOUT:   %F.call: init %.1 = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_inherited_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %Internal: type = class_type @Internal [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %B, %Internal [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %SomeFunc.type: type = fn_type @SomeFunc [template]
// CHECK:STDOUT:   %SomeFunc: %SomeFunc.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = struct_type {.internal: %Internal} [template]
// CHECK:STDOUT:   %.7: type = struct_type {.internal: %.4} [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %Internal.decl: type = class_decl @Internal [template = constants.%Internal] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_42.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_42.2: type = converted %int.make_type_32.loc5, %.loc5_42.1 [template = i32]
// CHECK:STDOUT:   %.loc5_48: i32 = int_literal 5 [template = constants.%.2]
// CHECK:STDOUT:   %SOME_PROTECTED_CONSTANT: i32 = bind_name SOME_PROTECTED_CONSTANT, %.loc5_48
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_38.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_38.2: type = converted %int.make_type_32.loc6, %.loc6_38.1 [template = i32]
// CHECK:STDOUT:   %.loc6_44: i32 = int_literal 5 [template = constants.%.2]
// CHECK:STDOUT:   %SOME_PRIVATE_CONSTANT: i32 = bind_name SOME_PRIVATE_CONSTANT, %.loc6_44
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .SOME_PROTECTED_CONSTANT [protected] = %SOME_PROTECTED_CONSTANT
// CHECK:STDOUT:   .SOME_PRIVATE_CONSTANT [private] = %SOME_PRIVATE_CONSTANT
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Internal {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc10_36.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc10_36.2: type = converted %int.make_type_32, %.loc10_36.1 [template = i32]
// CHECK:STDOUT:   %.loc10_42: i32 = int_literal 5 [template = constants.%.2]
// CHECK:STDOUT:   %INTERNAL_CONSTANT: i32 = bind_name INTERNAL_CONSTANT, %.loc10_42
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Internal
// CHECK:STDOUT:   .INTERNAL_CONSTANT [protected] = %INTERNAL_CONSTANT
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %Internal.ref: type = name_ref Internal, file.%Internal.decl [template = constants.%Internal]
// CHECK:STDOUT:   %.loc14: %.5 = field_decl internal, element0 [template]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %int.make_type_32.loc16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_13.1: type = value_of_initializer %int.make_type_32.loc16 [template = i32]
// CHECK:STDOUT:     %.loc16_13.2: type = converted %int.make_type_32.loc16, %.loc16_13.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc16: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SomeFunc.decl: %SomeFunc.type = fn_decl @SomeFunc [template = constants.%SomeFunc] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [template = constants.%B]
// CHECK:STDOUT:     %self.loc36_15.1: %B = param self
// CHECK:STDOUT:     %self.loc36_15.2: %B = bind_name self, %self.loc36_15.1
// CHECK:STDOUT:     %int.make_type_32.loc36: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc36_32.1: type = value_of_initializer %int.make_type_32.loc36 [template = i32]
// CHECK:STDOUT:     %.loc36_32.2: type = converted %int.make_type_32.loc36, %.loc36_32.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc36: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .internal [private] = %.loc14
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .SomeFunc = %SomeFunc.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref.loc24: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %SOME_PRIVATE_CONSTANT.ref: <error> = name_ref SOME_PRIVATE_CONSTANT, <error> [template = <error>]
// CHECK:STDOUT:   %A.ref.loc33: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %SOME_PROTECTED_CONSTANT.ref: <error> = name_ref SOME_PROTECTED_CONSTANT, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SomeFunc[@B.%self.loc36_15.2: %B]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %B = name_ref self, @B.%self.loc36_15.2
// CHECK:STDOUT:   %internal.ref: %.5 = name_ref internal, @B.%.loc14 [template = @B.%.loc14]
// CHECK:STDOUT:   %.loc43_16.1: ref %Internal = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc43_16.2: %Internal = bind_value %.loc43_16.1
// CHECK:STDOUT:   %INTERNAL_CONSTANT.ref: <error> = name_ref INTERNAL_CONSTANT, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
