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
    // CHECK:STDERR: fail_compound_member_access.carbon:[[@LINE+7]]:11: error: cannot access private member `x` of type `A` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     self.(A.x);
    // CHECK:STDERR:           ^~~
    // CHECK:STDERR: fail_compound_member_access.carbon:[[@LINE-9]]:15: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:   private var x: i32;
    // CHECK:STDERR:               ^~~~~~
    // CHECK:STDERR:
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
// CHECK:STDOUT:   %Shape: type = class_type @Shape [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Shape.elem: type = unbound_element_type %Shape, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.y: type = struct_type {.x: %i32, .y: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.70a: <witness> = complete_type_witness %struct_type.x.y [concrete]
// CHECK:STDOUT:   %Circle: type = class_type @Circle [concrete]
// CHECK:STDOUT:   %Circle.elem: type = unbound_element_type %Circle, %Shape [concrete]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Circle [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.511: type = pattern_type %tuple.type.d07 [concrete]
// CHECK:STDOUT:   %Circle.GetPosition.type: type = fn_type @Circle.GetPosition [concrete]
// CHECK:STDOUT:   %Circle.GetPosition: %Circle.GetPosition.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.0ed: type = struct_type {.base: %Shape} [concrete]
// CHECK:STDOUT:   %complete_type.a2b: <witness> = complete_type_witness %struct_type.base.0ed [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc17_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Shape = %Shape.decl
// CHECK:STDOUT:     .Circle = %Circle.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Shape.decl: type = class_decl @Shape [concrete = constants.%Shape] {} {}
// CHECK:STDOUT:   %Circle.decl: type = class_decl @Circle [concrete = constants.%Circle] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Shape {
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5: %Shape.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc6: %Shape.elem = field_decl y, element1 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x.y [concrete = constants.%complete_type.70a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Shape
// CHECK:STDOUT:   .x [protected] = %.loc5
// CHECK:STDOUT:   .y [protected] = %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Circle {
// CHECK:STDOUT:   %Shape.ref: type = name_ref Shape, file.%Shape.decl [concrete = constants.%Shape]
// CHECK:STDOUT:   %.loc10: %Circle.elem = base_decl %Shape.ref, element0 [concrete]
// CHECK:STDOUT:   %Circle.GetPosition.decl: %Circle.GetPosition.type = fn_decl @Circle.GetPosition [concrete = constants.%Circle.GetPosition] {
// CHECK:STDOUT:     %self.patt: %pattern_type.ce2 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ce2 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.511 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.511 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12_36: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_36: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc12_41: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_41: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12_44.1: %tuple.type.24b = tuple_literal (%i32.loc12_36, %i32.loc12_41)
// CHECK:STDOUT:     %.loc12_44.2: type = converted %.loc12_44.1, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     %self.param: %Circle = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Circle [concrete = constants.%Circle]
// CHECK:STDOUT:     %self: %Circle = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.d07 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.d07 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.0ed [concrete = constants.%complete_type.a2b]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Circle
// CHECK:STDOUT:   .Shape = <poisoned>
// CHECK:STDOUT:   .base = %.loc10
// CHECK:STDOUT:   .GetPosition = %Circle.GetPosition.decl
// CHECK:STDOUT:   .x = <poisoned>
// CHECK:STDOUT:   .y = <poisoned>
// CHECK:STDOUT:   extend %Shape.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Circle.GetPosition(%self.param: %Circle) -> %return.param: %tuple.type.d07 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc13_13: %Circle = name_ref self, %self
// CHECK:STDOUT:   %x.ref: %Shape.elem = name_ref x, @Shape.%.loc5 [concrete = @Shape.%.loc5]
// CHECK:STDOUT:   %.loc13_17.1: ref %Shape = class_element_access %self.ref.loc13_13, element0
// CHECK:STDOUT:   %.loc13_17.2: ref %Shape = converted %self.ref.loc13_13, %.loc13_17.1
// CHECK:STDOUT:   %.loc13_17.3: ref %i32 = class_element_access %.loc13_17.2, element0
// CHECK:STDOUT:   %self.ref.loc13_21: %Circle = name_ref self, %self
// CHECK:STDOUT:   %y.ref: %Shape.elem = name_ref y, @Shape.%.loc6 [concrete = @Shape.%.loc6]
// CHECK:STDOUT:   %.loc13_25.1: ref %Shape = class_element_access %self.ref.loc13_21, element0
// CHECK:STDOUT:   %.loc13_25.2: ref %Shape = converted %self.ref.loc13_21, %.loc13_25.1
// CHECK:STDOUT:   %.loc13_25.3: ref %i32 = class_element_access %.loc13_25.2, element1
// CHECK:STDOUT:   %.loc13_27.1: %tuple.type.d07 = tuple_literal (%.loc13_17.3, %.loc13_25.3)
// CHECK:STDOUT:   %.loc13_17.4: %i32 = bind_value %.loc13_17.3
// CHECK:STDOUT:   %impl.elem0.loc13_17: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc13_17.1: <bound method> = bound_method %.loc13_17.4, %impl.elem0.loc13_17
// CHECK:STDOUT:   %specific_fn.loc13_17: <specific function> = specific_function %impl.elem0.loc13_17, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_17.2: <bound method> = bound_method %.loc13_17.4, %specific_fn.loc13_17
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc13_17: init %i32 = call %bound_method.loc13_17.2(%.loc13_17.4)
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc13_27.2: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc13_17 to %tuple.elem0
// CHECK:STDOUT:   %.loc13_25.4: %i32 = bind_value %.loc13_25.3
// CHECK:STDOUT:   %impl.elem0.loc13_25: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc13_25.1: <bound method> = bound_method %.loc13_25.4, %impl.elem0.loc13_25
// CHECK:STDOUT:   %specific_fn.loc13_25: <specific function> = specific_function %impl.elem0.loc13_25, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_25.2: <bound method> = bound_method %.loc13_25.4, %specific_fn.loc13_25
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc13_25: init %i32 = call %bound_method.loc13_25.2(%.loc13_25.4)
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc13_27.3: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc13_25 to %tuple.elem1
// CHECK:STDOUT:   %.loc13_27.4: init %tuple.type.d07 = tuple_init (%.loc13_27.2, %.loc13_27.3) to %return
// CHECK:STDOUT:   %.loc13_28: init %tuple.type.d07 = converted %.loc13_27.1, %.loc13_27.4
// CHECK:STDOUT:   return %.loc13_28 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- shadowing_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.F: %A.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F [concrete]
// CHECK:STDOUT:   %B.F: %B.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.953: type = struct_type {.base: %A} [concrete]
// CHECK:STDOUT:   %complete_type.020: <witness> = complete_type_witness %struct_type.base.953 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %C.G.type: type = fn_type @C.G [concrete]
// CHECK:STDOUT:   %C.G: %C.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.0ff: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.98e: <witness> = complete_type_witness %struct_type.base.0ff [concrete]
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
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %A.F.decl: %A.F.type = fn_decl @A.F [concrete = constants.%A.F] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F = %A.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc9: %B.elem = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %B.F.decl: %B.F.type = fn_decl @B.F [concrete = constants.%B.F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.953 [concrete = constants.%complete_type.020]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F [private] = %B.F.decl
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc14: %C.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %C.G.decl: %C.G.type = fn_decl @C.G [concrete = constants.%C.G] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_26.2: type = converted %.loc15_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.0ff [concrete = constants.%complete_type.98e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc14
// CHECK:STDOUT:   .G = %C.G.decl
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.F() -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.G(%self.param: %C) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C = name_ref self, %self
// CHECK:STDOUT:   %F.ref: %A.F.type = name_ref F, @A.%A.F.decl [concrete = constants.%A.F]
// CHECK:STDOUT:   %A.F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return %A.F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inherited_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.204: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9e9, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.204) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %A.SomeProtectedFunction.type: type = fn_type @A.SomeProtectedFunction [concrete]
// CHECK:STDOUT:   %A.SomeProtectedFunction: %A.SomeProtectedFunction.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %B.G.type: type = fn_type @B.G [concrete]
// CHECK:STDOUT:   %B.G: %B.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.H.type: type = fn_type @B.H [concrete]
// CHECK:STDOUT:   %B.H: %B.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %A} [concrete]
// CHECK:STDOUT:   %complete_type.020: <witness> = complete_type_witness %struct_type.base [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc17_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %SOME_CONSTANT.patt: %pattern_type.7ce = binding_pattern SOME_CONSTANT [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc5_32: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc5_38.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_38.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc5_38.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc5_38.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc5_38.2: %i32 = converted %int_5, %.loc5_38.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %SOME_CONSTANT: %i32 = bind_name SOME_CONSTANT, %.loc5_38.2
// CHECK:STDOUT:   %A.SomeProtectedFunction.decl: %A.SomeProtectedFunction.type = fn_decl @A.SomeProtectedFunction [concrete = constants.%A.SomeProtectedFunction] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .SOME_CONSTANT [protected] = %SOME_CONSTANT
// CHECK:STDOUT:   .SomeProtectedFunction [protected] = %A.SomeProtectedFunction.decl
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc12: %B.elem = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %B.G.decl: %B.G.type = fn_decl @B.G [concrete = constants.%B.G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.H.decl: %B.H.type = fn_decl @B.H [concrete = constants.%B.H] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base [concrete = constants.%complete_type.020]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .base = %.loc12
// CHECK:STDOUT:   .G = %B.G.decl
// CHECK:STDOUT:   .H = %B.H.decl
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.SomeProtectedFunction() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc7_13.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_13.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc7_13.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc7: init %i32 = converted %int_5, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   return %.loc7 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %SOME_CONSTANT.ref: %i32 = name_ref SOME_CONSTANT, @A.%SOME_CONSTANT
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc15_13.1: <bound method> = bound_method %SOME_CONSTANT.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_13.2: <bound method> = bound_method %SOME_CONSTANT.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc15_13.2(%SOME_CONSTANT.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.H() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %SomeProtectedFunction.ref: %A.SomeProtectedFunction.type = name_ref SomeProtectedFunction, @A.%A.SomeProtectedFunction.decl [concrete = constants.%A.SomeProtectedFunction]
// CHECK:STDOUT:   %A.SomeProtectedFunction.call: init %i32 = call %SomeProtectedFunction.ref()
// CHECK:STDOUT:   return %A.SomeProtectedFunction.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inherited_private_field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Shape: type = class_type @Shape [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Shape.elem: type = unbound_element_type %Shape, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.0f9: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT:   %Square: type = class_type @Square [concrete]
// CHECK:STDOUT:   %Square.elem: type = unbound_element_type %Square, %Shape [concrete]
// CHECK:STDOUT:   %pattern_type.39b: type = pattern_type %Square [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Square.GetPosition.type: type = fn_type @Square.GetPosition [concrete]
// CHECK:STDOUT:   %Square.GetPosition: %Square.GetPosition.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.0ed: type = struct_type {.base: %Shape} [concrete]
// CHECK:STDOUT:   %complete_type.a2b: <witness> = complete_type_witness %struct_type.base.0ed [concrete]
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
// CHECK:STDOUT:     .Shape = %Shape.decl
// CHECK:STDOUT:     .Square = %Square.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Shape.decl: type = class_decl @Shape [concrete = constants.%Shape] {} {}
// CHECK:STDOUT:   %Square.decl: type = class_decl @Square [concrete = constants.%Square] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Shape {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5: %Shape.elem = field_decl y, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.y [concrete = constants.%complete_type.0f9]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Shape
// CHECK:STDOUT:   .y [private] = %.loc5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Square {
// CHECK:STDOUT:   %Shape.ref: type = name_ref Shape, file.%Shape.decl [concrete = constants.%Shape]
// CHECK:STDOUT:   %.loc9: %Square.elem = base_decl %Shape.ref, element0 [concrete]
// CHECK:STDOUT:   %Square.GetPosition.decl: %Square.GetPosition.type = fn_decl @Square.GetPosition [concrete = constants.%Square.GetPosition] {
// CHECK:STDOUT:     %self.patt: %pattern_type.39b = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.39b = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %Square = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Square [concrete = constants.%Square]
// CHECK:STDOUT:     %self: %Square = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.0ed [concrete = constants.%complete_type.a2b]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Square
// CHECK:STDOUT:   .Shape = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .GetPosition = %Square.GetPosition.decl
// CHECK:STDOUT:   .y = <poisoned>
// CHECK:STDOUT:   extend %Shape.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Square.GetPosition(%self.param: %Square) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Square = name_ref self, %self
// CHECK:STDOUT:   %y.ref: <error> = name_ref y, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- noninstance_private_on_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.G.type: type = fn_type @C.G [concrete]
// CHECK:STDOUT:   %C.G: %C.G.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %C.F.decl: %C.F.type = fn_decl @C.F [concrete = constants.%C.F] {} {}
// CHECK:STDOUT:   %C.G.decl: %C.G.type = fn_decl @C.G [concrete = constants.%C.G] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F [private] = %C.F.decl
// CHECK:STDOUT:   .G = %C.G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   %F.ref: %C.F.type = name_ref F, @C.%C.F.decl [concrete = constants.%C.F]
// CHECK:STDOUT:   %C.F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- noninstance_protected_on_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.G.type: type = fn_type @C.G [concrete]
// CHECK:STDOUT:   %C.G: %C.G.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %C.F.decl: %C.F.type = fn_decl @C.F [concrete = constants.%C.F] {} {}
// CHECK:STDOUT:   %C.G.decl: %C.G.type = fn_decl @C.G [concrete = constants.%C.G] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F [protected] = %C.F.decl
// CHECK:STDOUT:   .G = %C.G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   %F.ref: %C.F.type = name_ref F, @C.%C.F.decl [concrete = constants.%C.F]
// CHECK:STDOUT:   %C.F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_noninstance_private_on_parent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F [concrete]
// CHECK:STDOUT:   %B.F: %B.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B [concrete]
// CHECK:STDOUT:   %C.G.type: type = fn_type @C.G [concrete]
// CHECK:STDOUT:   %C.G: %C.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.0ff: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.98e: <witness> = complete_type_witness %struct_type.base.0ff [concrete]
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
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %B.F.decl: %B.F.type = fn_decl @B.F [concrete = constants.%B.F] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .F [private] = %B.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc9: %C.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %C.G.decl: %C.G.type = fn_decl @C.G [concrete = constants.%C.G] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.0ff [concrete = constants.%complete_type.98e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .G = %C.G.decl
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- noninstance_protected_on_parent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %B.F: %B.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B [concrete]
// CHECK:STDOUT:   %C.G.type: type = fn_type @C.G [concrete]
// CHECK:STDOUT:   %C.G: %C.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.0ff: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.98e: <witness> = complete_type_witness %struct_type.base.0ff [concrete]
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
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %B.F.decl: %B.F.type = fn_decl @B.F [concrete = constants.%B.F] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .F [protected] = %B.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc9: %C.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %C.G.decl: %C.G.type = fn_decl @C.G [concrete = constants.%C.G] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.0ff [concrete = constants.%complete_type.98e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .G = %C.G.decl
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   %F.ref: %B.F.type = name_ref F, @B.%B.F.decl [concrete = constants.%B.F]
// CHECK:STDOUT:   %B.F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_inherited_access.carbon
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
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.204: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9e9, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.204) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Internal: type = class_type @Internal [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %Internal [concrete]
// CHECK:STDOUT:   %B.G.type: type = fn_type @B.G [concrete]
// CHECK:STDOUT:   %B.G: %B.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %B.SomeFunc.type: type = fn_type @B.SomeFunc [concrete]
// CHECK:STDOUT:   %B.SomeFunc: %B.SomeFunc.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.internal.6f5: type = struct_type {.internal: %Internal} [concrete]
// CHECK:STDOUT:   %complete_type.d77: <witness> = complete_type_witness %struct_type.internal.6f5 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .Internal = %Internal.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %Internal.decl: type = class_decl @Internal [concrete = constants.%Internal] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %SOME_PROTECTED_CONSTANT.patt: %pattern_type.7ce = binding_pattern SOME_PROTECTED_CONSTANT [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_5.loc5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc5_42: type = splice_block %i32.loc5 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc5: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc5_48.1: <bound method> = bound_method %int_5.loc5, %impl.elem0.loc5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_48.2: <bound method> = bound_method %int_5.loc5, %specific_fn.loc5 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5: init %i32 = call %bound_method.loc5_48.2(%int_5.loc5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc5_48.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc5_48.2: %i32 = converted %int_5.loc5, %.loc5_48.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %SOME_PROTECTED_CONSTANT: %i32 = bind_name SOME_PROTECTED_CONSTANT, %.loc5_48.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %SOME_PRIVATE_CONSTANT.patt: %pattern_type.7ce = binding_pattern SOME_PRIVATE_CONSTANT [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_5.loc6: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc6_38: type = splice_block %i32.loc6 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc6: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc6_44.1: <bound method> = bound_method %int_5.loc6, %impl.elem0.loc6 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_44.2: <bound method> = bound_method %int_5.loc6, %specific_fn.loc6 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %i32 = call %bound_method.loc6_44.2(%int_5.loc6) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc6_44.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc6_44.2: %i32 = converted %int_5.loc6, %.loc6_44.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %SOME_PRIVATE_CONSTANT: %i32 = bind_name SOME_PRIVATE_CONSTANT, %.loc6_44.2
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .SOME_PROTECTED_CONSTANT [protected] = %SOME_PROTECTED_CONSTANT
// CHECK:STDOUT:   .SOME_PRIVATE_CONSTANT [private] = %SOME_PRIVATE_CONSTANT
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Internal {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %INTERNAL_CONSTANT.patt: %pattern_type.7ce = binding_pattern INTERNAL_CONSTANT [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc10_36: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc10_42.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_42.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc10_42.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc10_42.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc10_42.2: %i32 = converted %int_5, %.loc10_42.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %INTERNAL_CONSTANT: %i32 = bind_name INTERNAL_CONSTANT, %.loc10_42.2
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Internal
// CHECK:STDOUT:   .INTERNAL_CONSTANT [protected] = %INTERNAL_CONSTANT
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %Internal.ref: type = name_ref Internal, file.%Internal.decl [concrete = constants.%Internal]
// CHECK:STDOUT:   %.loc14: %B.elem = field_decl internal, element0 [concrete]
// CHECK:STDOUT:   %B.G.decl: %B.G.type = fn_decl @B.G [concrete = constants.%B.G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.SomeFunc.decl: %B.SomeFunc.type = fn_decl @B.SomeFunc [concrete = constants.%B.SomeFunc] {
// CHECK:STDOUT:     %self.patt: %pattern_type.049 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.049 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %B = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %self: %B = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.internal.6f5 [concrete = constants.%complete_type.d77]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .Internal = <poisoned>
// CHECK:STDOUT:   .internal [private] = %.loc14
// CHECK:STDOUT:   .G = %B.G.decl
// CHECK:STDOUT:   .SomeFunc = %B.SomeFunc.decl
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref.loc24: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %SOME_PRIVATE_CONSTANT.ref: <error> = name_ref SOME_PRIVATE_CONSTANT, <error> [concrete = <error>]
// CHECK:STDOUT:   %A.ref.loc33: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %SOME_PROTECTED_CONSTANT.ref: <error> = name_ref SOME_PROTECTED_CONSTANT, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.SomeFunc(%self.param: %B) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %B = name_ref self, %self
// CHECK:STDOUT:   %internal.ref: %B.elem = name_ref internal, @B.%.loc14 [concrete = @B.%.loc14]
// CHECK:STDOUT:   %.loc44_16.1: ref %Internal = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc44_16.2: %Internal = bind_value %.loc44_16.1
// CHECK:STDOUT:   %INTERNAL_CONSTANT.ref: <error> = name_ref INTERNAL_CONSTANT, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_compound_member_access.carbon
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
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F [concrete]
// CHECK:STDOUT:   %B.F: %B.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.953: type = struct_type {.base: %A} [concrete]
// CHECK:STDOUT:   %complete_type.020: <witness> = complete_type_witness %struct_type.base.953 [concrete]
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
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5: %A.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x [concrete = constants.%complete_type.1ec]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x [private] = %.loc5
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc9: %B.elem = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %B.F.decl: %B.F.type = fn_decl @B.F [concrete = constants.%B.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.049 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.049 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %B = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %self: %B = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.953 [concrete = constants.%complete_type.020]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F = %B.F.decl
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.F(%self.param: %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %B = name_ref self, %self
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %x.ref: <error> = name_ref x, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inherited_compound_member_access.carbon
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
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %pattern_type.049: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F [concrete]
// CHECK:STDOUT:   %B.F: %B.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.953: type = struct_type {.base: %A} [concrete]
// CHECK:STDOUT:   %complete_type.020: <witness> = complete_type_witness %struct_type.base.953 [concrete]
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
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5: %A.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x [concrete = constants.%complete_type.1ec]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x [protected] = %.loc5
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc9: %B.elem = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %B.F.decl: %B.F.type = fn_decl @B.F [concrete = constants.%B.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.049 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.049 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %B = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %self: %B = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.953 [concrete = constants.%complete_type.020]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F = %B.F.decl
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.F(%self.param: %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %B = name_ref self, %self
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %x.ref: %A.elem = name_ref x, @A.%.loc5 [concrete = @A.%.loc5]
// CHECK:STDOUT:   %.loc12_9.1: ref %A = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc12_9.2: ref %A = converted %self.ref, %.loc12_9.1
// CHECK:STDOUT:   %.loc12_9.3: ref %i32 = class_element_access %.loc12_9.2, element0
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
