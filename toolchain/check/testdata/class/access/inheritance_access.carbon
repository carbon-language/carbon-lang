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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/access/inheritance_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/access/inheritance_access.carbon


// --- instance_protected_field_access.carbon

library "[[@TEST_NAME]]";

base class Shape {
  protected var x: i32;
  protected var y: i32;
}

class Circle {
  extend base: Shape;

  fn GetPosition(self) -> (i32, i32) {
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
  fn G(self) -> () { return self.F(); }
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

  fn GetPosition(self) -> i32 {
    // CHECK:STDERR: fail_inherited_private_field_access.carbon:[[@LINE+3]]:12-17: error: cannot access private member `y` of type `Shape` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_inherited_private_field_access.carbon:[[@LINE-8]]:15-20: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    return self.y;
  }
}

// --- noninstance_private_on_self.carbon

library "[[@TEST_NAME]]";

class C {
  private fn F() {}
  fn G() {
    Self.F();
    F();
  }
}

// --- noninstance_protected_on_self.carbon

library "[[@TEST_NAME]]";

class C {
  protected fn F() {}
  fn G() {
    Self.F();
    F();
  }
}

// --- fail_noninstance_private_on_parent_qualified.carbon

library "[[@TEST_NAME]]";

base class B {
  private fn F() {}
}

class C {
  extend base: B;
  // CHECK:STDERR: fail_noninstance_private_on_parent_qualified.carbon:[[@LINE+3]]:12-17: error: cannot access private member `F` of type `B` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_noninstance_private_on_parent_qualified.carbon:[[@LINE-6]]:3-18: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  fn G() { Self.F(); }
}

// --- fail_noninstance_private_on_parent_unqualified.carbon

library "[[@TEST_NAME]]";

base class A {
  private fn Fa() {}
}

base class B {
  extend base: A;

  private let SOME_PRIVATE_CONSTANT: i32 = 86;
  private fn Fb() {}
}

class C {
  extend base: B;

  // CHECK:STDERR: fail_noninstance_private_on_parent_unqualified.carbon:[[@LINE+3]]:27-47: error: cannot access private member `SOME_PRIVATE_CONSTANT` of type `B` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_noninstance_private_on_parent_unqualified.carbon:[[@LINE-8]]:15-40: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  fn G1() -> i32 { return SOME_PRIVATE_CONSTANT; }

  // CHECK:STDERR: fail_noninstance_private_on_parent_unqualified.carbon:[[@LINE+3]]:13-14: error: cannot access private member `Fa` of type `A` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_noninstance_private_on_parent_unqualified.carbon:[[@LINE-19]]:3-19: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  fn G2() { Fa(); }

  // CHECK:STDERR: fail_noninstance_private_on_parent_unqualified.carbon:[[@LINE+3]]:13-14: error: cannot access private member `Fb` of type `B` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_noninstance_private_on_parent_unqualified.carbon:[[@LINE-17]]:3-19: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  fn G3() { Fb(); }
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
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE+3]]:5-27: error: cannot access private member `SOME_PRIVATE_CONSTANT` of type `A` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE-12]]:15-40: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    A.SOME_PRIVATE_CONSTANT;

    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE+3]]:12-36: error: cannot access protected member `SOME_PROTECTED_CONSTANT` of type `A` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE-18]]:17-44: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    return A.SOME_PROTECTED_CONSTANT;
  }

  fn SomeFunc(self) -> i32{
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE+3]]:12-42: error: cannot access protected member `INTERNAL_CONSTANT` of type `Internal` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE-20]]:17-38: note: declared here [ClassMemberDeclaration]
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
  fn F(self) {
    // CHECK:STDERR: fail_compound_member_access.carbon:[[@LINE+3]]:11-13: error: cannot access private member `x` of type `A` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_compound_member_access.carbon:[[@LINE-7]]:15-20: note: declared here [ClassMemberDeclaration]
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

  fn F(self) {
    self.(A.x);
  }
}

// --- fail_access_private_parent_in_generic.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct A {
  bool X() const;
};
struct B : private A {};

template<typename T>
auto F(T* _Nonnull t) -> void {
    t->X();
}
''';

class C(T: type) {
  extend base: Cpp.B;
};

fn G() {
    var d: C(i32) = {
        .base = Cpp.B.B(),
    };
    // CHECK:STDERR: fail_access_private_parent_in_generic.carbon:[[@LINE+8]]:13: error: cannot access private member `X` of type `Cpp.B` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_access_private_parent_in_generic.carbon:[[@LINE+7]]:13: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    // CHECK:STDERR: fail_access_private_parent_in_generic.carbon:[[@LINE-15]]:8: error: 'X' is a private member of 'A' [CppInteropParseError]
    // CHECK:STDERR: fail_access_private_parent_in_generic.carbon:[[@LINE+4]]:13: note: in instantiation of function template specialization 'F<Carbon::Main::C<int>>' requested here [CppInteropParseNote]
    // CHECK:STDERR: fail_access_private_parent_in_generic.carbon:[[@LINE-21]]:12-20: note: constrained by private inheritance here [CppInteropParseNote]
    // CHECK:STDERR: fail_access_private_parent_in_generic.carbon:[[@LINE-24]]:8: note: member is declared here [CppInteropParseNote]
    // CHECK:STDERR:
    Cpp.F(&d);
}

// CHECK:STDOUT: --- instance_protected_field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Shape: type = class_type @Shape [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Shape.elem: type = unbound_element_type %Shape, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.y: type = struct_type {.x: %i32, .y: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.506: <witness> = complete_type_witness %struct_type.x.y [concrete]
// CHECK:STDOUT:   %Circle: type = class_type @Circle [concrete]
// CHECK:STDOUT:   %Circle.elem: type = unbound_element_type %Circle, %Shape [concrete]
// CHECK:STDOUT:   %pattern_type.f70: type = pattern_type %Circle [concrete]
// CHECK:STDOUT:   %self.param_patt.769: %pattern_type.f70 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cc7: %pattern_type.f70 = wrapper_binding_pattern self, %self.param_patt.769 [concrete]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.be9: %tuple.type.693 = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.35f: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %.30d: Core.Form = init_form %tuple.type.35f [concrete]
// CHECK:STDOUT:   %pattern_type.354: type = pattern_type %tuple.type.35f [concrete]
// CHECK:STDOUT:   %return.param_patt.46d: %pattern_type.354 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.d16: %pattern_type.354 = return_slot_pattern %return.param_patt.46d, %tuple.type.35f [concrete]
// CHECK:STDOUT:   %Circle.GetPosition.type: type = fn_type @Circle.GetPosition [concrete]
// CHECK:STDOUT:   %Circle.GetPosition: %Circle.GetPosition.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.175: type = struct_type {.base: %Shape} [concrete]
// CHECK:STDOUT:   %complete_type.1e3: <witness> = complete_type_witness %struct_type.base.175 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %field_decl.loc5: %Shape.elem = field_decl x, element0, %i32.loc5 in [concrete] {
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %field_decl.loc6: %Shape.elem = field_decl y, element1, %i32.loc6 in [concrete] {
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x.y [concrete = constants.%complete_type.506]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Shape
// CHECK:STDOUT:   .x [protected] = %field_decl.loc5
// CHECK:STDOUT:   .y [protected] = %field_decl.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Circle {
// CHECK:STDOUT:   %Shape.ref: type = name_ref Shape, file.%Shape.decl [concrete = constants.%Shape]
// CHECK:STDOUT:   %.loc10: %Circle.elem = base_decl %Shape.ref, element0 [concrete]
// CHECK:STDOUT:   %Circle.GetPosition.decl: %Circle.GetPosition.type = fn_decl @Circle.GetPosition [concrete = constants.%Circle.GetPosition] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f70 = value_param_pattern [concrete = constants.%self.param_patt.769]
// CHECK:STDOUT:     %self.patt: %pattern_type.f70 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cc7]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.354 = out_param_pattern [concrete = constants.%return.param_patt.46d]
// CHECK:STDOUT:     %return.patt: %pattern_type.354 = return_slot_pattern %return.param_patt, %.loc12_36.2 [concrete = constants.%return.patt.d16]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc12_28: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc12_33: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12_36.1: %tuple.type.693 = tuple_literal (%i32.loc12_28, %i32.loc12_33) [concrete = constants.%tuple.be9]
// CHECK:STDOUT:     %.loc12_36.2: type = converted %.loc12_36.1, constants.%tuple.type.35f [concrete = constants.%tuple.type.35f]
// CHECK:STDOUT:     %.loc12_36.3: Core.Form = init_form %.loc12_36.2 [concrete = constants.%.30d]
// CHECK:STDOUT:     %self.param: %Circle = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Circle [concrete = constants.%Circle]
// CHECK:STDOUT:     %self: %Circle = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.35f = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.35f = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.175 [concrete = constants.%complete_type.1e3]
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
// CHECK:STDOUT: fn @Circle.GetPosition(%self.param: %Circle) -> out %return.param: %tuple.type.35f {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref.loc13_13: %Circle = name_ref self, %self
// CHECK:STDOUT:   %x.ref: %Shape.elem = name_ref x, @Shape.%field_decl.loc5 [concrete = @Shape.%field_decl.loc5]
// CHECK:STDOUT:   %.loc13_17.1: ref %Shape = class_element_access %self.ref.loc13_13, element0
// CHECK:STDOUT:   %.loc13_17.2: ref %Shape = converted %self.ref.loc13_13, %.loc13_17.1
// CHECK:STDOUT:   %.loc13_17.3: ref %i32 = class_element_access %.loc13_17.2, element0
// CHECK:STDOUT:   %self.ref.loc13_21: %Circle = name_ref self, %self
// CHECK:STDOUT:   %y.ref: %Shape.elem = name_ref y, @Shape.%field_decl.loc6 [concrete = @Shape.%field_decl.loc6]
// CHECK:STDOUT:   %.loc13_25.1: ref %Shape = class_element_access %self.ref.loc13_21, element0
// CHECK:STDOUT:   %.loc13_25.2: ref %Shape = converted %self.ref.loc13_21, %.loc13_25.1
// CHECK:STDOUT:   %.loc13_25.3: ref %i32 = class_element_access %.loc13_25.2, element1
// CHECK:STDOUT:   %.loc13_27.1: %tuple.type.35f = tuple_literal (%.loc13_17.3, %.loc13_25.3)
// CHECK:STDOUT:   %.loc13_17.4: %i32 = acquire_value %.loc13_17.3
// CHECK:STDOUT:   %impl.elem0.loc13_17: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc13_17.1: <bound method> = bound_method %.loc13_17.4, %impl.elem0.loc13_17
// CHECK:STDOUT:   %specific_fn.loc13_17: <specific function> = specific_function %impl.elem0.loc13_17, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_17.2: <bound method> = bound_method %.loc13_17.4, %specific_fn.loc13_17
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc13_17: init %i32 = call %bound_method.loc13_17.2(%.loc13_17.4)
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return.param, element0
// CHECK:STDOUT:   %.loc13_27.2: init %i32 to %tuple.elem0 = in_place_init %Int.as.Copy.impl.Op.call.loc13_17
// CHECK:STDOUT:   %.loc13_25.4: %i32 = acquire_value %.loc13_25.3
// CHECK:STDOUT:   %impl.elem0.loc13_25: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc13_25.1: <bound method> = bound_method %.loc13_25.4, %impl.elem0.loc13_25
// CHECK:STDOUT:   %specific_fn.loc13_25: <specific function> = specific_function %impl.elem0.loc13_25, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_25.2: <bound method> = bound_method %.loc13_25.4, %specific_fn.loc13_25
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc13_25: init %i32 = call %bound_method.loc13_25.2(%.loc13_25.4)
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return.param, element1
// CHECK:STDOUT:   %.loc13_27.3: init %i32 to %tuple.elem1 = in_place_init %Int.as.Copy.impl.Op.call.loc13_25
// CHECK:STDOUT:   %.loc13_27.4: init %tuple.type.35f to %return.param = tuple_init (%.loc13_27.2, %.loc13_27.3)
// CHECK:STDOUT:   %.loc13_28: init %tuple.type.35f = converted %.loc13_27.1, %.loc13_27.4
// CHECK:STDOUT:   return %.loc13_28 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- shadowing_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.F: %A.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F [concrete]
// CHECK:STDOUT:   %B.F: %B.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.cb7: type = struct_type {.base: %A} [concrete]
// CHECK:STDOUT:   %complete_type.d5e: <witness> = complete_type_witness %struct_type.base.cb7 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %self.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %return.param_patt.900: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.427: %pattern_type.cb1 = return_slot_pattern %return.param_patt.900, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %C.G.type: type = fn_type @C.G [concrete]
// CHECK:STDOUT:   %C.G: %C.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.e05: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.c77: <witness> = complete_type_witness %struct_type.base.e05 [concrete]
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
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc10: Core.Form = init_form %i32 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.cb7 [concrete = constants.%complete_type.d5e]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = value_param_pattern [concrete = constants.%self.param_patt]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete = constants.%return.param_patt.900]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %.loc15_18.2 [concrete = constants.%return.patt.427]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_18.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc15_18.2: type = converted %.loc15_18.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc15_18.3: Core.Form = init_form %.loc15_18.2 [concrete = constants.%.262]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.e05 [concrete = constants.%complete_type.c77]
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
// CHECK:STDOUT: fn @B.F() -> out %return.param: %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.G(%self.param: %C) -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C = name_ref self, %self
// CHECK:STDOUT:   %F.ref: %A.F.type = name_ref F, @A.%A.F.decl [concrete = constants.%A.F]
// CHECK:STDOUT:   %A.F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   return %A.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inherited_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %SOME_CONSTANT.patt: %pattern_type.d94 = value_binding_pattern SOME_CONSTANT [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.708: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
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
// CHECK:STDOUT:   %complete_type.d5e: <witness> = complete_type_witness %struct_type.base [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc5_38.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_38.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc5_38.2(%int_5) [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %.loc5_38.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %.loc5_38.2: %i32 = converted %int_5, %.loc5_38.1 [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %SOME_CONSTANT: %i32 = wrapper_binding SOME_CONSTANT, %.loc5_38.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %SOME_CONSTANT.patt: %pattern_type.d94 = value_binding_pattern SOME_CONSTANT [concrete = constants.%SOME_CONSTANT.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.SomeProtectedFunction.decl: %A.SomeProtectedFunction.type = fn_decl @A.SomeProtectedFunction [concrete = constants.%A.SomeProtectedFunction] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6: Core.Form = init_form %i32 [concrete = constants.%.0b3]
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
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14: Core.Form = init_form %i32 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.H.decl: %B.H.type = fn_decl @B.H [concrete = constants.%B.H] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18: Core.Form = init_form %i32 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base [concrete = constants.%complete_type.d5e]
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
// CHECK:STDOUT: fn @A.SomeProtectedFunction() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc7_13.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_13.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc7_13.2(%int_5) [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %.loc7: init %i32 = converted %int_5, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.708]
// CHECK:STDOUT:   return %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %SOME_CONSTANT.ref: %i32 = name_ref SOME_CONSTANT, @A.%SOME_CONSTANT
// CHECK:STDOUT:   %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc15_13.1: <bound method> = bound_method %SOME_CONSTANT.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_13.2: <bound method> = bound_method %SOME_CONSTANT.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc15_13.2(%SOME_CONSTANT.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.H() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %SomeProtectedFunction.ref: %A.SomeProtectedFunction.type = name_ref SomeProtectedFunction, @A.%A.SomeProtectedFunction.decl [concrete = constants.%A.SomeProtectedFunction]
// CHECK:STDOUT:   %A.SomeProtectedFunction.call: init %i32 = call %SomeProtectedFunction.ref()
// CHECK:STDOUT:   return %A.SomeProtectedFunction.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_inherited_private_field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Shape: type = class_type @Shape [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Shape.elem: type = unbound_element_type %Shape, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.a08: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT:   %Square: type = class_type @Square [concrete]
// CHECK:STDOUT:   %Square.elem: type = unbound_element_type %Square, %Shape [concrete]
// CHECK:STDOUT:   %pattern_type.1f6: type = pattern_type %Square [concrete]
// CHECK:STDOUT:   %self.param_patt: %pattern_type.1f6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt: %pattern_type.1f6 = wrapper_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   %Square.GetPosition.type: type = fn_type @Square.GetPosition [concrete]
// CHECK:STDOUT:   %Square.GetPosition: %Square.GetPosition.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.175: type = struct_type {.base: %Shape} [concrete]
// CHECK:STDOUT:   %complete_type.1e3: <witness> = complete_type_witness %struct_type.base.175 [concrete]
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
// CHECK:STDOUT:   %field_decl: %Shape.elem = field_decl y, element0, %i32 in [concrete] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.y [concrete = constants.%complete_type.a08]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Shape
// CHECK:STDOUT:   .y [private] = %field_decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Square {
// CHECK:STDOUT:   %Shape.ref: type = name_ref Shape, file.%Shape.decl [concrete = constants.%Shape]
// CHECK:STDOUT:   %.loc9: %Square.elem = base_decl %Shape.ref, element0 [concrete]
// CHECK:STDOUT:   %Square.GetPosition.decl: %Square.GetPosition.type = fn_decl @Square.GetPosition [concrete = constants.%Square.GetPosition] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1f6 = value_param_pattern [concrete = constants.%self.param_patt]
// CHECK:STDOUT:     %self.patt: %pattern_type.1f6 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11: Core.Form = init_form %i32 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %self.param: %Square = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Square [concrete = constants.%Square]
// CHECK:STDOUT:     %self: %Square = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.175 [concrete = constants.%complete_type.1e3]
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
// CHECK:STDOUT: fn @Square.GetPosition(%self.param: %Square) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Square = name_ref self, %self
// CHECK:STDOUT:   %y.ref: <error> = name_ref y, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- noninstance_private_on_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %F.ref.loc7: %C.F.type = name_ref F, @C.%C.F.decl [concrete = constants.%C.F]
// CHECK:STDOUT:   %C.F.call.loc7: init %empty_tuple.type = call %F.ref.loc7()
// CHECK:STDOUT:   %F.ref.loc8: %C.F.type = name_ref F, @C.%C.F.decl [concrete = constants.%C.F]
// CHECK:STDOUT:   %C.F.call.loc8: init %empty_tuple.type = call %F.ref.loc8()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- noninstance_protected_on_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %F.ref.loc7: %C.F.type = name_ref F, @C.%C.F.decl [concrete = constants.%C.F]
// CHECK:STDOUT:   %C.F.call.loc7: init %empty_tuple.type = call %F.ref.loc7()
// CHECK:STDOUT:   %F.ref.loc8: %C.F.type = name_ref F, @C.%C.F.decl [concrete = constants.%C.F]
// CHECK:STDOUT:   %C.F.call.loc8: init %empty_tuple.type = call %F.ref.loc8()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_noninstance_private_on_parent_qualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F [concrete]
// CHECK:STDOUT:   %B.F: %B.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B [concrete]
// CHECK:STDOUT:   %C.G.type: type = fn_type @C.G [concrete]
// CHECK:STDOUT:   %C.G: %C.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.e05: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.c77: <witness> = complete_type_witness %struct_type.base.e05 [concrete]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.e05 [concrete = constants.%complete_type.c77]
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
// CHECK:STDOUT: --- fail_noninstance_private_on_parent_unqualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %A.Fa.type: type = fn_type @A.Fa [concrete]
// CHECK:STDOUT:   %A.Fa: %A.Fa.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %SOME_PRIVATE_CONSTANT.patt: %pattern_type.d94 = value_binding_pattern SOME_PRIVATE_CONSTANT [concrete]
// CHECK:STDOUT:   %int_86.bd3: Core.IntLiteral = int_value 86 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_86.bd3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_86.bd3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_86.f9f: %i32 = int_value 86 [concrete]
// CHECK:STDOUT:   %B.Fb.type: type = fn_type @B.Fb [concrete]
// CHECK:STDOUT:   %B.Fb: %B.Fb.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.cb7: type = struct_type {.base: %A} [concrete]
// CHECK:STDOUT:   %complete_type.d5e: <witness> = complete_type_witness %struct_type.base.cb7 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %C.G1.type: type = fn_type @C.G1 [concrete]
// CHECK:STDOUT:   %C.G1: %C.G1.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.G2.type: type = fn_type @C.G2 [concrete]
// CHECK:STDOUT:   %C.G2: %C.G2.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.G3.type: type = fn_type @C.G3 [concrete]
// CHECK:STDOUT:   %C.G3: %C.G3.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.e05: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.c77: <witness> = complete_type_witness %struct_type.base.e05 [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %A.Fa.decl: %A.Fa.type = fn_decl @A.Fa [concrete = constants.%A.Fa] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .Fa [private] = %A.Fa.decl
// CHECK:STDOUT:   .SOME_PRIVATE_CONSTANT = <poisoned>
// CHECK:STDOUT:   .Fb = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc9: %B.elem = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %int_86: Core.IntLiteral = int_value 86 [concrete = constants.%int_86.bd3]
// CHECK:STDOUT:   %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc11_44.1: <bound method> = bound_method %int_86, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_44.2: <bound method> = bound_method %int_86, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_44.2(%int_86) [concrete = constants.%int_86.f9f]
// CHECK:STDOUT:   %.loc11_44.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_86.f9f]
// CHECK:STDOUT:   %.loc11_44.2: %i32 = converted %int_86, %.loc11_44.1 [concrete = constants.%int_86.f9f]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %SOME_PRIVATE_CONSTANT: %i32 = wrapper_binding SOME_PRIVATE_CONSTANT, %.loc11_44.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %SOME_PRIVATE_CONSTANT.patt: %pattern_type.d94 = value_binding_pattern SOME_PRIVATE_CONSTANT [concrete = constants.%SOME_PRIVATE_CONSTANT.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.Fb.decl: %B.Fb.type = fn_decl @B.Fb [concrete = constants.%B.Fb] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.cb7 [concrete = constants.%complete_type.d5e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .SOME_PRIVATE_CONSTANT [private] = %SOME_PRIVATE_CONSTANT
// CHECK:STDOUT:   .Fb [private] = %B.Fb.decl
// CHECK:STDOUT:   .Fa = <poisoned>
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc16: %C.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %C.G1.decl: %C.G1.type = fn_decl @C.G1 [concrete = constants.%C.G1] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc21: Core.Form = init_form %i32 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.G2.decl: %C.G2.type = fn_decl @C.G2 [concrete = constants.%C.G2] {} {}
// CHECK:STDOUT:   %C.G3.decl: %C.G3.type = fn_decl @C.G3 [concrete = constants.%C.G3] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.e05 [concrete = constants.%complete_type.c77]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .G1 = %C.G1.decl
// CHECK:STDOUT:   .G2 = %C.G2.decl
// CHECK:STDOUT:   .G3 = %C.G3.decl
// CHECK:STDOUT:   .SOME_PRIVATE_CONSTANT = <poisoned>
// CHECK:STDOUT:   .Fa = <poisoned>
// CHECK:STDOUT:   .Fb = <poisoned>
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.Fa() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.Fb() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.G1() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %SOME_PRIVATE_CONSTANT.ref: <error> = name_ref SOME_PRIVATE_CONSTANT, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.G2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Fa.ref: <error> = name_ref Fa, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.G3() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Fb.ref: <error> = name_ref Fb, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- noninstance_protected_on_parent.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %struct_type.base.e05: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.c77: <witness> = complete_type_witness %struct_type.base.e05 [concrete]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.e05 [concrete = constants.%complete_type.c77]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %SOME_PROTECTED_CONSTANT.patt: %pattern_type.d94 = value_binding_pattern SOME_PROTECTED_CONSTANT [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.708: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %SOME_PRIVATE_CONSTANT.patt: %pattern_type.d94 = value_binding_pattern SOME_PRIVATE_CONSTANT [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Internal: type = class_type @Internal [concrete]
// CHECK:STDOUT:   %INTERNAL_CONSTANT.patt: %pattern_type.d94 = value_binding_pattern INTERNAL_CONSTANT [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %Internal [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %B.G.type: type = fn_type @B.G [concrete]
// CHECK:STDOUT:   %B.G: %B.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %self.param_patt.f1f: %pattern_type.e39 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4c3: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt.f1f [concrete]
// CHECK:STDOUT:   %B.SomeFunc.type: type = fn_type @B.SomeFunc [concrete]
// CHECK:STDOUT:   %B.SomeFunc: %B.SomeFunc.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.internal.a00: type = struct_type {.internal: %Internal} [concrete]
// CHECK:STDOUT:   %complete_type.aff: <witness> = complete_type_witness %struct_type.internal.a00 [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %int_5.loc5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc5: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc5_48.1: <bound method> = bound_method %int_5.loc5, %impl.elem0.loc5 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_48.2: <bound method> = bound_method %int_5.loc5, %specific_fn.loc5 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5: init %i32 = call %bound_method.loc5_48.2(%int_5.loc5) [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %.loc5_48.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5 [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %.loc5_48.2: %i32 = converted %int_5.loc5, %.loc5_48.1 [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %SOME_PROTECTED_CONSTANT: %i32 = wrapper_binding SOME_PROTECTED_CONSTANT, %.loc5_48.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %SOME_PROTECTED_CONSTANT.patt: %pattern_type.d94 = value_binding_pattern SOME_PROTECTED_CONSTANT [concrete = constants.%SOME_PROTECTED_CONSTANT.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_5.loc6: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0.loc6: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc6_44.1: <bound method> = bound_method %int_5.loc6, %impl.elem0.loc6 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_44.2: <bound method> = bound_method %int_5.loc6, %specific_fn.loc6 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6: init %i32 = call %bound_method.loc6_44.2(%int_5.loc6) [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %.loc6_44.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %.loc6_44.2: %i32 = converted %int_5.loc6, %.loc6_44.1 [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %SOME_PRIVATE_CONSTANT: %i32 = wrapper_binding SOME_PRIVATE_CONSTANT, %.loc6_44.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %SOME_PRIVATE_CONSTANT.patt: %pattern_type.d94 = value_binding_pattern SOME_PRIVATE_CONSTANT [concrete = constants.%SOME_PRIVATE_CONSTANT.patt]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc10_42.1: <bound method> = bound_method %int_5, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_42.2: <bound method> = bound_method %int_5, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc10_42.2(%int_5) [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %.loc10_42.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %.loc10_42.2: %i32 = converted %int_5, %.loc10_42.1 [concrete = constants.%int_5.708]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %INTERNAL_CONSTANT: %i32 = wrapper_binding INTERNAL_CONSTANT, %.loc10_42.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %INTERNAL_CONSTANT.patt: %pattern_type.d94 = value_binding_pattern INTERNAL_CONSTANT [concrete = constants.%INTERNAL_CONSTANT.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Internal
// CHECK:STDOUT:   .INTERNAL_CONSTANT [protected] = %INTERNAL_CONSTANT
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %field_decl: %B.elem = field_decl internal, element0, %Internal.ref in [concrete] {
// CHECK:STDOUT:     %Internal.ref: type = name_ref Internal, file.%Internal.decl [concrete = constants.%Internal]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.G.decl: %B.G.type = fn_decl @B.G [concrete = constants.%B.G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc16: Core.Form = init_form %i32 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.SomeFunc.decl: %B.SomeFunc.type = fn_decl @B.SomeFunc [concrete = constants.%B.SomeFunc] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e39 = value_param_pattern [concrete = constants.%self.param_patt.f1f]
// CHECK:STDOUT:     %self.patt: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4c3]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc28: Core.Form = init_form %i32 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %self.param: %B = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %self: %B = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.internal.a00 [concrete = constants.%complete_type.aff]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .Internal = <poisoned>
// CHECK:STDOUT:   .internal [private] = %field_decl
// CHECK:STDOUT:   .G = %B.G.decl
// CHECK:STDOUT:   .SomeFunc = %B.SomeFunc.decl
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref.loc20: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %SOME_PRIVATE_CONSTANT.ref: <error> = name_ref SOME_PRIVATE_CONSTANT, <error> [concrete = <error>]
// CHECK:STDOUT:   %A.ref.loc25: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %SOME_PROTECTED_CONSTANT.ref: <error> = name_ref SOME_PROTECTED_CONSTANT, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.SomeFunc(%self.param: %B) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %B = name_ref self, %self
// CHECK:STDOUT:   %internal.ref: %B.elem = name_ref internal, @B.%field_decl [concrete = @B.%field_decl]
// CHECK:STDOUT:   %.loc32_16.1: ref %Internal = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc32_16.2: %Internal = acquire_value %.loc32_16.1
// CHECK:STDOUT:   %INTERNAL_CONSTANT.ref: <error> = name_ref INTERNAL_CONSTANT, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_compound_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.c49: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %self.param_patt: %pattern_type.e39 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F [concrete]
// CHECK:STDOUT:   %B.F: %B.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.cb7: type = struct_type {.base: %A} [concrete]
// CHECK:STDOUT:   %complete_type.d5e: <witness> = complete_type_witness %struct_type.base.cb7 [concrete]
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
// CHECK:STDOUT:   %field_decl: %A.elem = field_decl x, element0, %i32 in [concrete] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x [concrete = constants.%complete_type.c49]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x [private] = %field_decl
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc9: %B.elem = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %B.F.decl: %B.F.type = fn_decl @B.F [concrete = constants.%B.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e39 = value_param_pattern [concrete = constants.%self.param_patt]
// CHECK:STDOUT:     %self.patt: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %B = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %self: %B = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.cb7 [concrete = constants.%complete_type.d5e]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.c49: <witness> = complete_type_witness %struct_type.x [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %self.param_patt: %pattern_type.e39 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F [concrete]
// CHECK:STDOUT:   %B.F: %B.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.cb7: type = struct_type {.base: %A} [concrete]
// CHECK:STDOUT:   %complete_type.d5e: <witness> = complete_type_witness %struct_type.base.cb7 [concrete]
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
// CHECK:STDOUT:   %field_decl: %A.elem = field_decl x, element0, %i32 in [concrete] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x [concrete = constants.%complete_type.c49]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .x [protected] = %field_decl
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc9: %B.elem = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %B.F.decl: %B.F.type = fn_decl @B.F [concrete = constants.%B.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e39 = value_param_pattern [concrete = constants.%self.param_patt]
// CHECK:STDOUT:     %self.patt: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %B = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %self: %B = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.cb7 [concrete = constants.%complete_type.d5e]
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
// CHECK:STDOUT:   %x.ref: %A.elem = name_ref x, @A.%field_decl [concrete = @A.%field_decl]
// CHECK:STDOUT:   %.loc12_9.1: ref %A = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc12_9.2: ref %A = converted %self.ref, %.loc12_9.1
// CHECK:STDOUT:   %.loc12_9.3: ref %i32 = class_element_access %.loc12_9.2, element0
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_access_private_parent_in_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.04a: type = class_type @C, @C(%T.67d) [symbolic]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %.c9a: type = custom_layout_type {size=1, align=1, .base@0: %A} [concrete]
// CHECK:STDOUT:   %complete_type.39d: <witness> = complete_type_witness %.c9a [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.elem.bb6: type = unbound_element_type %C.04a, %B [symbolic]
// CHECK:STDOUT:   %struct_type.base.9aa: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.0ba: <witness> = complete_type_witness %struct_type.base.9aa [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.4da: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %C.elem.79b: type = unbound_element_type %C.4da, %B [concrete]
// CHECK:STDOUT:   %pattern_type.b1b: type = pattern_type %C.4da [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.b1b = ref_binding_pattern d [concrete]
// CHECK:STDOUT:   %d.var_patt: %pattern_type.b1b = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   %B.B.cpp_overload_set.type: type = cpp_overload_set_type @B.B.cpp_overload_set [concrete]
// CHECK:STDOUT:   %B.B.cpp_overload_set.value: %B.B.cpp_overload_set.type = cpp_overload_set_value @B.B.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.dc0: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.733: type = pattern_type %ptr.dc0 [concrete]
// CHECK:STDOUT:   %return.param_patt.6fe: %pattern_type.733 = value_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.61d: %pattern_type.733 = wrapper_binding_pattern r#return, %return.param_patt.6fe [concrete]
// CHECK:STDOUT:   %B__carbon_thunk.type: type = fn_type @B__carbon_thunk [concrete]
// CHECK:STDOUT:   %B__carbon_thunk: %B__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.ab8: type = partial_type %B [concrete]
// CHECK:STDOUT:   %F.cpp_overload_set.type: type = cpp_overload_set_type @F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %F.cpp_overload_set.value: %F.cpp_overload_set.type = cpp_overload_set_value @F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.434: type = ptr_type %C.4da [concrete]
// CHECK:STDOUT:   %pattern_type.c84: type = pattern_type %ptr.434 [concrete]
// CHECK:STDOUT:   %t.param_patt.6a0: %pattern_type.c84 = value_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.1a9: %pattern_type.c84 = wrapper_binding_pattern t, %t.param_patt.6a0 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc21_5.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.918: %SubobjectDestroy.type = facet_value %C.4da, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.755: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.918) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C.4da, (%DestroyImpl.impl_witness.755, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet) [concrete]
// CHECK:STDOUT:   %B.X.cpp_overload_set.type: type = cpp_overload_set_type @B.X.cpp_overload_set [concrete]
// CHECK:STDOUT:   %B.X.cpp_overload_set.value: %B.X.cpp_overload_set.type = cpp_overload_set_value @B.X.cpp_overload_set [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .F = %F.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %B.B.cpp_overload_set.value: %B.B.cpp_overload_set.type = cpp_overload_set_value @B.B.cpp_overload_set [concrete = constants.%B.B.cpp_overload_set.value]
// CHECK:STDOUT:   %B__carbon_thunk.decl: %B__carbon_thunk.type = fn_decl @B__carbon_thunk [concrete = constants.%B__carbon_thunk] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.733 = value_param_pattern [concrete = constants.%return.param_patt.6fe]
// CHECK:STDOUT:     %return.patt: %pattern_type.733 = wrapper_binding_pattern r#return, %return.param_patt [concrete = constants.%return.patt.61d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: %ptr.dc0 = value_param call_param0
// CHECK:STDOUT:     %return: %ptr.dc0 = wrapper_binding r#return, %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.cpp_overload_set.value: %F.cpp_overload_set.type = cpp_overload_set_value @F.cpp_overload_set [concrete = constants.%F.cpp_overload_set.value]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %t.param_patt: %pattern_type.c84 = value_param_pattern [concrete = constants.%t.param_patt.6a0]
// CHECK:STDOUT:     %t.patt: %pattern_type.c84 = wrapper_binding_pattern t, %t.param_patt [concrete = constants.%t.patt.1a9]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %t.param: %ptr.434 = value_param call_param0
// CHECK:STDOUT:     %t: %ptr.434 = wrapper_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %B.X.cpp_overload_set.value: %B.X.cpp_overload_set.type = cpp_overload_set_value @B.X.cpp_overload_set [concrete = constants.%B.X.cpp_overload_set.value]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp inline
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc16_10.1: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc16_10.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_12.1: type = splice_block %.loc16_12.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc16_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc16_10.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc16_10.2: type) {
// CHECK:STDOUT:   %T.patt.loc16_10.2: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc16_10.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   %T.loc16_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc16_10.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc16_10.1) [symbolic = %C (constants.%C.04a)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, constants.%B [symbolic = %C.elem (constants.%C.elem.bb6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %B.ref: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %.loc17: @C.%C.elem (%C.elem.bb6) = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.base.9aa [concrete = constants.%complete_type.0ba]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.04a
// CHECK:STDOUT:     .base = %.loc17
// CHECK:STDOUT:     .X = <poisoned>
// CHECK:STDOUT:     extend %B.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %.1: %B.elem = base_decl imports.%A.decl, element0 [concrete]
// CHECK:STDOUT:   %.2: type = custom_layout_type {size=1, align=1, .base@0: %A} [concrete = constants.%.c9a]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.2 [concrete = constants.%complete_type.39d]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = imports.%B.B.cpp_overload_set.value
// CHECK:STDOUT:   .X [private] = imports.%B.X.cpp_overload_set.value
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.var: ref %C.4da = var_storage %d.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %B.ref.loc22_20: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %B.ref.loc22_22: %B.B.cpp_overload_set.type = name_ref B, imports.%B.B.cpp_overload_set.value [concrete = constants.%B.B.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc23_5.1: ref %.ab8 = class_element_access %d.var, element0
// CHECK:STDOUT:   %addr.loc22: %ptr.dc0 = addr_of %.loc23_5.1
// CHECK:STDOUT:   %B__carbon_thunk.call: init %empty_tuple.type = call imports.%B__carbon_thunk.decl(%addr.loc22)
// CHECK:STDOUT:   %.loc22: init %B to %.loc23_5.1 = mark_in_place_init %B__carbon_thunk.call
// CHECK:STDOUT:   %.loc23_5.2: %struct_type.base.9aa = struct_literal (%.loc22)
// CHECK:STDOUT:   %.loc23_5.3: init %.ab8 = as_compatible %.loc22
// CHECK:STDOUT:   %.loc23_5.4: init %.ab8 = converted %.loc22, %.loc23_5.3
// CHECK:STDOUT:   %.loc23_5.5: init %B = as_compatible %.loc23_5.4
// CHECK:STDOUT:   %.loc23_5.6: init %C.4da to %d.var = class_init (%.loc23_5.5)
// CHECK:STDOUT:   %.loc21_5.1: init %C.4da = converted %.loc23_5.2, %.loc23_5.6
// CHECK:STDOUT:   assign %d.var, %.loc21_5.1
// CHECK:STDOUT:   %.loc21_17: type = splice_block %C [concrete = constants.%C.4da] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.4da]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %C.4da = wrapper_binding d, %d.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.b1b = ref_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.b1b = var_pattern %d.patt [concrete = constants.%d.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc32: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %F.ref: %F.cpp_overload_set.type = name_ref F, imports.%F.cpp_overload_set.value [concrete = constants.%F.cpp_overload_set.value]
// CHECK:STDOUT:   %d.ref: ref %C.4da = name_ref d, %d
// CHECK:STDOUT:   %addr.loc32: %ptr.434 = addr_of %d.ref
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call imports.%F.decl(%addr.loc32)
// CHECK:STDOUT:   %Destroy.facet.loc21_5.1: %Destroy.type = facet_value constants.%C.4da, (constants.%DestroyImpl.impl_witness.755, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc21_5.2: %Destroy.type = converted constants.%C.4da, %Destroy.facet.loc21_5.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc21_5.2: %Destroy.type = facet_value constants.%C.4da, (constants.%DestroyImpl.impl_witness.755, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc21_5.3: %Destroy.type = converted constants.%C.4da, %Destroy.facet.loc21_5.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%d.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.B() -> out %return.param: %B;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B__carbon_thunk(%return.param: %ptr.dc0);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%t.param: %ptr.434);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc21_5.1(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc21_5.2(%self.param: ref %struct_type.base.9aa) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc21_5.3(%self.param: ref %C.4da) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc16_10.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc16_10.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc16_10.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc16_10.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C => constants.%C.4da
// CHECK:STDOUT:   %C.elem => constants.%C.elem.79b
// CHECK:STDOUT: }
// CHECK:STDOUT:
