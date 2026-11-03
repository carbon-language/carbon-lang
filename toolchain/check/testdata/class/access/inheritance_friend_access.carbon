// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/access/inheritance_friend_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/access/inheritance_friend_access.carbon

// --- friend_of_base.carbon

library "[[@TEST_NAME]]";

class Derived;

base class Base {
  friend Derived;
  private var x: () = ();
  private fn F() {}
}

class Derived {
  extend base: Base;

  fn G(self) {
    self.x;
    Self.F();
  }
}

// --- fail_todo_unqualified_in_friend_of_base.carbon

library "[[@TEST_NAME]]";

class Derived;

base class Base {
  friend Derived;
  private fn F() {}
}

class Derived {
  extend base: Base;

  fn G() {
    // TODO: This should be valid, as for `Self.F()`. Unqualified lookup doesn't
    // consider the access context when looking in extended scopes.
    // CHECK:STDERR: fail_todo_unqualified_in_friend_of_base.carbon:[[@LINE+3]]:5: error: cannot access private member `F` of type `Base` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_todo_unqualified_in_friend_of_base.carbon:[[@LINE-10]]:3-18: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    F();
  }
}

// --- friend_of_base_via_other_class.carbon

library "[[@TEST_NAME]]";

class Friend;

base class Base {
  friend Friend;
  private var x: () = ();
}

class Derived {
  extend base: Base;
}

class Friend {
  fn F(d: Derived) {
    d.x;
  }
}

// --- nested_in_friend_of_base.carbon

library "[[@TEST_NAME]]";

class Derived;

base class Base {
  friend Derived;
  private var x: () = ();
}

class Derived {
  extend base: Base;

  class Inner {
    fn ViaBase(b: Base) {
      b.x;
    }
    fn ViaDerived(d: Derived) {
      d.x;
    }
  }
}

// --- friend_of_base_via_template.carbon

library "[[@TEST_NAME]]";

class A(template T: type);

base class Base {
  friend A;
  private fn F() {}
}

class A(template T: type) {
  fn G() {
    T.F();
  }
}

class Derived {
  extend base: Base;
}

fn H() {
  A(Derived).G();
}

// --- base_member_via_derived.carbon

library "[[@TEST_NAME]]";

class Derived;

base class Base {
  private var x: () = ();
  fn F(d: Derived);
}

class Derived {
  extend base: Base;
}

fn Base.F(d: Derived) {
  d.x;
}

// --- generic_base_member_via_derived.carbon

library "[[@TEST_NAME]]";

class Derived;

base class Base(T: type) {
  private var x: () = ();
  fn F(d: Derived) -> type;
}

class Derived {
  extend base: Base(());
}

fn Base(T: type).F(d: Derived) -> type {
  d.x;
  return T;
}

// --- generic_base_member_via_derived_in_nested_class.carbon

library "[[@TEST_NAME]]";

class Derived;

base class Base(T: type) {
  private var x: () = ();

  class Nested {
    fn F(d: Derived) -> type;
  }
}

class Derived {
  extend base: Base(());
}

fn Base(T: type).Nested.F(d: Derived) -> type {
  d.x;
  return T;
}

// --- friend_of_indirect_base.carbon

library "[[@TEST_NAME]]";

class Friend;

base class Top {
  friend Friend;
  private var x: () = ();
}

base class Middle {
  extend base: Top;
}

class Bottom {
  extend base: Middle;
}

class Friend {
  fn F(b: Bottom) {
    b.x;
  }
}

// --- friend_interface_of_base.carbon

library "[[@TEST_NAME]]";

interface I;

base class Base {
  friend I;
  private var x: () = ();
}

class Derived {
  extend base: Base;
}

interface I {
  default fn F(d: Derived) {
    d.x;
  }
}

// --- friend_of_adapted_class.carbon

library "[[@TEST_NAME]]";

class Adapter;

class Adapted {
  friend Adapter;
  private fn G() {}
}

class Adapter {
  extend adapt Adapted;

  fn F() {
    Self.G();
  }
}

// --- protected_via_indirect_base.carbon

library "[[@TEST_NAME]]";

base class Base {
  protected var x: () = ();
}

base class Middle {
  extend base: Base;
}

class Derived {
  extend base: Middle;

  fn F(self) {
    self.x;
  }
}

// --- private_member_shadows_indirect_base_member.carbon

library "[[@TEST_NAME]]";

class Friend;

base class Top {
  var x: () = ();
}

base class Base {
  extend base: Top;
  friend Friend;
  private var x: () = ();
}

class Derived {
  extend base: Base;
}

class Friend {
  fn F(d: Derived) {
    // A friend of `Base` finds `Base.x`.
    //@dump-sem-ir-begin
    d.x;
    //@dump-sem-ir-end
  }
}

fn NonFriend(d: Derived) {
  // Other code skips the inaccessible `Base.x` and finds `Top.x`.
  //@dump-sem-ir-begin
  d.x;
  //@dump-sem-ir-end
}

// --- fail_friend_of_base_ambiguous_with_extended_impl.carbon

library "[[@TEST_NAME]]";

class Friend;

interface I {
  fn F();
}

base class Base {
  friend Friend;
  private fn F() {}
}

class Derived {
  extend base: Base;
  extend impl as I {
    fn F() {}
  }
}

class Friend {
  fn G() {
    // A friend of `Base` can see `Base.F`, so `F` is ambiguous between it and
    // `I.F`.
    // CHECK:STDERR: fail_friend_of_base_ambiguous_with_extended_impl.carbon:[[@LINE+2]]:5-13: error: ambiguous use of name `F` found in multiple extended scopes [NameAmbiguousDueToExtend]
    // CHECK:STDERR:
    Derived.F();
  }
}

fn NonFriend() {
  // Other code can't see `Base.F`, so finds only `I.F`.
  Derived.F();
}

// --- fail_todo_friend_of_base_protected_member_of_indirect_base.carbon

library "[[@TEST_NAME]]";

class Friend;

base class Top {
  protected var x: () = ();
}

base class Base {
  extend base: Top;
  friend Friend;
}

class Derived {
  extend base: Base;
}

class Friend {
  fn F(b: Base, d: Derived) {
    b.x;
    // TODO: This should be valid, as for `b.x`. Only private members of
    // extended scopes consider the access context's access to the scope they
    // were found in.
    // CHECK:STDERR: fail_todo_friend_of_base_protected_member_of_indirect_base.carbon:[[@LINE+3]]:5-7: error: cannot access protected member `x` of type `Derived` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_todo_friend_of_base_protected_member_of_indirect_base.carbon:[[@LINE-19]]:17-21: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    d.x;
  }
}

// --- fail_not_friend_of_base.carbon

library "[[@TEST_NAME]]";

class Derived;

base class Base {
  friend Derived;
  private var x: () = ();
}

class Derived {
  extend base: Base;
}

fn F(d: Derived) {
  // CHECK:STDERR: fail_not_friend_of_base.carbon:[[@LINE+3]]:3-5: error: cannot access private member `x` of type `Base` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_not_friend_of_base.carbon:[[@LINE-9]]:15-19: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  d.x;
}

// --- fail_friend_of_derived_only.carbon

library "[[@TEST_NAME]]";

class Friend;

base class Base {
  private var x: () = ();
}

class Derived {
  extend base: Base;
  friend Friend;
  private var y: () = ();
}

class Friend {
  fn F(d: Derived) {
    // Friendship with `Derived` grants access to its private members, but not
    // to the private members of its base class.
    d.y;
    // CHECK:STDERR: fail_friend_of_derived_only.carbon:[[@LINE+3]]:5-7: error: cannot access private member `x` of type `Base` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_friend_of_derived_only.carbon:[[@LINE-15]]:15-19: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    d.x;
  }
}

// --- fail_derived_without_friendship.carbon

library "[[@TEST_NAME]]";

base class Base {
  private var x: () = ();
  private fn F() {}
}

class Derived {
  extend base: Base;

  fn Qualified(self) {
    // CHECK:STDERR: fail_derived_without_friendship.carbon:[[@LINE+3]]:5-10: error: cannot access private member `x` of type `Base` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_derived_without_friendship.carbon:[[@LINE-9]]:15-19: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    self.x;
  }

  fn Unqualified() {
    // CHECK:STDERR: fail_derived_without_friendship.carbon:[[@LINE+3]]:5: error: cannot access private member `F` of type `Base` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_derived_without_friendship.carbon:[[@LINE-15]]:3-18: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    F();
  }
}

// --- fail_friend_of_middle_only.carbon

library "[[@TEST_NAME]]";

class MiddleFriend;

base class Top {
  private var x: () = ();
}

base class Middle {
  extend base: Top;
  friend MiddleFriend;
}

class Bottom {
  extend base: Middle;
}

class MiddleFriend {
  fn F(b: Bottom) {
    // TODO: The diagnostic should name `Top`, where `x` is declared, rather
    // than `Middle`.
    // CHECK:STDERR: fail_friend_of_middle_only.carbon:[[@LINE+3]]:5-7: error: cannot access private member `x` of type `Middle` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_friend_of_middle_only.carbon:[[@LINE-17]]:15-19: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    b.x;
  }
}

// --- fail_not_friend_of_adapted_class.carbon

library "[[@TEST_NAME]]";

class Adapted {
  private fn G() {}
}

class Adapter {
  extend adapt Adapted;

  fn F() {
    // CHECK:STDERR: fail_not_friend_of_adapted_class.carbon:[[@LINE+3]]:5-10: error: cannot access private member `G` of type `Adapted` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_not_friend_of_adapted_class.carbon:[[@LINE-8]]:3-18: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.G();
  }
}

// --- friend_does_not_see_hidden_override.carbon

library "[[@TEST_NAME]]";

class Friend;

base class Base {
  virtual fn F(unused self) {}
}

class Derived {
  extend base: Base;
  friend Friend;
  override fn F(unused self) {}
}

class Friend {
  fn G(d: Derived) {
    // The `override fn` is hidden, even from friends, so this calls `Base.F`.
    //@dump-sem-ir-begin
    d.F();
    //@dump-sem-ir-end
  }
}

// CHECK:STDOUT: --- private_member_shadows_indirect_base_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Friend: type = class_type @Friend [concrete]
// CHECK:STDOUT:   %Top: type = class_type @Top [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Top.elem: type = unbound_element_type %Top, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %Base.elem.731: type = unbound_element_type %Base, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Friend {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Friend
// CHECK:STDOUT:   .Derived = <poisoned>
// CHECK:STDOUT:   .F = %Friend.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Friend.F(%d.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %x.ref: %Base.elem.731 = name_ref x, @Base.%field_decl [concrete = @Base.%field_decl]
// CHECK:STDOUT:   %.loc24_6.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc24_6.2: ref %Base = converted %d.ref, %.loc24_6.1
// CHECK:STDOUT:   %.loc24_6.3: ref %empty_tuple.type = class_element_access %.loc24_6.2, element1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NonFriend(%d.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %x.ref: %Top.elem = name_ref x, @Top.%field_decl [concrete = @Top.%field_decl]
// CHECK:STDOUT:   %.loc32_4.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc32_4.2: ref %Top = class_element_access %.loc32_4.1, element0
// CHECK:STDOUT:   %.loc32_4.3: ref %Top = converted %d.ref, %.loc32_4.2
// CHECK:STDOUT:   %.loc32_4.4: ref %empty_tuple.type = class_element_access %.loc32_4.3, element0
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- friend_does_not_see_hidden_override.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Friend: type = class_type @Friend [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Base.F: %Base.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Friend {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Friend
// CHECK:STDOUT:   .Derived = <poisoned>
// CHECK:STDOUT:   .G = %Friend.G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%Derived.F.decl.loc13_30.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Friend.G(%d.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %F.ref: %Base.F.type = name_ref F, @Base.%Base.F.decl [concrete = constants.%Base.F]
// CHECK:STDOUT:   %Base.F.bound: <bound method> = bound_method %d.ref, %F.ref
// CHECK:STDOUT:   %.loc20_5.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc20_5.2: ref %Base = converted %d.ref, %.loc20_5.1
// CHECK:STDOUT:   %.loc20_5.3: %Base = acquire_value %.loc20_5.2
// CHECK:STDOUT:   %Base.F.call: init %empty_tuple.type = call %Base.F.bound(%.loc20_5.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
