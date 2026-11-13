// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
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
  private fn F() {}
}

class Derived {
  extend base: Base;

  fn G() {
    Self.F();
  }
}

// --- unqualified_in_friend_of_base.carbon
library "[[@TEST_NAME]]";

class Derived;

base class Base {
  friend Derived;
  private fn F() {}
}

class Derived {
  extend base: Base;

  fn G() {
    F();
  }
}

// --- friend_of_base_via_other_class.carbon
library "[[@TEST_NAME]]";

class Friend;

base class Base {
  friend Friend;
  private fn F() {}
}

class Derived {
  extend base: Base;
}

class Friend {
  fn G(d: Derived) {
    d.F();
  }
}

// --- fail_todo_nested_in_friend_of_base.carbon
library "[[@TEST_NAME]]";

class Derived;

base class Base {
  friend Derived;
  private fn F() {}
}

class Derived {
  extend base: Base;

  class Inner {
    fn ViaBase(b: Base) {
      b.F();
    }
    fn ViaDerived(d: Derived) {
      // CHECK:STDERR:        ╭─┤ fail_todo_nested_in_friend_of_base.carbon:[[@LINE+11]]:7
      // CHECK:STDERR:        │   private fn F() {}
      // CHECK:STDERR:        ·   ────────┬───────
      // CHECK:STDERR:        ·           ╰──┤ declared here [ClassMemberDeclaration]
      // CHECK:STDERR:        ┆
      // CHECK:STDERR: ->     │       d.F();
      // CHECK:STDERR:        ·       ━┳━
      // CHECK:STDERR:        ·        ┗━━┫ error: cannot access private member `F` of type `Base`
      // CHECK:STDERR:        ·           ┃        [ClassInvalidMemberAccess]
      // CHECK:STDERR:        ╰────
      // CHECK:STDERR:
      d.F();
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

// --- fail_todo_base_member_via_derived.carbon
library "[[@TEST_NAME]]";

class Derived;

base class Base {
  private fn F() {}
  fn G(d: Derived);
}

class Derived {
  extend base: Base;
}

fn Base.G(d: Derived) {
  // CHECK:STDERR:        ╭─┤ fail_todo_base_member_via_derived.carbon:[[@LINE+11]]:3
  // CHECK:STDERR:        │   private fn F() {}
  // CHECK:STDERR:        ·   ────────┬───────
  // CHECK:STDERR:        ·           ╰──┤ declared here [ClassMemberDeclaration]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   d.F();
  // CHECK:STDERR:        ·   ━┳━
  // CHECK:STDERR:        ·    ┗━━┫ error: cannot access private member `F` of type `Base`
  // CHECK:STDERR:        ·       ┃        [ClassInvalidMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  d.F();
}

// --- fail_todo_generic_base_member_via_derived.carbon
library "[[@TEST_NAME]]";

class Derived;

base class Base(T: type) {
  private fn F() {}
  fn G(d: Derived);
}

class Derived {
  extend base: Base(());
}

fn Base(T: type).G(d: Derived) {
  // CHECK:STDERR:        ╭─┤ fail_todo_generic_base_member_via_derived.carbon:[[@LINE+11]]:3
  // CHECK:STDERR:        │   private fn F() {}
  // CHECK:STDERR:        ·   ────────┬───────
  // CHECK:STDERR:        ·           ╰──┤ declared here [ClassMemberDeclaration]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   d.F();
  // CHECK:STDERR:        ·   ━┳━
  // CHECK:STDERR:        ·    ┗━━┫ error: cannot access private member `F` of type `Base(())`
  // CHECK:STDERR:        ·       ┃        [ClassInvalidMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  d.F();
  T;
}

// --- fail_todo_generic_base_member_via_derived_in_nested_class.carbon
library "[[@TEST_NAME]]";

class Derived;

base class Base(T: type) {
  private fn F() {}

  class Nested {
    fn G(d: Derived);
  }
}

class Derived {
  extend base: Base(());
}

fn Base(T: type).Nested.G(d: Derived) {
  // CHECK:STDERR:        ╭─┤ fail_todo_generic_base_member_via_derived_in_nested_class.carbon:[[@LINE+11]]:3
  // CHECK:STDERR:        │   private fn F() {}
  // CHECK:STDERR:        ·   ────────┬───────
  // CHECK:STDERR:        ·           ╰──┤ declared here [ClassMemberDeclaration]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   d.F();
  // CHECK:STDERR:        ·   ━┳━
  // CHECK:STDERR:        ·    ┗━━┫ error: cannot access private member `F` of type `Base(())`
  // CHECK:STDERR:        ·       ┃        [ClassInvalidMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  d.F();
  T;
}

// --- friend_of_indirect_base.carbon
library "[[@TEST_NAME]]";

class Friend;

base class Top {
  friend Friend;
  private fn F() {}
}

base class Middle {
  extend base: Top;
}

class Bottom {
  extend base: Middle;
}

class Friend {
  fn G(b: Bottom) {
    b.F();
  }
}

// --- fail_todo_friend_interface_of_base.carbon
library "[[@TEST_NAME]]";

interface I;

base class Base {
  friend I;
  private fn F() {}
}

class Derived {
  extend base: Base;
}

interface I {
  default fn G(d: Derived) {
    // CHECK:STDERR:        ╭─┤ fail_todo_friend_interface_of_base.carbon:[[@LINE+11]]:5
    // CHECK:STDERR:        │   private fn F() {}
    // CHECK:STDERR:        ·   ────────┬───────
    // CHECK:STDERR:        ·           ╰──┤ declared here [ClassMemberDeclaration]
    // CHECK:STDERR:        ┆
    // CHECK:STDERR: ->     │     d.F();
    // CHECK:STDERR:        ·     ━┳━
    // CHECK:STDERR:        ·      ┗━━┫ error: cannot access private member `F` of type `Base`
    // CHECK:STDERR:        ·         ┃        [ClassInvalidMemberAccess]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    d.F();
  }
}

// --- friend_of_adapted_class.carbon
library "[[@TEST_NAME]]";

class Adapter;

class Adapted {
  friend Adapter;
  private fn F() {}
}

class Adapter {
  extend adapt Adapted;

  fn G() {
    Self.F();
  }
}

// --- protected_via_indirect_base.carbon
library "[[@TEST_NAME]]";

base class Top {
  protected fn F() {}
}

base class Middle {
  extend base: Top;
}

class Bottom {
  extend base: Middle;

  fn G(self) {
    self.F();
  }
}

// --- private_member_shadows_indirect_base_member.carbon
library "[[@TEST_NAME]]";

class Friend;

base class Top {
  fn F() {}
}

base class Base {
  extend base: Top;
  friend Friend;
  private fn F() {}
}

class Derived {
  extend base: Base;
}

class Friend {
  fn G(d: Derived) {
    d.F();
  }
}

fn NonFriend(d: Derived) {
  d.F();
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
    // CHECK:STDERR:        ╭─┤ fail_friend_of_base_ambiguous_with_extended_impl.carbon:[[@LINE+7]]:5
    // CHECK:STDERR: ->     │     Derived.F();
    // CHECK:STDERR:        ·     ━━━━┳━━━━
    // CHECK:STDERR:        ·         ┗━━┫ error: ambiguous use of name `F` found in multiple extended scopes
    // CHECK:STDERR:        ·            ┃        [NameAmbiguousDueToExtend]
    // CHECK:STDERR:        ╰────
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
  protected fn F() {}
}

base class Middle {
  extend base: Top;
  friend Friend;
}

class Bottom {
  extend base: Middle;
}

class Friend {
  fn G(m: Middle, b: Bottom) {
    m.F();
    // CHECK:STDERR:        ╭─┤ fail_todo_friend_of_base_protected_member_of_indirect_base.carbon:[[@LINE+11]]:5
    // CHECK:STDERR:        │   protected fn F() {}
    // CHECK:STDERR:        ·   ─────────┬────────
    // CHECK:STDERR:        ·            ╰──┤ declared here [ClassMemberDeclaration]
    // CHECK:STDERR:        ┆
    // CHECK:STDERR: ->     │     b.F();
    // CHECK:STDERR:        ·     ━┳━
    // CHECK:STDERR:        ·      ┗━━┫ error: cannot access protected member `F` of type `Bottom`
    // CHECK:STDERR:        ·         ┃        [ClassInvalidMemberAccess]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    b.F();
  }
}

// --- fail_not_friend_of_base.carbon
library "[[@TEST_NAME]]";

class Derived;

base class Base {
  friend Derived;
  private fn F();
}

class Derived {
  extend base: Base;
}

fn G(d: Derived) {
  // CHECK:STDERR:        ╭─┤ fail_not_friend_of_base.carbon:[[@LINE+11]]:3
  // CHECK:STDERR:        │   private fn F();
  // CHECK:STDERR:        ·   ───────┬───────
  // CHECK:STDERR:        ·          ╰──┤ declared here [ClassMemberDeclaration]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   d.F();
  // CHECK:STDERR:        ·   ━┳━
  // CHECK:STDERR:        ·    ┗━━┫ error: cannot access private member `F` of type `Base`
  // CHECK:STDERR:        ·       ┃        [ClassInvalidMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  d.F();
}

// --- fail_friend_of_derived_only.carbon
library "[[@TEST_NAME]]";

class Friend;

base class Base {
  private fn F();
}

class Derived {
  extend base: Base;
  friend Friend;
  private fn G();
}

class Friend {
  fn H(d: Derived) {
    d.G();
    // CHECK:STDERR:        ╭─┤ fail_friend_of_derived_only.carbon:[[@LINE+11]]:5
    // CHECK:STDERR:        │   private fn F();
    // CHECK:STDERR:        ·   ───────┬───────
    // CHECK:STDERR:        ·          ╰──┤ declared here [ClassMemberDeclaration]
    // CHECK:STDERR:        ┆
    // CHECK:STDERR: ->     │     d.F();
    // CHECK:STDERR:        ·     ━┳━
    // CHECK:STDERR:        ·      ┗━━┫ error: cannot access private member `F` of type `Base`
    // CHECK:STDERR:        ·         ┃        [ClassInvalidMemberAccess]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    d.F();
  }
}

// --- fail_derived_without_friendship_qualified.carbon
library "[[@TEST_NAME]]";

base class Base {
  private fn F() {}
}

class Derived {
  extend base: Base;

  fn Qualified() {
    // CHECK:STDERR:        ╭─┤ fail_derived_without_friendship_qualified.carbon:[[@LINE+11]]:5
    // CHECK:STDERR:        │   private fn F() {}
    // CHECK:STDERR:        ·   ────────┬───────
    // CHECK:STDERR:        ·           ╰──┤ declared here [ClassMemberDeclaration]
    // CHECK:STDERR:        ┆
    // CHECK:STDERR: ->     │     Self.F();
    // CHECK:STDERR:        ·     ━━━┳━━
    // CHECK:STDERR:        ·        ┗━━┫ error: cannot access private member `F` of type `Base`
    // CHECK:STDERR:        ·           ┃        [ClassInvalidMemberAccess]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    Self.F();
  }
}

// --- fail_derived_without_friendship_unqualified.carbon
library "[[@TEST_NAME]]";

base class Base {
  private fn F() {}
}

class Derived {
  extend base: Base;

  fn Unqualified() {
    // CHECK:STDERR:        ╭─┤ fail_derived_without_friendship_unqualified.carbon:[[@LINE+11]]:5
    // CHECK:STDERR:        │   private fn F() {}
    // CHECK:STDERR:        ·   ────────┬───────
    // CHECK:STDERR:        ·           ╰──┤ declared here [ClassMemberDeclaration]
    // CHECK:STDERR:        ┆
    // CHECK:STDERR: ->     │     F();
    // CHECK:STDERR:        ·     ┳
    // CHECK:STDERR:        ·     ┗━━┫ error: cannot access private member `F` of type `Base`
    // CHECK:STDERR:        ·        ┃        [ClassInvalidMemberAccess]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    F();
  }
}

// --- fail_friend_of_middle_only.carbon
library "[[@TEST_NAME]]";

class Friend;

base class Top {
  private fn F();
}

base class Middle {
  extend base: Top;
  friend Friend;
}

class Bottom {
  extend base: Middle;
}

class Friend {
  fn G(b: Bottom) {
    // CHECK:STDERR:        ╭─┤ fail_friend_of_middle_only.carbon:[[@LINE+11]]:5
    // CHECK:STDERR:        │   private fn F();
    // CHECK:STDERR:        ·   ───────┬───────
    // CHECK:STDERR:        ·          ╰──┤ declared here [ClassMemberDeclaration]
    // CHECK:STDERR:        ┆
    // CHECK:STDERR: ->     │     b.F();
    // CHECK:STDERR:        ·     ━┳━
    // CHECK:STDERR:        ·      ┗━━┫ error: cannot access private member `F` of type `Middle`
    // CHECK:STDERR:        ·         ┃        [ClassInvalidMemberAccess]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    b.F();
  }
}

// --- fail_not_friend_of_adapted_class.carbon
library "[[@TEST_NAME]]";

class Adapted {
  private fn F() {}
}

class Adapter {
  extend adapt Adapted;

  fn G() {
    // CHECK:STDERR:        ╭─┤ fail_not_friend_of_adapted_class.carbon:[[@LINE+11]]:5
    // CHECK:STDERR:        │   private fn F() {}
    // CHECK:STDERR:        ·   ────────┬───────
    // CHECK:STDERR:        ·           ╰──┤ declared here [ClassMemberDeclaration]
    // CHECK:STDERR:        ┆
    // CHECK:STDERR: ->     │     Self.F();
    // CHECK:STDERR:        ·     ━━━┳━━
    // CHECK:STDERR:        ·        ┗━━┫ error: cannot access private member `F` of type `Adapted`
    // CHECK:STDERR:        ·           ┃        [ClassInvalidMemberAccess]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    Self.F();
  }
}
