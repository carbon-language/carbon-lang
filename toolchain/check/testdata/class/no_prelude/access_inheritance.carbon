// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/access_inheritance.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/access_inheritance.carbon


// --- instance_protected_field_access.carbon

library "[[@TEST_NAME]]";

base class Shape {
  protected var x: ();
  protected var y: ();
}

class Circle {
  extend base: Shape;

  fn GetPosition[self: Self]() -> ((), ()) {
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
  private fn F() -> ();
}

base class C {
  extend base: B;
  fn G[self: Self]() -> () { return self.F(); }
}

// --- inherited_member_access.carbon

library "[[@TEST_NAME]]";

base class A {
  protected let SOME_CONSTANT: () = ();
  protected fn SomeProtectedFunction() -> () {
    return ();
  }
}

class B {
  extend base: A;

  fn G() -> () {
    return A.SOME_CONSTANT;
  }

  fn H() -> () {
    return A.SomeProtectedFunction();
  }
}

// --- fail_inherited_private_field_access.carbon

library "[[@TEST_NAME]]";

base class Shape {
  private var y: ();
}

class Square {
  extend base: Shape;

  fn GetPosition[self: Self]() -> () {
    // CHECK:STDERR: fail_inherited_private_field_access.carbon:[[@LINE+7]]:12: error: cannot access private member `y` of type `Shape` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     return self.y;
    // CHECK:STDERR:            ^~~~~~
    // CHECK:STDERR: fail_inherited_private_field_access.carbon:[[@LINE-10]]:15: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:   private var y: ();
    // CHECK:STDERR:               ^~~~~
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
  protected let SOME_PROTECTED_CONSTANT: () = ();
  private let SOME_PRIVATE_CONSTANT: () = ();
}

base class Internal {
  protected let INTERNAL_CONSTANT: () = ();
}

class B {
  private var internal: Internal;

  fn G() -> () {
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE+7]]:5: error: cannot access private member `SOME_PRIVATE_CONSTANT` of type `A` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     A.SOME_PRIVATE_CONSTANT;
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE-14]]:15: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:   private let SOME_PRIVATE_CONSTANT: () = ();
    // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    A.SOME_PRIVATE_CONSTANT;

    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE+7]]:12: error: cannot access protected member `SOME_PROTECTED_CONSTANT` of type `A` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     return A.SOME_PROTECTED_CONSTANT;
    // CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE-24]]:17: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:   protected let SOME_PROTECTED_CONSTANT: () = ();
    // CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    return A.SOME_PROTECTED_CONSTANT;
  }

  fn SomeFunc[self: Self]() -> () {
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE+7]]:12: error: cannot access protected member `INTERNAL_CONSTANT` of type `Internal` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     return self.internal.INTERNAL_CONSTANT;
    // CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_non_inherited_access.carbon:[[@LINE-30]]:17: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:   protected let INTERNAL_CONSTANT: () = ();
    // CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    return self.internal.INTERNAL_CONSTANT;
  }
}

// --- fail_compound_member_access.carbon

library "[[@TEST_NAME]]";

base class A {
  private var x: ();
}

class B {
  extend base: A;
  fn F[self: Self]() {
    // CHECK:STDERR: fail_compound_member_access.carbon:[[@LINE+7]]:11: error: cannot access private member `x` of type `A` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     self.(A.x);
    // CHECK:STDERR:           ^~~
    // CHECK:STDERR: fail_compound_member_access.carbon:[[@LINE-9]]:15: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:   private var x: ();
    // CHECK:STDERR:               ^~~~~
    // CHECK:STDERR:
    self.(A.x);
  }
}

// --- inherited_compound_member_access.carbon

library "[[@TEST_NAME]]";

base class A {
  protected var x: ();
}

class B {
  extend base: A;

  fn F[self: Self]() {
    self.(A.x);
  }
}
