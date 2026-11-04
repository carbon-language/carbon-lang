// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
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
