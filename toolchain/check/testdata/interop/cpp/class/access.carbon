// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/access.carbon

// ============================================================================
// Public non-function member
// ============================================================================

// --- non_function_member_public.h

struct S {
  int instance_data;
  static int static_data;
};

// --- import_non_function_member_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_public.h";

fn F(s: Cpp.S) {
  //@dump-sem-ir-begin
  let unused instance_data: i32 = s.instance_data;
  let unused static_data: i32 = Cpp.S.static_data;
  //@dump-sem-ir-end
}

// --- import_non_function_member_public_extend.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_public.h";

class Derived {
  extend base: Cpp.S;
}

fn F(d: Derived) {
  //@dump-sem-ir-begin
  let unused instance_data: i32 = d.instance_data;
  let unused static_data: i32 = Derived.static_data;
  //@dump-sem-ir-end
}

// ============================================================================
// Protected non-function member
// ============================================================================

// --- non_function_member_protected.h

class C {
 protected:
  int instance_data;
  static int static_data;
};

// --- fail_import_non_function_member_protected.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_protected.h";

fn F(c: Cpp.C) {
  // CHECK:STDERR: fail_import_non_function_member_protected.carbon:[[@LINE+8]]:35: error: cannot access protected member `instance_data` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let unused instance_data: i32 = c.instance_data;
  // CHECK:STDERR:                                   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_protected.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./non_function_member_protected.h:2:7: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: class C {
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let unused instance_data: i32 = c.instance_data;
  // CHECK:STDERR: fail_import_non_function_member_protected.carbon:[[@LINE+7]]:33: error: cannot access protected member `static_data` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let unused static_data: i32 = Cpp.C.static_data;
  // CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_protected.carbon:[[@LINE+4]]:33: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   let unused static_data: i32 = Cpp.C.static_data;
  // CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused static_data: i32 = Cpp.C.static_data;
}

// --- fail_import_non_function_member_protected_extend.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_protected.h";

class Derived {
  extend base: Cpp.C;
}

fn F(d: Derived) {
  // CHECK:STDERR: fail_import_non_function_member_protected_extend.carbon:[[@LINE+8]]:35: error: cannot access protected member `instance_data` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let unused instance_data: i32 = d.instance_data;
  // CHECK:STDERR:                                   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_protected_extend.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./non_function_member_protected.h:2:7: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: class C {
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let unused instance_data: i32 = d.instance_data;
  // CHECK:STDERR: fail_import_non_function_member_protected_extend.carbon:[[@LINE+7]]:33: error: cannot access protected member `static_data` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let unused static_data: i32 = Derived.static_data;
  // CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_protected_extend.carbon:[[@LINE+4]]:33: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   let unused static_data: i32 = Derived.static_data;
  // CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused static_data: i32 = Derived.static_data;
}

// --- import_non_function_member_protected_extend_call_in_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_protected.h";

class Derived {
  extend base: Cpp.C;
  fn F[self: Self]() {
    //@dump-sem-ir-begin
    let unused instance_data: i32 = self.instance_data;
    let unused unqualified_static_data: i32 = static_data;
    let unused derived_static_data: i32 = Derived.static_data;
    let unused base_static_data: i32 = Cpp.C.static_data;
    //@dump-sem-ir-end
  }
}

// ============================================================================
// Private non-function member
// ============================================================================

// --- non_function_member_private.h

class C {
  int instance_data;
  static int static_data;
};

// --- fail_import_non_function_member_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_private.h";

fn F(c: Cpp.C) {
  // CHECK:STDERR: fail_import_non_function_member_private.carbon:[[@LINE+8]]:35: error: cannot access private member `instance_data` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let unused instance_data: i32 = c.instance_data;
  // CHECK:STDERR:                                   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_private.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./non_function_member_private.h:2:7: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: class C {
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let unused instance_data: i32 = c.instance_data;
  // CHECK:STDERR: fail_import_non_function_member_private.carbon:[[@LINE+7]]:33: error: cannot access private member `static_data` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let unused static_data: i32 = Cpp.C.static_data;
  // CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_private.carbon:[[@LINE+4]]:33: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   let unused static_data: i32 = Cpp.C.static_data;
  // CHECK:STDERR:                                 ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused static_data: i32 = Cpp.C.static_data;
}

// --- fail_import_non_function_member_private_extend.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_private.h";

class Derived {
  extend base: Cpp.C;
  fn F[self: Self]() {
    // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE+8]]:37: error: cannot access private member `instance_data` of type `Cpp.C` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     let unused instance_data: i32 = self.instance_data;
    // CHECK:STDERR:                                     ^~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE-8]]:10: in file included here [InCppInclude]
    // CHECK:STDERR: ./non_function_member_private.h:2:7: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR: class C {
    // CHECK:STDERR:       ^
    // CHECK:STDERR:
    let unused instance_data: i32 = self.instance_data;
  }
}

fn F(d: Derived) {
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE+8]]:35: error: cannot access private member `instance_data` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let unused instance_data: i32 = d.instance_data;
  // CHECK:STDERR:                                   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE-21]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./non_function_member_private.h:2:7: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: class C {
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let unused instance_data: i32 = d.instance_data;
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE+7]]:41: error: cannot access private member `static_data` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let unused derived_static_data: i32 = Derived.static_data;
  // CHECK:STDERR:                                         ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE+4]]:41: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   let unused derived_static_data: i32 = Derived.static_data;
  // CHECK:STDERR:                                         ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused derived_static_data: i32 = Derived.static_data;
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE+7]]:38: error: cannot access private member `static_data` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let unused base_static_data: i32 = Cpp.C.static_data;
  // CHECK:STDERR:                                      ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE-4]]:41: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   let unused derived_static_data: i32 = Derived.static_data;
  // CHECK:STDERR:                                         ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused base_static_data: i32 = Cpp.C.static_data;
}

// --- fail_import_non_function_member_private_extend_static_within_member_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_private.h";

class Derived {
  extend base: Cpp.C;
  fn F() {
    // TODO: `C.static_data` is private, so this should fail.
    // CHECK:STDERR: fail_import_non_function_member_private_extend_static_within_member_function.carbon:[[@LINE+7]]:47: error: cannot access private member `static_data` of type `Cpp.C` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     let unused unqualified_static_data: i32 = static_data;
    // CHECK:STDERR:                                               ^~~~~~~~~~~
    // CHECK:STDERR: fail_import_non_function_member_private_extend_static_within_member_function.carbon:[[@LINE+4]]:47: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     let unused unqualified_static_data: i32 = static_data;
    // CHECK:STDERR:                                               ^~~~~~~~~~~
    // CHECK:STDERR:
    let unused unqualified_static_data: i32 = static_data;
  }
}

// ============================================================================
// Public function member
// ============================================================================

// --- function_member_public.h

struct S {
  auto instance_fn() -> void;
  static auto static_fn() -> void;
};

// --- import_function_member_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "function_member_public.h";

fn F(s: Cpp.S*) {
  //@dump-sem-ir-begin
  s->instance_fn();
  Cpp.S.static_fn();
  //@dump-sem-ir-end
}

// ============================================================================
// Protected function member
// ============================================================================

// --- function_member_protected.h

class C {
 protected:
  auto instance_fn() -> void;
  static auto static_fn() -> void;
};

// --- fail_import_function_member_protected.carbon

library "[[@TEST_NAME]]";

import Cpp library "function_member_protected.h";

fn F(c: Cpp.C*) {
  // CHECK:STDERR: fail_import_function_member_protected.carbon:[[@LINE+7]]:3: error: cannot access protected member `instance_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   c->instance_fn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_function_member_protected.carbon:[[@LINE+4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   c->instance_fn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  c->instance_fn();
  // CHECK:STDERR: fail_import_function_member_protected.carbon:[[@LINE+7]]:3: error: cannot access protected member `static_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.C.static_fn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_function_member_protected.carbon:[[@LINE+4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.C.static_fn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.C.static_fn();
}

// --- fail_todo_import_function_member_protected_extend_call_instance_in_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "function_member_protected.h";

class Derived {
  extend base: Cpp.C;
  fn F[self: Self]() {
    //@dump-sem-ir-begin
    // TODO: It should be possible to call protected methods in the base class.
    // CHECK:STDERR: fail_todo_import_function_member_protected_extend_call_instance_in_member.carbon:[[@LINE+4]]:5: error: call argument of type `Derived` is not supported [CppCallArgTypeNotSupported]
    // CHECK:STDERR:     self.instance_fn();
    // CHECK:STDERR:     ^~~~
    // CHECK:STDERR:
    self.instance_fn();
    //@dump-sem-ir-end
  }
}

// --- import_function_member_protected_extend_call_static_in_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "function_member_protected.h";

class Derived {
  extend base: Cpp.C;
  fn F[unused self: Self]() {
    //@dump-sem-ir-begin
    static_fn();
    Derived.static_fn();
    Cpp.C.static_fn();
    //@dump-sem-ir-end
  }
}

// ============================================================================
// Private function member
// ============================================================================

// --- function_member_private.h

class C {
 private:
  auto instance_fn() -> void;
  static auto static_fn() -> void;
};

// --- fail_import_function_member_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "function_member_private.h";

class Derived {
  extend base: Cpp.C;
  fn F[self: Self]() {
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+7]]:5: error: cannot access private member `instance_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     self.instance_fn();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     self.instance_fn();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    self.instance_fn();
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+7]]:5: error: cannot access private member `static_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     static_fn();
    // CHECK:STDERR:     ^~~~~~~~~
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     static_fn();
    // CHECK:STDERR:     ^~~~~~~~~
    // CHECK:STDERR:
    static_fn();
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+7]]:5: error: cannot access private member `static_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Derived.static_fn();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE-4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     static_fn();
    // CHECK:STDERR:     ^~~~~~~~~
    // CHECK:STDERR:
    Derived.static_fn();
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+7]]:5: error: cannot access private member `static_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.C.static_fn();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE-12]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     static_fn();
    // CHECK:STDERR:     ^~~~~~~~~
    // CHECK:STDERR:
    Cpp.C.static_fn();
  }
}

fn F(c: Cpp.C*) {
  // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+7]]:3: error: cannot access private member `instance_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   c->instance_fn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE-32]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     self.instance_fn();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  c->instance_fn();
  // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+7]]:3: error: cannot access private member `static_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.static_fn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE-32]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     static_fn();
  // CHECK:STDERR:     ^~~~~~~~~
  // CHECK:STDERR:
  Derived.static_fn();
  // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+7]]:3: error: cannot access private member `static_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.C.static_fn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE-40]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     static_fn();
  // CHECK:STDERR:     ^~~~~~~~~
  // CHECK:STDERR:
  Cpp.C.static_fn();
}

// ============================================================================
// Overload set
// ============================================================================

// --- overload_set.h

struct PublicCall {};
struct ProtectedCall {};
struct PrivateCall {};

class C {
 public:
  C(PublicCall x);
  static auto Overload(PublicCall x) -> void;
 protected:
  C(ProtectedCall x);
  static auto Overload(ProtectedCall x) -> void;
 private:
  C(PrivateCall x);
  static auto Overload(PrivateCall x) -> void;
};

// --- import_overload_set_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.C.C(Cpp.PublicCall.PublicCall());
  Cpp.C.Overload(Cpp.PublicCall.PublicCall());
  //@dump-sem-ir-end
}

// --- fail_import_overload_set_protected.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

fn F() {
  // CHECK:STDERR: fail_import_overload_set_protected.carbon:[[@LINE+8]]:3: error: cannot access protected member `C` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.C.C(Cpp.ProtectedCall.ProtectedCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_protected.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload_set.h:11:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   C(ProtectedCall x);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  Cpp.C.C(Cpp.ProtectedCall.ProtectedCall());

  // CHECK:STDERR: fail_import_overload_set_protected.carbon:[[@LINE+8]]:3: error: cannot access protected member `Overload` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.C.Overload(Cpp.ProtectedCall.ProtectedCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_protected.carbon:[[@LINE-16]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload_set.h:12:15: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   static auto Overload(ProtectedCall x) -> void;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  Cpp.C.Overload(Cpp.ProtectedCall.ProtectedCall());
}

// --- import_overload_set_protected_base.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

class D {
  extend base: Cpp.C;

  fn MakePublic() -> D {
    //@dump-sem-ir-begin
    return {.base = C(Cpp.PublicCall.PublicCall())};
    //@dump-sem-ir-end
  }

  fn MakeProtected() -> D {
    //@dump-sem-ir-begin
    return {.base = C(Cpp.ProtectedCall.ProtectedCall())};
    //@dump-sem-ir-end
  }

  fn CallPublic() {
    //@dump-sem-ir-begin
    Overload(Cpp.PublicCall.PublicCall());
    //@dump-sem-ir-end
  }

  fn CallProtected() {
    //@dump-sem-ir-begin
    Overload(Cpp.ProtectedCall.ProtectedCall());
    //@dump-sem-ir-end
  }
}

// --- fail_import_overload_set_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

fn F() {
  // CHECK:STDERR: fail_import_overload_set_private.carbon:[[@LINE+8]]:3: error: cannot access private member `C` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.C.C(Cpp.PrivateCall.PrivateCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_private.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload_set.h:14:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   C(PrivateCall x);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  Cpp.C.C(Cpp.PrivateCall.PrivateCall());

  // CHECK:STDERR: fail_import_overload_set_private.carbon:[[@LINE+8]]:3: error: cannot access private member `Overload` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.C.Overload(Cpp.PrivateCall.PrivateCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_private.carbon:[[@LINE-16]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload_set.h:15:15: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   static auto Overload(PrivateCall x) -> void;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  Cpp.C.Overload(Cpp.PrivateCall.PrivateCall());
}

// --- fail_import_overload_set_private_base.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

class D {
  extend base: Cpp.C;

  fn MakePrivate() -> D {
    // CHECK:STDERR: fail_import_overload_set_private_base.carbon:[[@LINE+8]]:21: error: cannot access private member `C` of type `Cpp.C` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     return {.base = C(Cpp.PrivateCall.PrivateCall())};
    // CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_overload_set_private_base.carbon:[[@LINE-9]]:10: in file included here [InCppInclude]
    // CHECK:STDERR: ./overload_set.h:14:3: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:   C(PrivateCall x);
    // CHECK:STDERR:   ^
    // CHECK:STDERR:
    return {.base = C(Cpp.PrivateCall.PrivateCall())};
  }

  fn CallPrivate() {
    // CHECK:STDERR: fail_import_overload_set_private_base.carbon:[[@LINE+8]]:5: error: cannot access private member `Overload` of type `Cpp.C` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Overload(Cpp.PrivateCall.PrivateCall());
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_overload_set_private_base.carbon:[[@LINE-21]]:10: in file included here [InCppInclude]
    // CHECK:STDERR: ./overload_set.h:15:15: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:   static auto Overload(PrivateCall x) -> void;
    // CHECK:STDERR:               ^
    // CHECK:STDERR:
    Overload(Cpp.PrivateCall.PrivateCall());
  }
}

// --- overload_set_base_class.h

#include "overload_set.h"

class Public : public C {};
class Protected : protected C {};
class Private : private C {};

// --- import_overload_set_public_base_class_call_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.Public.Overload(Cpp.PublicCall.PublicCall());
  //@dump-sem-ir-end
}

// --- fail_import_overload_set_public_base_class_call_non_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

fn F() {
  // CHECK:STDERR: fail_import_overload_set_public_base_class_call_non_public.carbon:[[@LINE+9]]:3: error: cannot access protected member `Overload` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Public.Overload(Cpp.ProtectedCall.ProtectedCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_public_base_class_call_non_public.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload_set_base_class.h:2:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload_set.h:12:15: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   static auto Overload(ProtectedCall x) -> void;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  Cpp.Public.Overload(Cpp.ProtectedCall.ProtectedCall());
  // CHECK:STDERR: fail_import_overload_set_public_base_class_call_non_public.carbon:[[@LINE+9]]:3: error: cannot access private member `Overload` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Public.Overload(Cpp.PrivateCall.PrivateCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_public_base_class_call_non_public.carbon:[[@LINE-16]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload_set_base_class.h:2:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./overload_set.h:15:15: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   static auto Overload(PrivateCall x) -> void;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  Cpp.Public.Overload(Cpp.PrivateCall.PrivateCall());
}

// --- import_overload_set_public_base_class_derived_call_non_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

class Derived {
  extend base: Cpp.Public;

  fn F() {
    //@dump-sem-ir-begin
    Overload(Cpp.PublicCall.PublicCall());
    Overload(Cpp.ProtectedCall.ProtectedCall());
    //@dump-sem-ir-end
  }
}

// --- fail_import_overload_set_public_base_class_derived_call_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

class Derived {
  extend base: Cpp.Public;

  fn F() {
    // CHECK:STDERR: fail_import_overload_set_public_base_class_derived_call_private.carbon:[[@LINE+9]]:5: error: cannot access private member `Overload` of type `Cpp.Public` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Overload(Cpp.PrivateCall.PrivateCall());
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_overload_set_public_base_class_derived_call_private.carbon:[[@LINE-9]]:10: in file included here [InCppInclude]
    // CHECK:STDERR: ./overload_set_base_class.h:2:10: in file included here [InCppInclude]
    // CHECK:STDERR: ./overload_set.h:15:15: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:   static auto Overload(PrivateCall x) -> void;
    // CHECK:STDERR:               ^
    // CHECK:STDERR:
    Overload(Cpp.PrivateCall.PrivateCall());
  }
}

// --- fail_import_overload_set_protected_base_class.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

fn F() {
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon:[[@LINE+7]]:3: error: cannot access protected member `Overload` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Protected.Overload(Cpp.PublicCall.PublicCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon:[[@LINE+4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.Protected.Overload(Cpp.PublicCall.PublicCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Protected.Overload(Cpp.PublicCall.PublicCall());
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon:[[@LINE+7]]:3: error: cannot access protected member `Overload` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Protected.Overload(Cpp.ProtectedCall.ProtectedCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon:[[@LINE-4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.Protected.Overload(Cpp.PublicCall.PublicCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Protected.Overload(Cpp.ProtectedCall.ProtectedCall());
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon:[[@LINE+7]]:3: error: cannot access protected member `Overload` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Protected.Overload(Cpp.PrivateCall.PrivateCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon:[[@LINE-12]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.Protected.Overload(Cpp.PublicCall.PublicCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Protected.Overload(Cpp.PrivateCall.PrivateCall());
}

// --- import_overload_set_protected_base_class_derived_call_non_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

class Derived {
  extend base: Cpp.Protected;

  fn F() {
    //@dump-sem-ir-begin
    Overload(Cpp.PublicCall.PublicCall());
    Overload(Cpp.ProtectedCall.ProtectedCall());
    //@dump-sem-ir-end
  }
}

// --- fail_import_overload_set_protected_base_class_derived_call_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

class Derived {
  extend base: Cpp.Protected;

  fn F() {
    // CHECK:STDERR: fail_import_overload_set_protected_base_class_derived_call_private.carbon:[[@LINE+9]]:5: error: cannot access private member `Overload` of type `Cpp.Protected` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Overload(Cpp.PrivateCall.PrivateCall());
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_overload_set_protected_base_class_derived_call_private.carbon:[[@LINE-9]]:10: in file included here [InCppInclude]
    // CHECK:STDERR: ./overload_set_base_class.h:2:10: in file included here [InCppInclude]
    // CHECK:STDERR: ./overload_set.h:15:15: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:   static auto Overload(PrivateCall x) -> void;
    // CHECK:STDERR:               ^
    // CHECK:STDERR:
    Overload(Cpp.PrivateCall.PrivateCall());
  }
}

// --- fail_import_overload_set_private_base_class.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

fn F() {
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon:[[@LINE+7]]:3: error: cannot access private member `Overload` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Private.Overload(Cpp.PublicCall.PublicCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon:[[@LINE+4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.Private.Overload(Cpp.PublicCall.PublicCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Private.Overload(Cpp.PublicCall.PublicCall());
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon:[[@LINE+7]]:3: error: cannot access private member `Overload` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Private.Overload(Cpp.ProtectedCall.ProtectedCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon:[[@LINE-4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.Private.Overload(Cpp.PublicCall.PublicCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Private.Overload(Cpp.ProtectedCall.ProtectedCall());
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon:[[@LINE+7]]:3: error: cannot access private member `Overload` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Private.Overload(Cpp.PrivateCall.PrivateCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon:[[@LINE-12]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.Private.Overload(Cpp.PublicCall.PublicCall());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Private.Overload(Cpp.PrivateCall.PrivateCall());
}

// --- fail_import_overload_set_private_base_class_derived.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

class Derived {
  extend base: Cpp.Private;

  fn F() {
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon:[[@LINE+7]]:5: error: cannot access private member `Overload` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Overload(Cpp.PublicCall.PublicCall());
    // CHECK:STDERR:     ^~~~~~~~
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Overload(Cpp.PublicCall.PublicCall());
    // CHECK:STDERR:     ^~~~~~~~
    // CHECK:STDERR:
    Overload(Cpp.PublicCall.PublicCall());
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon:[[@LINE+7]]:5: error: cannot access private member `Overload` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Overload(Cpp.ProtectedCall.ProtectedCall());
    // CHECK:STDERR:     ^~~~~~~~
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon:[[@LINE-4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Overload(Cpp.PublicCall.PublicCall());
    // CHECK:STDERR:     ^~~~~~~~
    // CHECK:STDERR:
    Overload(Cpp.ProtectedCall.ProtectedCall());
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon:[[@LINE+7]]:5: error: cannot access private member `Overload` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Overload(Cpp.PrivateCall.PrivateCall());
    // CHECK:STDERR:     ^~~~~~~~
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon:[[@LINE-12]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Overload(Cpp.PublicCall.PublicCall());
    // CHECK:STDERR:     ^~~~~~~~
    // CHECK:STDERR:
    Overload(Cpp.PrivateCall.PrivateCall());
  }
}

// ============================================================================
// Base class
// ============================================================================

// --- base_class.h

class Base {
 public:
  static auto PublicStatic() -> void;
  auto PublicInstance() -> void;
 protected:
  static auto ProtectedStatic() -> void;
  auto ProtectedInstance() -> void;
 private:
  static auto PrivateStatic() -> void;
  auto PrivateInstance() -> void;
};

class Public : public Base {};
class Protected : protected Base {};
class Private : private Base {};

class PublicProtected : public Protected {};
class PublicPrivate : public Private {};

// --- base_class_public_access_allowed.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

class Derived {
  extend base: Cpp.Public;

  fn CallStatic() {
    //@dump-sem-ir-begin
    // OK, we can access a public member of our base class.
    Cpp.Public.PublicStatic();
    Self.PublicStatic();

    // OK, we can access a protected member of our base class.
    Cpp.Public.ProtectedStatic();
    Self.ProtectedStatic();
    //@dump-sem-ir-end
  }

  fn CallInstance(var instance: Cpp.Public) {
    //@dump-sem-ir-begin
    // OK, we can access a public member of our base class.
    instance.PublicInstance();
    // OK, we can access a protected member of our base class.
    instance.ProtectedInstance();
    //@dump-sem-ir-end
  }
}

fn Call(var instance: Cpp.Public) {
  //@dump-sem-ir-begin
  Cpp.Public.PublicStatic();
  Derived.PublicStatic();

  instance.PublicInstance();
  instance.PublicInstance();
  //@dump-sem-ir-end
}

// --- fail_base_class_public_access_denied.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

class Derived {
  extend base: Cpp.Public;

  fn CallStatic() {
    // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.Public` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.Public.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.Public.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Cpp.Public.PrivateStatic();
    // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.Public` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE-4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.Public.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Self.PrivateStatic();
  }

  fn CallInstance(var instance: Cpp.Public) {
    //@dump-sem-ir-begin
    // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PrivateInstance` of type `Cpp.Public` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.PrivateInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     instance.PrivateInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    instance.PrivateInstance();
    //@dump-sem-ir-end
  }
}

fn Call(var instance: Cpp.Public) {
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+7]]:3: error: cannot access protected member `ProtectedStatic` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Public.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.Public.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Public.ProtectedStatic();
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+7]]:3: error: cannot access protected member `ProtectedStatic` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE-4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.Public.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Derived.ProtectedStatic();

  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Public.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE-46]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.Public.PrivateStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Public.PrivateStatic();
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE-54]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.Public.PrivateStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Derived.PrivateStatic();

  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+7]]:3: error: cannot access protected member `ProtectedInstance` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.ProtectedInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   instance.ProtectedInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  instance.ProtectedInstance();
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PrivateInstance` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PrivateInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE-51]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     instance.PrivateInstance();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  instance.PrivateInstance();
}

// --- base_class_protected_access_allowed.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

class Derived {
  extend base: Cpp.Protected;

  fn CallStatic() {
    //@dump-sem-ir-begin
    // OK, we can access a public member of our base class.
    Cpp.Protected.PublicStatic();
    Self.PublicStatic();

    // OK, we can access a protected member of our base class.
    Cpp.Protected.ProtectedStatic();
    Self.ProtectedStatic();
    //@dump-sem-ir-end
  }

  fn CallInstance(var instance: Cpp.Protected) {
    //@dump-sem-ir-begin
    // OK, we can access a public member of our base class.
    instance.PublicInstance();
    // OK, we can access a protected member of our base class.
    instance.ProtectedInstance();
    //@dump-sem-ir-end
  }
}

// --- fail_base_class_protected_access_denied.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

class Derived {
  extend base: Cpp.Protected;

  fn CallStatic() {
    // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.Protected.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.Protected.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Cpp.Protected.PrivateStatic();
    // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE-4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.Protected.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Self.PrivateStatic();
  }

  fn CallInstance(var instance: Cpp.Protected) {
    // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PrivateInstance` of type `Cpp.Protected` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.PrivateInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     instance.PrivateInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    instance.PrivateInstance();
  }
}

fn Call(var instance: Cpp.Protected) {
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access protected member `PublicStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Protected.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.Protected.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Protected.PublicStatic();
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access protected member `PublicStatic` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE-4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.Protected.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Derived.PublicStatic();

  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access protected member `ProtectedStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Protected.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.Protected.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Protected.ProtectedStatic();
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access protected member `ProtectedStatic` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE-4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.Protected.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Derived.ProtectedStatic();

  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Protected.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE-61]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.Protected.PrivateStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Protected.PrivateStatic();
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE-69]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.Protected.PrivateStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Derived.PrivateStatic();

  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access protected member `PublicInstance` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PublicInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   instance.PublicInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  instance.PublicInstance();
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access protected member `ProtectedInstance` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.ProtectedInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   instance.ProtectedInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  instance.ProtectedInstance();
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PrivateInstance` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PrivateInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE-75]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     instance.PrivateInstance();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  instance.PrivateInstance();
}

// --- fail_base_class_private_access_denied.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

class Derived {
  extend base: Cpp.Private;

  fn CallStatic() {
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PublicStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.Private.PublicStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.Private.PublicStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Cpp.Private.PublicStatic();
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PublicStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.PublicStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.Private.PublicStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Self.PublicStatic();

    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `ProtectedStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.Private.ProtectedStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.Private.ProtectedStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Cpp.Private.ProtectedStatic();
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `ProtectedStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.ProtectedStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.Private.ProtectedStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Self.ProtectedStatic();

    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.Private.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.Private.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Cpp.Private.PrivateStatic();
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.Private.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Self.PrivateStatic();
  }

  fn CallInstance(var instance: Cpp.Private) {
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PublicInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.PublicInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     instance.PublicInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    instance.PublicInstance();

    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `ProtectedInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.ProtectedInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     instance.ProtectedInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    instance.ProtectedInstance();

    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PrivateInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.PrivateInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     instance.PrivateInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    instance.PrivateInstance();
  }
}

fn Call(var instance: Cpp.Private) {
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PublicStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Private.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-79]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.Private.PublicStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Private.PublicStatic();
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PublicStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-87]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.Private.PublicStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Derived.PublicStatic();

  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `ProtectedStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Private.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-79]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.Private.ProtectedStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Private.ProtectedStatic();
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `ProtectedStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-87]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.Private.ProtectedStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Derived.ProtectedStatic();

  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Private.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-79]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.Private.PrivateStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.Private.PrivateStatic();
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-87]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.Private.PrivateStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Derived.PrivateStatic();

  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PublicInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PublicInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-77]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     instance.PublicInstance();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  instance.PublicInstance();
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `ProtectedInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.ProtectedInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-76]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     instance.ProtectedInstance();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  instance.ProtectedInstance();
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PrivateInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PrivateInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-75]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     instance.PrivateInstance();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  instance.PrivateInstance();
}

// --- base_class_public_protected_access_allowed.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

class Derived {
  extend base: Cpp.PublicProtected;

  fn CallStatic() {
    //@dump-sem-ir-begin
    // OK, we can access a public member of our base class.
    Cpp.PublicProtected.PublicStatic();
    Self.PublicStatic();

    // OK, we can access a protected member of our base class.
    Cpp.PublicProtected.ProtectedStatic();
    Self.ProtectedStatic();
    //@dump-sem-ir-end
  }

  fn CallInstance(var instance: Cpp.PublicProtected) {
    //@dump-sem-ir-begin
    // OK, we can access a public member of our base class.
    instance.PublicInstance();
    // OK, we can access a protected member of our base class.
    instance.ProtectedInstance();
    //@dump-sem-ir-end
  }
}

// --- fail_base_class_public_protected_access_denied.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

class Derived {
  extend base: Cpp.PublicProtected;

  fn CallStatic() {
    // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.PublicProtected.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.PublicProtected.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Cpp.PublicProtected.PrivateStatic();
    // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE-4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.PublicProtected.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Self.PrivateStatic();
  }

  fn CallInstance(var instance: Cpp.PublicProtected) {
    // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PrivateInstance` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.PrivateInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     instance.PrivateInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    instance.PrivateInstance();
  }
}

fn Call(var instance: Cpp.PublicProtected) {
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access protected member `PublicStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.PublicProtected.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.PublicProtected.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.PublicProtected.PublicStatic();
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access protected member `PublicStatic` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE-4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.PublicProtected.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Derived.PublicStatic();

  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access protected member `ProtectedStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.PublicProtected.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.PublicProtected.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.PublicProtected.ProtectedStatic();
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access protected member `ProtectedStatic` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE-4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   Cpp.PublicProtected.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Derived.ProtectedStatic();

  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.PublicProtected.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE-61]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.PublicProtected.PrivateStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.PublicProtected.PrivateStatic();
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE-69]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.PublicProtected.PrivateStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Derived.PrivateStatic();

  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access protected member `PublicInstance` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PublicInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   instance.PublicInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  instance.PublicInstance();
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access protected member `ProtectedInstance` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.ProtectedInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+4]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   instance.ProtectedInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  instance.ProtectedInstance();
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PrivateInstance` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PrivateInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE-75]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     instance.PrivateInstance();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  instance.PrivateInstance();
}

// --- fail_base_class_public_private_access_denied.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

class Derived {
  extend base: Cpp.PublicPrivate;

  fn CallStatic() {
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PublicStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.PublicPrivate.PublicStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.PublicPrivate.PublicStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Cpp.PublicPrivate.PublicStatic();
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PublicStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.PublicStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.PublicPrivate.PublicStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Self.PublicStatic();

    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `ProtectedStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.PublicPrivate.ProtectedStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.PublicPrivate.ProtectedStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Cpp.PublicPrivate.ProtectedStatic();
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `ProtectedStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.ProtectedStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.PublicPrivate.ProtectedStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Self.ProtectedStatic();

    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.PublicPrivate.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.PublicPrivate.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Cpp.PublicPrivate.PrivateStatic();
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     Cpp.PublicPrivate.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    Self.PrivateStatic();
  }

  fn CallInstance(var instance: Cpp.PublicPrivate) {
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PublicInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.PublicInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     instance.PublicInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    instance.PublicInstance();

    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `ProtectedInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.ProtectedInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     instance.ProtectedInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    instance.ProtectedInstance();

    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:5: error: cannot access private member `PrivateInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.PrivateInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+4]]:5: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:     instance.PrivateInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    instance.PrivateInstance();
  }
}

fn Call(var instance: Cpp.PublicPrivate) {
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PublicStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.PublicPrivate.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-79]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.PublicPrivate.PublicStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.PublicPrivate.PublicStatic();
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PublicStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-87]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.PublicPrivate.PublicStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Derived.PublicStatic();

  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `ProtectedStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.PublicPrivate.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-79]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.PublicPrivate.ProtectedStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.PublicPrivate.ProtectedStatic();
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `ProtectedStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-87]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.PublicPrivate.ProtectedStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Derived.ProtectedStatic();

  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.PublicPrivate.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-79]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.PublicPrivate.PrivateStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.PublicPrivate.PrivateStatic();
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-87]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     Cpp.PublicPrivate.PrivateStatic();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Derived.PrivateStatic();

  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PublicInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PublicInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-77]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     instance.PublicInstance();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  instance.PublicInstance();
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `ProtectedInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.ProtectedInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-76]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     instance.ProtectedInstance();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  instance.ProtectedInstance();
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+7]]:3: error: cannot access private member `PrivateInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PrivateInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-75]]:5: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:     instance.PrivateInstance();
  // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  instance.PrivateInstance();
}

// CHECK:STDOUT: --- import_non_function_member_public.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %S.elem: type = unbound_element_type %S, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %static_data.patt: %pattern_type.7ce = ref_binding_pattern static_data [concrete]
// CHECK:STDOUT:   %static_data.var_patt: %pattern_type.7ce = var_pattern %static_data.patt [concrete]
// CHECK:STDOUT:   %static_data.var: ref %i32 = var %static_data.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %S) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %instance_data.patt: %pattern_type.7ce = value_binding_pattern instance_data [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %S = name_ref s, %s
// CHECK:STDOUT:   %instance_data.ref: %S.elem = name_ref instance_data, @S.%.1 [concrete = @S.%.1]
// CHECK:STDOUT:   %.loc26.1: ref %i32 = class_element_access %s.ref, element0
// CHECK:STDOUT:   %.loc26.2: %i32 = acquire_value %.loc26.1
// CHECK:STDOUT:   %i32.loc23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %instance_data: %i32 = value_binding instance_data, %.loc26.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %static_data.patt: %pattern_type.7ce = value_binding_pattern static_data [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc36: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %static_data.ref: ref %i32 = name_ref static_data, imports.%static_data.var [concrete = imports.%static_data.var]
// CHECK:STDOUT:   %i32.loc33: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc38: %i32 = acquire_value %static_data.ref
// CHECK:STDOUT:   %static_data: %i32 = value_binding static_data, %.loc38
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_function_member_public_extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %S.elem: type = unbound_element_type %S, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %static_data.patt: %pattern_type.7ce = ref_binding_pattern static_data [concrete]
// CHECK:STDOUT:   %static_data.var_patt: %pattern_type.7ce = var_pattern %static_data.patt [concrete]
// CHECK:STDOUT:   %static_data.var: ref %i32 = var %static_data.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%d.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %instance_data.patt: %pattern_type.7ce = value_binding_pattern instance_data [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %instance_data.ref: %S.elem = name_ref instance_data, @S.%.1 [concrete = @S.%.1]
// CHECK:STDOUT:   %.loc35.1: ref %S = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc35.2: ref %S = converted %d.ref, %.loc35.1
// CHECK:STDOUT:   %.loc35.3: ref %i32 = class_element_access %.loc35.2, element0
// CHECK:STDOUT:   %i32.loc32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc35.4: %i32 = acquire_value %.loc35.3
// CHECK:STDOUT:   %instance_data: %i32 = value_binding instance_data, %.loc35.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %static_data.patt: %pattern_type.7ce = value_binding_pattern static_data [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.ref.loc44: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %static_data.ref: ref %i32 = name_ref static_data, imports.%static_data.var [concrete = imports.%static_data.var]
// CHECK:STDOUT:   %i32.loc42: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc45: %i32 = acquire_value %static_data.ref
// CHECK:STDOUT:   %static_data: %i32 = value_binding static_data, %.loc45
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_function_member_protected_extend_call_in_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %static_data.patt: %pattern_type.7ce = ref_binding_pattern static_data [concrete]
// CHECK:STDOUT:   %static_data.var_patt: %pattern_type.7ce = var_pattern %static_data.patt [concrete]
// CHECK:STDOUT:   %static_data.var: ref %i32 = var %static_data.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   .instance_data = <poisoned>
// CHECK:STDOUT:   .static_data = <poisoned>
// CHECK:STDOUT:   .Derived = <poisoned>
// CHECK:STDOUT:   extend %C.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.F(%self.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %instance_data.patt: %pattern_type.7ce = value_binding_pattern instance_data [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %self.ref: %Derived = name_ref self, %self
// CHECK:STDOUT:   %instance_data.ref: %C.elem = name_ref instance_data, @C.%.1 [concrete = @C.%.1]
// CHECK:STDOUT:   %.loc36.1: ref %C = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc36.2: ref %C = converted %self.ref, %.loc36.1
// CHECK:STDOUT:   %.loc36.3: ref %i32 = class_element_access %.loc36.2, element0
// CHECK:STDOUT:   %i32.loc33: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc36.4: %i32 = acquire_value %.loc36.3
// CHECK:STDOUT:   %instance_data: %i32 = value_binding instance_data, %.loc36.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %unqualified_static_data.patt: %pattern_type.7ce = value_binding_pattern unqualified_static_data [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %static_data.ref.loc45: ref %i32 = name_ref static_data, imports.%static_data.var [concrete = imports.%static_data.var]
// CHECK:STDOUT:   %i32.loc43: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc45: %i32 = acquire_value %static_data.ref.loc45
// CHECK:STDOUT:   %unqualified_static_data: %i32 = value_binding unqualified_static_data, %.loc45
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %derived_static_data.patt: %pattern_type.7ce = value_binding_pattern derived_static_data [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %static_data.ref.loc54: ref %i32 = name_ref static_data, imports.%static_data.var [concrete = imports.%static_data.var]
// CHECK:STDOUT:   %i32.loc51: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc54: %i32 = acquire_value %static_data.ref.loc54
// CHECK:STDOUT:   %derived_static_data: %i32 = value_binding derived_static_data, %.loc54
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %base_static_data.patt: %pattern_type.7ce = value_binding_pattern base_static_data [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %static_data.ref.loc66: ref %i32 = name_ref static_data, imports.%static_data.var [concrete = imports.%static_data.var]
// CHECK:STDOUT:   %i32.loc61: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc66: %i32 = acquire_value %static_data.ref.loc66
// CHECK:STDOUT:   %base_static_data: %i32 = value_binding base_static_data, %.loc66
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_function_member_public.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %S.instance_fn.cpp_overload_set.type: type = cpp_overload_set_type @S.instance_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %S.instance_fn.cpp_overload_set.value: %S.instance_fn.cpp_overload_set.type = cpp_overload_set_value @S.instance_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %pattern_type.7da: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %S.instance_fn.type: type = fn_type @S.instance_fn [concrete]
// CHECK:STDOUT:   %S.instance_fn: %S.instance_fn.type = struct_value () [concrete]
// CHECK:STDOUT:   %S.static_fn.cpp_overload_set.type: type = cpp_overload_set_type @S.static_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %S.static_fn.cpp_overload_set.value: %S.static_fn.cpp_overload_set.type = cpp_overload_set_value @S.static_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %S.static_fn.type: type = fn_type @S.static_fn [concrete]
// CHECK:STDOUT:   %S.static_fn: %S.static_fn.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %S.instance_fn.cpp_overload_set.value: %S.instance_fn.cpp_overload_set.type = cpp_overload_set_value @S.instance_fn.cpp_overload_set [concrete = constants.%S.instance_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %S.instance_fn.decl: %S.instance_fn.type = fn_decl @S.instance_fn [concrete = constants.%S.instance_fn] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7da = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.7da = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %S = ref_param call_param0
// CHECK:STDOUT:     %self: ref %S = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.static_fn.cpp_overload_set.value: %S.static_fn.cpp_overload_set.type = cpp_overload_set_value @S.static_fn.cpp_overload_set [concrete = constants.%S.static_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %S.static_fn.decl: %S.static_fn.type = fn_decl @S.static_fn [concrete = constants.%S.static_fn] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %ptr.5c7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %ptr.5c7 = name_ref s, %s
// CHECK:STDOUT:   %.loc21: ref %S = deref %s.ref
// CHECK:STDOUT:   %instance_fn.ref: %S.instance_fn.cpp_overload_set.type = name_ref instance_fn, imports.%S.instance_fn.cpp_overload_set.value [concrete = constants.%S.instance_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc21, %instance_fn.ref
// CHECK:STDOUT:   %S.instance_fn.call: init %empty_tuple.type = call imports.%S.instance_fn.decl(%.loc21)
// CHECK:STDOUT:   %Cpp.ref.loc26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc27: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   %static_fn.ref: %S.static_fn.cpp_overload_set.type = name_ref static_fn, imports.%S.static_fn.cpp_overload_set.value [concrete = constants.%S.static_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %S.static_fn.call: init %empty_tuple.type = call imports.%S.static_fn.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_function_member_protected_extend_call_instance_in_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %C.instance_fn.cpp_overload_set.type: type = cpp_overload_set_type @C.instance_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.instance_fn.cpp_overload_set.value: %C.instance_fn.cpp_overload_set.type = cpp_overload_set_value @C.instance_fn.cpp_overload_set [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %C.instance_fn.cpp_overload_set.value: %C.instance_fn.cpp_overload_set.type = cpp_overload_set_value @C.instance_fn.cpp_overload_set [concrete = constants.%C.instance_fn.cpp_overload_set.value]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   .instance_fn = <poisoned>
// CHECK:STDOUT:   extend %C.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.F(%self.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Derived = name_ref self, %self
// CHECK:STDOUT:   %instance_fn.ref: %C.instance_fn.cpp_overload_set.type = name_ref instance_fn, imports.%C.instance_fn.cpp_overload_set.value [concrete = constants.%C.instance_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %self.ref, %instance_fn.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_function_member_protected_extend_call_static_in_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.static_fn.cpp_overload_set.type: type = cpp_overload_set_type @C.static_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.static_fn.cpp_overload_set.value: %C.static_fn.cpp_overload_set.type = cpp_overload_set_value @C.static_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.static_fn.type: type = fn_type @C.static_fn [concrete]
// CHECK:STDOUT:   %C.static_fn: %C.static_fn.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.static_fn.cpp_overload_set.value: %C.static_fn.cpp_overload_set.type = cpp_overload_set_value @C.static_fn.cpp_overload_set [concrete = constants.%C.static_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %C.static_fn.decl: %C.static_fn.type = fn_decl @C.static_fn [concrete = constants.%C.static_fn] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   .static_fn = <poisoned>
// CHECK:STDOUT:   .Derived = <poisoned>
// CHECK:STDOUT:   extend %C.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.F(%self.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %static_fn.ref.loc30: %C.static_fn.cpp_overload_set.type = name_ref static_fn, imports.%C.static_fn.cpp_overload_set.value [concrete = constants.%C.static_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %C.static_fn.call.loc32: init %empty_tuple.type = call imports.%C.static_fn.decl()
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %static_fn.ref.loc35: %C.static_fn.cpp_overload_set.type = name_ref static_fn, imports.%C.static_fn.cpp_overload_set.value [concrete = constants.%C.static_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %C.static_fn.call.loc38: init %empty_tuple.type = call imports.%C.static_fn.decl()
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %static_fn.ref.loc43: %C.static_fn.cpp_overload_set.type = name_ref static_fn, imports.%C.static_fn.cpp_overload_set.value [concrete = constants.%C.static_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %C.static_fn.call.loc46: init %empty_tuple.type = call imports.%C.static_fn.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overload_set_public.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall: type = class_type @PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.type: type = cpp_overload_set_type @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.dfa: type = ptr_type %PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.type: type = fn_type @PublicCall__carbon_thunk [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk: %PublicCall__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Overload.cpp_overload_set.type: type = cpp_overload_set_type @C.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.Overload.cpp_overload_set.value: %C.Overload.cpp_overload_set.type = cpp_overload_set_value @C.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.type: type = fn_type @Overload__carbon_thunk [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk: %Overload__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor.type: type = fn_type @PublicCall.cpp_destructor [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor: %PublicCall.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .PublicCall = %PublicCall.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall.decl: type = class_decl @PublicCall [concrete = constants.%PublicCall] {} {}
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.decl: %PublicCall__carbon_thunk.type = fn_decl @PublicCall__carbon_thunk [concrete = constants.%PublicCall__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.Overload.cpp_overload_set.value: %C.Overload.cpp_overload_set.type = cpp_overload_set_value @C.Overload.cpp_overload_set [concrete = constants.%C.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Overload__carbon_thunk.decl: %Overload__carbon_thunk.type = fn_decl @Overload__carbon_thunk [concrete = constants.%Overload__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc15: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc17: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicCall.ref.loc21: type = name_ref PublicCall, imports.%PublicCall.decl [concrete = constants.%PublicCall]
// CHECK:STDOUT:   %PublicCall.ref.loc23: %PublicCall.PublicCall.cpp_overload_set.type = name_ref PublicCall, imports.%PublicCall.PublicCall.cpp_overload_set.value [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc26.1: ref %PublicCall = temporary_storage
// CHECK:STDOUT:   %addr.loc26: %ptr.dfa = addr_of %.loc26.1
// CHECK:STDOUT:   %PublicCall__carbon_thunk.call.loc26: init %empty_tuple.type = call imports.%PublicCall__carbon_thunk.decl(%addr.loc26)
// CHECK:STDOUT:   %.loc26.2: init %PublicCall to %.loc26.1 = mark_in_place_init %PublicCall__carbon_thunk.call.loc26
// CHECK:STDOUT:   %.loc27.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc26.3: ref %PublicCall = temporary %.loc26.1, %.loc26.2
// CHECK:STDOUT:   %.loc26.4: %PublicCall = acquire_value %.loc26.3
// CHECK:STDOUT:   %.loc26.5: ref %PublicCall = value_as_ref %.loc26.4
// CHECK:STDOUT:   %addr.loc27.1: %ptr.dfa = addr_of %.loc26.5
// CHECK:STDOUT:   %addr.loc27.2: %ptr.d9e = addr_of %.loc27.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc27.1, %addr.loc27.2)
// CHECK:STDOUT:   %.loc27.2: init %C to %.loc27.1 = mark_in_place_init %C__carbon_thunk.call
// CHECK:STDOUT:   %.loc27.3: ref %C = temporary %.loc27.1, %.loc27.2
// CHECK:STDOUT:   %Cpp.ref.loc29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc30: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Overload.ref: %C.Overload.cpp_overload_set.type = name_ref Overload, imports.%C.Overload.cpp_overload_set.value [concrete = constants.%C.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicCall.ref.loc36: type = name_ref PublicCall, imports.%PublicCall.decl [concrete = constants.%PublicCall]
// CHECK:STDOUT:   %PublicCall.ref.loc38: %PublicCall.PublicCall.cpp_overload_set.type = name_ref PublicCall, imports.%PublicCall.PublicCall.cpp_overload_set.value [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc41.1: ref %PublicCall = temporary_storage
// CHECK:STDOUT:   %addr.loc41: %ptr.dfa = addr_of %.loc41.1
// CHECK:STDOUT:   %PublicCall__carbon_thunk.call.loc41: init %empty_tuple.type = call imports.%PublicCall__carbon_thunk.decl(%addr.loc41)
// CHECK:STDOUT:   %.loc41.2: init %PublicCall to %.loc41.1 = mark_in_place_init %PublicCall__carbon_thunk.call.loc41
// CHECK:STDOUT:   %.loc41.3: ref %PublicCall = temporary %.loc41.1, %.loc41.2
// CHECK:STDOUT:   %.loc41.4: %PublicCall = acquire_value %.loc41.3
// CHECK:STDOUT:   %.loc41.5: ref %PublicCall = value_as_ref %.loc41.4
// CHECK:STDOUT:   %addr.loc42: %ptr.dfa = addr_of %.loc41.5
// CHECK:STDOUT:   %Overload__carbon_thunk.call: init %empty_tuple.type = call imports.%Overload__carbon_thunk.decl(%addr.loc42)
// CHECK:STDOUT:   %PublicCall.cpp_destructor.bound.loc41: <bound method> = bound_method %.loc41.3, constants.%PublicCall.cpp_destructor
// CHECK:STDOUT:   %PublicCall.cpp_destructor.call.loc41: init %empty_tuple.type = call %PublicCall.cpp_destructor.bound.loc41(%.loc41.3)
// CHECK:STDOUT:   %C.cpp_destructor.bound: <bound method> = bound_method %.loc27.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call: init %empty_tuple.type = call %C.cpp_destructor.bound(%.loc27.3)
// CHECK:STDOUT:   %PublicCall.cpp_destructor.bound.loc26: <bound method> = bound_method %.loc26.3, constants.%PublicCall.cpp_destructor
// CHECK:STDOUT:   %PublicCall.cpp_destructor.call.loc26: init %empty_tuple.type = call %PublicCall.cpp_destructor.bound.loc26(%.loc26.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overload_set_protected_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %struct_type.base.7c3: type = struct_type {.base: %C} [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall: type = class_type @PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.type: type = cpp_overload_set_type @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.dfa: type = ptr_type %PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.type: type = fn_type @PublicCall__carbon_thunk [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk: %PublicCall__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.1: type = fn_type @C__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.1: %C__carbon_thunk.type.65f120.1 = struct_value () [concrete]
// CHECK:STDOUT:   %.73a: type = partial_type %C [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor.type: type = fn_type @PublicCall.cpp_destructor [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor: %PublicCall.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ProtectedCall: type = class_type @ProtectedCall [concrete]
// CHECK:STDOUT:   %ProtectedCall.ProtectedCall.cpp_overload_set.type: type = cpp_overload_set_type @ProtectedCall.ProtectedCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ProtectedCall.ProtectedCall.cpp_overload_set.value: %ProtectedCall.ProtectedCall.cpp_overload_set.type = cpp_overload_set_value @ProtectedCall.ProtectedCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.8e6: type = ptr_type %ProtectedCall [concrete]
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.type: type = fn_type @ProtectedCall__carbon_thunk [concrete]
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk: %ProtectedCall__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type.65f120.2: type = fn_type @C__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.d98342.2: %C__carbon_thunk.type.65f120.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.type: type = fn_type @ProtectedCall.cpp_destructor [concrete]
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor: %ProtectedCall.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Overload.cpp_overload_set.type: type = cpp_overload_set_type @C.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.Overload.cpp_overload_set.value: %C.Overload.cpp_overload_set.type = cpp_overload_set_value @C.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.type.9a101f.1: type = fn_type @Overload__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.794c5b.1: %Overload__carbon_thunk.type.9a101f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.type.9a101f.2: type = fn_type @Overload__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.794c5b.2: %Overload__carbon_thunk.type.9a101f.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .PublicCall = %PublicCall.decl
// CHECK:STDOUT:     .ProtectedCall = %ProtectedCall.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall.decl: type = class_decl @PublicCall [concrete = constants.%PublicCall] {} {}
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.decl: %PublicCall__carbon_thunk.type = fn_decl @PublicCall__carbon_thunk [concrete = constants.%PublicCall__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.1: %C__carbon_thunk.type.65f120.1 = fn_decl @C__carbon_thunk.1 [concrete = constants.%C__carbon_thunk.d98342.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ProtectedCall.decl: type = class_decl @ProtectedCall [concrete = constants.%ProtectedCall] {} {}
// CHECK:STDOUT:   %ProtectedCall.ProtectedCall.cpp_overload_set.value: %ProtectedCall.ProtectedCall.cpp_overload_set.type = cpp_overload_set_value @ProtectedCall.ProtectedCall.cpp_overload_set [concrete = constants.%ProtectedCall.ProtectedCall.cpp_overload_set.value]
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.decl: %ProtectedCall__carbon_thunk.type = fn_decl @ProtectedCall__carbon_thunk [concrete = constants.%ProtectedCall__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C__carbon_thunk.decl.8acdfe.2: %C__carbon_thunk.type.65f120.2 = fn_decl @C__carbon_thunk.2 [concrete = constants.%C__carbon_thunk.d98342.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.Overload.cpp_overload_set.value: %C.Overload.cpp_overload_set.type = cpp_overload_set_value @C.Overload.cpp_overload_set [concrete = constants.%C.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Overload__carbon_thunk.decl.85994f.1: %Overload__carbon_thunk.type.9a101f.1 = fn_decl @Overload__carbon_thunk.1 [concrete = constants.%Overload__carbon_thunk.794c5b.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Overload__carbon_thunk.decl.85994f.2: %Overload__carbon_thunk.type.9a101f.2 = fn_decl @Overload__carbon_thunk.2 [concrete = constants.%Overload__carbon_thunk.794c5b.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .MakePublic = %D.MakePublic.decl
// CHECK:STDOUT:   .MakeProtected = %D.MakeProtected.decl
// CHECK:STDOUT:   .CallPublic = %D.CallPublic.decl
// CHECK:STDOUT:   .CallProtected = %D.CallProtected.decl
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .Overload = <poisoned>
// CHECK:STDOUT:   extend %C.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.MakePublic() -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicCall.ref.loc34: type = name_ref PublicCall, imports.%PublicCall.decl [concrete = constants.%PublicCall]
// CHECK:STDOUT:   %PublicCall.ref.loc36: %PublicCall.PublicCall.cpp_overload_set.type = name_ref PublicCall, imports.%PublicCall.PublicCall.cpp_overload_set.value [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc39.1: ref %PublicCall = temporary_storage
// CHECK:STDOUT:   %addr.loc39: %ptr.dfa = addr_of %.loc39.1
// CHECK:STDOUT:   %PublicCall__carbon_thunk.call: init %empty_tuple.type = call imports.%PublicCall__carbon_thunk.decl(%addr.loc39)
// CHECK:STDOUT:   %.loc39.2: init %PublicCall to %.loc39.1 = mark_in_place_init %PublicCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc41.1: ref %.73a = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc39.3: ref %PublicCall = temporary %.loc39.1, %.loc39.2
// CHECK:STDOUT:   %.loc39.4: %PublicCall = acquire_value %.loc39.3
// CHECK:STDOUT:   %.loc39.5: ref %PublicCall = value_as_ref %.loc39.4
// CHECK:STDOUT:   %addr.loc40.1: %ptr.dfa = addr_of %.loc39.5
// CHECK:STDOUT:   %addr.loc40.2: %ptr.d9e = addr_of %.loc41.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.1(%addr.loc40.1, %addr.loc40.2)
// CHECK:STDOUT:   %.loc40: init %C to %.loc41.1 = mark_in_place_init %C__carbon_thunk.call
// CHECK:STDOUT:   %.loc41.2: %struct_type.base.7c3 = struct_literal (%.loc40)
// CHECK:STDOUT:   %.loc41.3: init %.73a = as_compatible %.loc40
// CHECK:STDOUT:   %.loc41.4: init %.73a = converted %.loc40, %.loc41.3
// CHECK:STDOUT:   %.loc41.5: init %C = as_compatible %.loc41.4
// CHECK:STDOUT:   %.loc41.6: init %D to %return.param = class_init (%.loc41.5)
// CHECK:STDOUT:   %.loc42: init %D = converted %.loc41.2, %.loc41.6
// CHECK:STDOUT:   %PublicCall.cpp_destructor.bound: <bound method> = bound_method %.loc39.3, constants.%PublicCall.cpp_destructor
// CHECK:STDOUT:   %PublicCall.cpp_destructor.call: init %empty_tuple.type = call %PublicCall.cpp_destructor.bound(%.loc39.3)
// CHECK:STDOUT:   return %.loc42 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.MakeProtected() -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ProtectedCall.ref.loc59: type = name_ref ProtectedCall, imports.%ProtectedCall.decl [concrete = constants.%ProtectedCall]
// CHECK:STDOUT:   %ProtectedCall.ref.loc61: %ProtectedCall.ProtectedCall.cpp_overload_set.type = name_ref ProtectedCall, imports.%ProtectedCall.ProtectedCall.cpp_overload_set.value [concrete = constants.%ProtectedCall.ProtectedCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc64.1: ref %ProtectedCall = temporary_storage
// CHECK:STDOUT:   %addr.loc64: %ptr.8e6 = addr_of %.loc64.1
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.call: init %empty_tuple.type = call imports.%ProtectedCall__carbon_thunk.decl(%addr.loc64)
// CHECK:STDOUT:   %.loc64.2: init %ProtectedCall to %.loc64.1 = mark_in_place_init %ProtectedCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc66.1: ref %.73a = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc64.3: ref %ProtectedCall = temporary %.loc64.1, %.loc64.2
// CHECK:STDOUT:   %.loc64.4: %ProtectedCall = acquire_value %.loc64.3
// CHECK:STDOUT:   %.loc64.5: ref %ProtectedCall = value_as_ref %.loc64.4
// CHECK:STDOUT:   %addr.loc65.1: %ptr.8e6 = addr_of %.loc64.5
// CHECK:STDOUT:   %addr.loc65.2: %ptr.d9e = addr_of %.loc66.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl.8acdfe.2(%addr.loc65.1, %addr.loc65.2)
// CHECK:STDOUT:   %.loc65: init %C to %.loc66.1 = mark_in_place_init %C__carbon_thunk.call
// CHECK:STDOUT:   %.loc66.2: %struct_type.base.7c3 = struct_literal (%.loc65)
// CHECK:STDOUT:   %.loc66.3: init %.73a = as_compatible %.loc65
// CHECK:STDOUT:   %.loc66.4: init %.73a = converted %.loc65, %.loc66.3
// CHECK:STDOUT:   %.loc66.5: init %C = as_compatible %.loc66.4
// CHECK:STDOUT:   %.loc66.6: init %D to %return.param = class_init (%.loc66.5)
// CHECK:STDOUT:   %.loc67: init %D = converted %.loc66.2, %.loc66.6
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.bound: <bound method> = bound_method %.loc64.3, constants.%ProtectedCall.cpp_destructor
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.call: init %empty_tuple.type = call %ProtectedCall.cpp_destructor.bound(%.loc64.3)
// CHECK:STDOUT:   return %.loc67 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.CallPublic() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Overload.ref: %C.Overload.cpp_overload_set.type = name_ref Overload, imports.%C.Overload.cpp_overload_set.value [concrete = constants.%C.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicCall.ref.loc77: type = name_ref PublicCall, imports.%PublicCall.decl [concrete = constants.%PublicCall]
// CHECK:STDOUT:   %PublicCall.ref.loc79: %PublicCall.PublicCall.cpp_overload_set.type = name_ref PublicCall, imports.%PublicCall.PublicCall.cpp_overload_set.value [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc82.1: ref %PublicCall = temporary_storage
// CHECK:STDOUT:   %addr.loc82: %ptr.dfa = addr_of %.loc82.1
// CHECK:STDOUT:   %PublicCall__carbon_thunk.call: init %empty_tuple.type = call imports.%PublicCall__carbon_thunk.decl(%addr.loc82)
// CHECK:STDOUT:   %.loc82.2: init %PublicCall to %.loc82.1 = mark_in_place_init %PublicCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc82.3: ref %PublicCall = temporary %.loc82.1, %.loc82.2
// CHECK:STDOUT:   %.loc82.4: %PublicCall = acquire_value %.loc82.3
// CHECK:STDOUT:   %.loc82.5: ref %PublicCall = value_as_ref %.loc82.4
// CHECK:STDOUT:   %addr.loc83: %ptr.dfa = addr_of %.loc82.5
// CHECK:STDOUT:   %Overload__carbon_thunk.call: init %empty_tuple.type = call imports.%Overload__carbon_thunk.decl.85994f.1(%addr.loc83)
// CHECK:STDOUT:   %PublicCall.cpp_destructor.bound: <bound method> = bound_method %.loc82.3, constants.%PublicCall.cpp_destructor
// CHECK:STDOUT:   %PublicCall.cpp_destructor.call: init %empty_tuple.type = call %PublicCall.cpp_destructor.bound(%.loc82.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.CallProtected() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Overload.ref: %C.Overload.cpp_overload_set.type = name_ref Overload, imports.%C.Overload.cpp_overload_set.value [concrete = constants.%C.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ProtectedCall.ref.loc94: type = name_ref ProtectedCall, imports.%ProtectedCall.decl [concrete = constants.%ProtectedCall]
// CHECK:STDOUT:   %ProtectedCall.ref.loc96: %ProtectedCall.ProtectedCall.cpp_overload_set.type = name_ref ProtectedCall, imports.%ProtectedCall.ProtectedCall.cpp_overload_set.value [concrete = constants.%ProtectedCall.ProtectedCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc99.1: ref %ProtectedCall = temporary_storage
// CHECK:STDOUT:   %addr.loc99: %ptr.8e6 = addr_of %.loc99.1
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.call: init %empty_tuple.type = call imports.%ProtectedCall__carbon_thunk.decl(%addr.loc99)
// CHECK:STDOUT:   %.loc99.2: init %ProtectedCall to %.loc99.1 = mark_in_place_init %ProtectedCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc99.3: ref %ProtectedCall = temporary %.loc99.1, %.loc99.2
// CHECK:STDOUT:   %.loc99.4: %ProtectedCall = acquire_value %.loc99.3
// CHECK:STDOUT:   %.loc99.5: ref %ProtectedCall = value_as_ref %.loc99.4
// CHECK:STDOUT:   %addr.loc100: %ptr.8e6 = addr_of %.loc99.5
// CHECK:STDOUT:   %Overload__carbon_thunk.call: init %empty_tuple.type = call imports.%Overload__carbon_thunk.decl.85994f.2(%addr.loc100)
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.bound: <bound method> = bound_method %.loc99.3, constants.%ProtectedCall.cpp_destructor
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.call: init %empty_tuple.type = call %ProtectedCall.cpp_destructor.bound(%.loc99.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overload_set_public_base_class_call_public.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Public: type = class_type @Public [concrete]
// CHECK:STDOUT:   %Public.Overload.cpp_overload_set.type: type = cpp_overload_set_type @Public.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Public.Overload.cpp_overload_set.value: %Public.Overload.cpp_overload_set.type = cpp_overload_set_value @Public.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall: type = class_type @PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.type: type = cpp_overload_set_type @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.dfa: type = ptr_type %PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.type: type = fn_type @PublicCall__carbon_thunk [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk: %PublicCall__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.type: type = fn_type @Overload__carbon_thunk [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk: %Overload__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor.type: type = fn_type @PublicCall.cpp_destructor [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor: %PublicCall.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Public = %Public.decl
// CHECK:STDOUT:     .PublicCall = %PublicCall.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Public.decl: type = class_decl @Public [concrete = constants.%Public] {} {}
// CHECK:STDOUT:   %Public.Overload.cpp_overload_set.value: %Public.Overload.cpp_overload_set.type = cpp_overload_set_value @Public.Overload.cpp_overload_set [concrete = constants.%Public.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall.decl: type = class_decl @PublicCall [concrete = constants.%PublicCall] {} {}
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.decl: %PublicCall__carbon_thunk.type = fn_decl @PublicCall__carbon_thunk [concrete = constants.%PublicCall__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Overload__carbon_thunk.decl: %Overload__carbon_thunk.type = fn_decl @Overload__carbon_thunk [concrete = constants.%Overload__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Public.ref: type = name_ref Public, imports.%Public.decl [concrete = constants.%Public]
// CHECK:STDOUT:   %Overload.ref: %Public.Overload.cpp_overload_set.type = name_ref Overload, imports.%Public.Overload.cpp_overload_set.value [concrete = constants.%Public.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicCall.ref.loc21: type = name_ref PublicCall, imports.%PublicCall.decl [concrete = constants.%PublicCall]
// CHECK:STDOUT:   %PublicCall.ref.loc23: %PublicCall.PublicCall.cpp_overload_set.type = name_ref PublicCall, imports.%PublicCall.PublicCall.cpp_overload_set.value [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc26.1: ref %PublicCall = temporary_storage
// CHECK:STDOUT:   %addr.loc26: %ptr.dfa = addr_of %.loc26.1
// CHECK:STDOUT:   %PublicCall__carbon_thunk.call: init %empty_tuple.type = call imports.%PublicCall__carbon_thunk.decl(%addr.loc26)
// CHECK:STDOUT:   %.loc26.2: init %PublicCall to %.loc26.1 = mark_in_place_init %PublicCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc26.3: ref %PublicCall = temporary %.loc26.1, %.loc26.2
// CHECK:STDOUT:   %.loc26.4: %PublicCall = acquire_value %.loc26.3
// CHECK:STDOUT:   %.loc26.5: ref %PublicCall = value_as_ref %.loc26.4
// CHECK:STDOUT:   %addr.loc27: %ptr.dfa = addr_of %.loc26.5
// CHECK:STDOUT:   %Overload__carbon_thunk.call: init %empty_tuple.type = call imports.%Overload__carbon_thunk.decl(%addr.loc27)
// CHECK:STDOUT:   %PublicCall.cpp_destructor.bound: <bound method> = bound_method %.loc26.3, constants.%PublicCall.cpp_destructor
// CHECK:STDOUT:   %PublicCall.cpp_destructor.call: init %empty_tuple.type = call %PublicCall.cpp_destructor.bound(%.loc26.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overload_set_public_base_class_derived_call_non_private.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Public: type = class_type @Public [concrete]
// CHECK:STDOUT:   %Public.Overload.cpp_overload_set.type: type = cpp_overload_set_type @Public.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Public.Overload.cpp_overload_set.value: %Public.Overload.cpp_overload_set.type = cpp_overload_set_value @Public.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall: type = class_type @PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.type: type = cpp_overload_set_type @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.dfa: type = ptr_type %PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.type: type = fn_type @PublicCall__carbon_thunk [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk: %PublicCall__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.type.9a101f.1: type = fn_type @Overload__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.794c5b.1: %Overload__carbon_thunk.type.9a101f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ProtectedCall: type = class_type @ProtectedCall [concrete]
// CHECK:STDOUT:   %ProtectedCall.ProtectedCall.cpp_overload_set.type: type = cpp_overload_set_type @ProtectedCall.ProtectedCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ProtectedCall.ProtectedCall.cpp_overload_set.value: %ProtectedCall.ProtectedCall.cpp_overload_set.type = cpp_overload_set_value @ProtectedCall.ProtectedCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.8e6: type = ptr_type %ProtectedCall [concrete]
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.type: type = fn_type @ProtectedCall__carbon_thunk [concrete]
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk: %ProtectedCall__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.type.9a101f.2: type = fn_type @Overload__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.794c5b.2: %Overload__carbon_thunk.type.9a101f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.type: type = fn_type @ProtectedCall.cpp_destructor [concrete]
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor: %ProtectedCall.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor.type: type = fn_type @PublicCall.cpp_destructor [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor: %PublicCall.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Public = %Public.decl
// CHECK:STDOUT:     .PublicCall = %PublicCall.decl
// CHECK:STDOUT:     .ProtectedCall = %ProtectedCall.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Public.decl: type = class_decl @Public [concrete = constants.%Public] {} {}
// CHECK:STDOUT:   %Public.Overload.cpp_overload_set.value: %Public.Overload.cpp_overload_set.type = cpp_overload_set_value @Public.Overload.cpp_overload_set [concrete = constants.%Public.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall.decl: type = class_decl @PublicCall [concrete = constants.%PublicCall] {} {}
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.decl: %PublicCall__carbon_thunk.type = fn_decl @PublicCall__carbon_thunk [concrete = constants.%PublicCall__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Overload__carbon_thunk.decl.85994f.1: %Overload__carbon_thunk.type.9a101f.1 = fn_decl @Overload__carbon_thunk.1 [concrete = constants.%Overload__carbon_thunk.794c5b.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ProtectedCall.decl: type = class_decl @ProtectedCall [concrete = constants.%ProtectedCall] {} {}
// CHECK:STDOUT:   %ProtectedCall.ProtectedCall.cpp_overload_set.value: %ProtectedCall.ProtectedCall.cpp_overload_set.type = cpp_overload_set_value @ProtectedCall.ProtectedCall.cpp_overload_set [concrete = constants.%ProtectedCall.ProtectedCall.cpp_overload_set.value]
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.decl: %ProtectedCall__carbon_thunk.type = fn_decl @ProtectedCall__carbon_thunk [concrete = constants.%ProtectedCall__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Overload__carbon_thunk.decl.85994f.2: %Overload__carbon_thunk.type.9a101f.2 = fn_decl @Overload__carbon_thunk.2 [concrete = constants.%Overload__carbon_thunk.794c5b.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   .Overload = <poisoned>
// CHECK:STDOUT:   extend %Public.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Overload.ref.loc24: %Public.Overload.cpp_overload_set.type = name_ref Overload, imports.%Public.Overload.cpp_overload_set.value [concrete = constants.%Public.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicCall.ref.loc27: type = name_ref PublicCall, imports.%PublicCall.decl [concrete = constants.%PublicCall]
// CHECK:STDOUT:   %PublicCall.ref.loc29: %PublicCall.PublicCall.cpp_overload_set.type = name_ref PublicCall, imports.%PublicCall.PublicCall.cpp_overload_set.value [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc32.1: ref %PublicCall = temporary_storage
// CHECK:STDOUT:   %addr.loc32: %ptr.dfa = addr_of %.loc32.1
// CHECK:STDOUT:   %PublicCall__carbon_thunk.call: init %empty_tuple.type = call imports.%PublicCall__carbon_thunk.decl(%addr.loc32)
// CHECK:STDOUT:   %.loc32.2: init %PublicCall to %.loc32.1 = mark_in_place_init %PublicCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc32.3: ref %PublicCall = temporary %.loc32.1, %.loc32.2
// CHECK:STDOUT:   %.loc32.4: %PublicCall = acquire_value %.loc32.3
// CHECK:STDOUT:   %.loc32.5: ref %PublicCall = value_as_ref %.loc32.4
// CHECK:STDOUT:   %addr.loc33: %ptr.dfa = addr_of %.loc32.5
// CHECK:STDOUT:   %Overload__carbon_thunk.call.loc33: init %empty_tuple.type = call imports.%Overload__carbon_thunk.decl.85994f.1(%addr.loc33)
// CHECK:STDOUT:   %Overload.ref.loc35: %Public.Overload.cpp_overload_set.type = name_ref Overload, imports.%Public.Overload.cpp_overload_set.value [concrete = constants.%Public.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ProtectedCall.ref.loc38: type = name_ref ProtectedCall, imports.%ProtectedCall.decl [concrete = constants.%ProtectedCall]
// CHECK:STDOUT:   %ProtectedCall.ref.loc40: %ProtectedCall.ProtectedCall.cpp_overload_set.type = name_ref ProtectedCall, imports.%ProtectedCall.ProtectedCall.cpp_overload_set.value [concrete = constants.%ProtectedCall.ProtectedCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc43.1: ref %ProtectedCall = temporary_storage
// CHECK:STDOUT:   %addr.loc43: %ptr.8e6 = addr_of %.loc43.1
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.call: init %empty_tuple.type = call imports.%ProtectedCall__carbon_thunk.decl(%addr.loc43)
// CHECK:STDOUT:   %.loc43.2: init %ProtectedCall to %.loc43.1 = mark_in_place_init %ProtectedCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc43.3: ref %ProtectedCall = temporary %.loc43.1, %.loc43.2
// CHECK:STDOUT:   %.loc43.4: %ProtectedCall = acquire_value %.loc43.3
// CHECK:STDOUT:   %.loc43.5: ref %ProtectedCall = value_as_ref %.loc43.4
// CHECK:STDOUT:   %addr.loc44: %ptr.8e6 = addr_of %.loc43.5
// CHECK:STDOUT:   %Overload__carbon_thunk.call.loc44: init %empty_tuple.type = call imports.%Overload__carbon_thunk.decl.85994f.2(%addr.loc44)
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.bound: <bound method> = bound_method %.loc43.3, constants.%ProtectedCall.cpp_destructor
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.call: init %empty_tuple.type = call %ProtectedCall.cpp_destructor.bound(%.loc43.3)
// CHECK:STDOUT:   %PublicCall.cpp_destructor.bound: <bound method> = bound_method %.loc32.3, constants.%PublicCall.cpp_destructor
// CHECK:STDOUT:   %PublicCall.cpp_destructor.call: init %empty_tuple.type = call %PublicCall.cpp_destructor.bound(%.loc32.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overload_set_protected_base_class_derived_call_non_private.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Protected: type = class_type @Protected [concrete]
// CHECK:STDOUT:   %Protected.Overload.cpp_overload_set.type: type = cpp_overload_set_type @Protected.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Protected.Overload.cpp_overload_set.value: %Protected.Overload.cpp_overload_set.type = cpp_overload_set_value @Protected.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall: type = class_type @PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.type: type = cpp_overload_set_type @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.dfa: type = ptr_type %PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.type: type = fn_type @PublicCall__carbon_thunk [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk: %PublicCall__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.type.9a101f.1: type = fn_type @Overload__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.794c5b.1: %Overload__carbon_thunk.type.9a101f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ProtectedCall: type = class_type @ProtectedCall [concrete]
// CHECK:STDOUT:   %ProtectedCall.ProtectedCall.cpp_overload_set.type: type = cpp_overload_set_type @ProtectedCall.ProtectedCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ProtectedCall.ProtectedCall.cpp_overload_set.value: %ProtectedCall.ProtectedCall.cpp_overload_set.type = cpp_overload_set_value @ProtectedCall.ProtectedCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.8e6: type = ptr_type %ProtectedCall [concrete]
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.type: type = fn_type @ProtectedCall__carbon_thunk [concrete]
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk: %ProtectedCall__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.type.9a101f.2: type = fn_type @Overload__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.794c5b.2: %Overload__carbon_thunk.type.9a101f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.type: type = fn_type @ProtectedCall.cpp_destructor [concrete]
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor: %ProtectedCall.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor.type: type = fn_type @PublicCall.cpp_destructor [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor: %PublicCall.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Protected = %Protected.decl
// CHECK:STDOUT:     .PublicCall = %PublicCall.decl
// CHECK:STDOUT:     .ProtectedCall = %ProtectedCall.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Protected.decl: type = class_decl @Protected [concrete = constants.%Protected] {} {}
// CHECK:STDOUT:   %Protected.Overload.cpp_overload_set.value: %Protected.Overload.cpp_overload_set.type = cpp_overload_set_value @Protected.Overload.cpp_overload_set [concrete = constants.%Protected.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall.decl: type = class_decl @PublicCall [concrete = constants.%PublicCall] {} {}
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.decl: %PublicCall__carbon_thunk.type = fn_decl @PublicCall__carbon_thunk [concrete = constants.%PublicCall__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Overload__carbon_thunk.decl.85994f.1: %Overload__carbon_thunk.type.9a101f.1 = fn_decl @Overload__carbon_thunk.1 [concrete = constants.%Overload__carbon_thunk.794c5b.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ProtectedCall.decl: type = class_decl @ProtectedCall [concrete = constants.%ProtectedCall] {} {}
// CHECK:STDOUT:   %ProtectedCall.ProtectedCall.cpp_overload_set.value: %ProtectedCall.ProtectedCall.cpp_overload_set.type = cpp_overload_set_value @ProtectedCall.ProtectedCall.cpp_overload_set [concrete = constants.%ProtectedCall.ProtectedCall.cpp_overload_set.value]
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.decl: %ProtectedCall__carbon_thunk.type = fn_decl @ProtectedCall__carbon_thunk [concrete = constants.%ProtectedCall__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Overload__carbon_thunk.decl.85994f.2: %Overload__carbon_thunk.type.9a101f.2 = fn_decl @Overload__carbon_thunk.2 [concrete = constants.%Overload__carbon_thunk.794c5b.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   .Overload = <poisoned>
// CHECK:STDOUT:   extend %Protected.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Overload.ref.loc24: %Protected.Overload.cpp_overload_set.type = name_ref Overload, imports.%Protected.Overload.cpp_overload_set.value [concrete = constants.%Protected.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicCall.ref.loc27: type = name_ref PublicCall, imports.%PublicCall.decl [concrete = constants.%PublicCall]
// CHECK:STDOUT:   %PublicCall.ref.loc29: %PublicCall.PublicCall.cpp_overload_set.type = name_ref PublicCall, imports.%PublicCall.PublicCall.cpp_overload_set.value [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc32.1: ref %PublicCall = temporary_storage
// CHECK:STDOUT:   %addr.loc32: %ptr.dfa = addr_of %.loc32.1
// CHECK:STDOUT:   %PublicCall__carbon_thunk.call: init %empty_tuple.type = call imports.%PublicCall__carbon_thunk.decl(%addr.loc32)
// CHECK:STDOUT:   %.loc32.2: init %PublicCall to %.loc32.1 = mark_in_place_init %PublicCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc32.3: ref %PublicCall = temporary %.loc32.1, %.loc32.2
// CHECK:STDOUT:   %.loc32.4: %PublicCall = acquire_value %.loc32.3
// CHECK:STDOUT:   %.loc32.5: ref %PublicCall = value_as_ref %.loc32.4
// CHECK:STDOUT:   %addr.loc33: %ptr.dfa = addr_of %.loc32.5
// CHECK:STDOUT:   %Overload__carbon_thunk.call.loc33: init %empty_tuple.type = call imports.%Overload__carbon_thunk.decl.85994f.1(%addr.loc33)
// CHECK:STDOUT:   %Overload.ref.loc35: %Protected.Overload.cpp_overload_set.type = name_ref Overload, imports.%Protected.Overload.cpp_overload_set.value [concrete = constants.%Protected.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ProtectedCall.ref.loc38: type = name_ref ProtectedCall, imports.%ProtectedCall.decl [concrete = constants.%ProtectedCall]
// CHECK:STDOUT:   %ProtectedCall.ref.loc40: %ProtectedCall.ProtectedCall.cpp_overload_set.type = name_ref ProtectedCall, imports.%ProtectedCall.ProtectedCall.cpp_overload_set.value [concrete = constants.%ProtectedCall.ProtectedCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc43.1: ref %ProtectedCall = temporary_storage
// CHECK:STDOUT:   %addr.loc43: %ptr.8e6 = addr_of %.loc43.1
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.call: init %empty_tuple.type = call imports.%ProtectedCall__carbon_thunk.decl(%addr.loc43)
// CHECK:STDOUT:   %.loc43.2: init %ProtectedCall to %.loc43.1 = mark_in_place_init %ProtectedCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc43.3: ref %ProtectedCall = temporary %.loc43.1, %.loc43.2
// CHECK:STDOUT:   %.loc43.4: %ProtectedCall = acquire_value %.loc43.3
// CHECK:STDOUT:   %.loc43.5: ref %ProtectedCall = value_as_ref %.loc43.4
// CHECK:STDOUT:   %addr.loc44: %ptr.8e6 = addr_of %.loc43.5
// CHECK:STDOUT:   %Overload__carbon_thunk.call.loc44: init %empty_tuple.type = call imports.%Overload__carbon_thunk.decl.85994f.2(%addr.loc44)
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.bound: <bound method> = bound_method %.loc43.3, constants.%ProtectedCall.cpp_destructor
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.call: init %empty_tuple.type = call %ProtectedCall.cpp_destructor.bound(%.loc43.3)
// CHECK:STDOUT:   %PublicCall.cpp_destructor.bound: <bound method> = bound_method %.loc32.3, constants.%PublicCall.cpp_destructor
// CHECK:STDOUT:   %PublicCall.cpp_destructor.call: init %empty_tuple.type = call %PublicCall.cpp_destructor.bound(%.loc32.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- base_class_public_access_allowed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Public: type = class_type @Public [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %Public.PublicStatic.cpp_overload_set.type: type = cpp_overload_set_type @Public.PublicStatic.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Public.PublicStatic.cpp_overload_set.value: %Public.PublicStatic.cpp_overload_set.type = cpp_overload_set_value @Public.PublicStatic.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Base.PublicStatic.type: type = fn_type @Base.PublicStatic [concrete]
// CHECK:STDOUT:   %Base.PublicStatic: %Base.PublicStatic.type = struct_value () [concrete]
// CHECK:STDOUT:   %Public.ProtectedStatic.cpp_overload_set.type: type = cpp_overload_set_type @Public.ProtectedStatic.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Public.ProtectedStatic.cpp_overload_set.value: %Public.ProtectedStatic.cpp_overload_set.type = cpp_overload_set_value @Public.ProtectedStatic.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Base.ProtectedStatic.type: type = fn_type @Base.ProtectedStatic [concrete]
// CHECK:STDOUT:   %Base.ProtectedStatic: %Base.ProtectedStatic.type = struct_value () [concrete]
// CHECK:STDOUT:   %Public.PublicInstance.cpp_overload_set.type: type = cpp_overload_set_type @Public.PublicInstance.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Public.PublicInstance.cpp_overload_set.value: %Public.PublicInstance.cpp_overload_set.type = cpp_overload_set_value @Public.PublicInstance.cpp_overload_set [concrete]
// CHECK:STDOUT:   %pattern_type.a3a: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %Base.PublicInstance.type: type = fn_type @Base.PublicInstance [concrete]
// CHECK:STDOUT:   %Base.PublicInstance: %Base.PublicInstance.type = struct_value () [concrete]
// CHECK:STDOUT:   %Public.ProtectedInstance.cpp_overload_set.type: type = cpp_overload_set_type @Public.ProtectedInstance.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Public.ProtectedInstance.cpp_overload_set.value: %Public.ProtectedInstance.cpp_overload_set.type = cpp_overload_set_value @Public.ProtectedInstance.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Base.ProtectedInstance.type: type = fn_type @Base.ProtectedInstance [concrete]
// CHECK:STDOUT:   %Base.ProtectedInstance: %Base.ProtectedInstance.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Public = %Public.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Public.decl: type = class_decl @Public [concrete = constants.%Public] {} {}
// CHECK:STDOUT:   %Public.PublicStatic.cpp_overload_set.value: %Public.PublicStatic.cpp_overload_set.type = cpp_overload_set_value @Public.PublicStatic.cpp_overload_set [concrete = constants.%Public.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.decl: %Base.PublicStatic.type = fn_decl @Base.PublicStatic [concrete = constants.%Base.PublicStatic] {} {}
// CHECK:STDOUT:   %Public.ProtectedStatic.cpp_overload_set.value: %Public.ProtectedStatic.cpp_overload_set.type = cpp_overload_set_value @Public.ProtectedStatic.cpp_overload_set [concrete = constants.%Public.ProtectedStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedStatic.decl: %Base.ProtectedStatic.type = fn_decl @Base.ProtectedStatic [concrete = constants.%Base.ProtectedStatic] {} {}
// CHECK:STDOUT:   %Public.PublicInstance.cpp_overload_set.value: %Public.PublicInstance.cpp_overload_set.type = cpp_overload_set_value @Public.PublicInstance.cpp_overload_set [concrete = constants.%Public.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicInstance.decl: %Base.PublicInstance.type = fn_decl @Base.PublicInstance [concrete = constants.%Base.PublicInstance] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a3a = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.a3a = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Base = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Base = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Public.ProtectedInstance.cpp_overload_set.value: %Public.ProtectedInstance.cpp_overload_set.type = cpp_overload_set_value @Public.ProtectedInstance.cpp_overload_set [concrete = constants.%Public.ProtectedInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedInstance.decl: %Base.ProtectedInstance.type = fn_decl @Base.ProtectedInstance [concrete = constants.%Base.ProtectedInstance] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a3a = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.a3a = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Base = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Base = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .CallStatic = %Derived.CallStatic.decl
// CHECK:STDOUT:   .CallInstance = %Derived.CallInstance.decl
// CHECK:STDOUT:   .PublicStatic = <poisoned>
// CHECK:STDOUT:   .ProtectedStatic = <poisoned>
// CHECK:STDOUT:   extend %Public.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.CallStatic() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Public.ref.loc25: type = name_ref Public, imports.%Public.decl [concrete = constants.%Public]
// CHECK:STDOUT:   %PublicStatic.ref.loc27: %Public.PublicStatic.cpp_overload_set.type = name_ref PublicStatic, imports.%Public.PublicStatic.cpp_overload_set.value [concrete = constants.%Public.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.call.loc30: init %empty_tuple.type = call imports.%Base.PublicStatic.decl()
// CHECK:STDOUT:   %Self.ref.loc32: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:   %PublicStatic.ref.loc33: %Public.PublicStatic.cpp_overload_set.type = name_ref PublicStatic, imports.%Public.PublicStatic.cpp_overload_set.value [concrete = constants.%Public.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.call.loc36: init %empty_tuple.type = call imports.%Base.PublicStatic.decl()
// CHECK:STDOUT:   %Cpp.ref.loc38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Public.ref.loc39: type = name_ref Public, imports.%Public.decl [concrete = constants.%Public]
// CHECK:STDOUT:   %ProtectedStatic.ref.loc41: %Public.ProtectedStatic.cpp_overload_set.type = name_ref ProtectedStatic, imports.%Public.ProtectedStatic.cpp_overload_set.value [concrete = constants.%Public.ProtectedStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedStatic.call.loc44: init %empty_tuple.type = call imports.%Base.ProtectedStatic.decl()
// CHECK:STDOUT:   %Self.ref.loc46: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:   %ProtectedStatic.ref.loc47: %Public.ProtectedStatic.cpp_overload_set.type = name_ref ProtectedStatic, imports.%Public.ProtectedStatic.cpp_overload_set.value [concrete = constants.%Public.ProtectedStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedStatic.call.loc50: init %empty_tuple.type = call imports.%Base.ProtectedStatic.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.CallInstance(%instance.param: ref %Public) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %instance.ref.loc64: ref %Public = name_ref instance, %instance
// CHECK:STDOUT:   %PublicInstance.ref: %Public.PublicInstance.cpp_overload_set.type = name_ref PublicInstance, imports.%Public.PublicInstance.cpp_overload_set.value [concrete = constants.%Public.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc65: <bound method> = bound_method %instance.ref.loc64, %PublicInstance.ref
// CHECK:STDOUT:   %.loc64.1: ref %Base = class_element_access %instance.ref.loc64, element0
// CHECK:STDOUT:   %.loc64.2: ref %Base = converted %instance.ref.loc64, %.loc64.1
// CHECK:STDOUT:   %Base.PublicInstance.call: init %empty_tuple.type = call imports.%Base.PublicInstance.decl(%.loc64.2)
// CHECK:STDOUT:   %instance.ref.loc70: ref %Public = name_ref instance, %instance
// CHECK:STDOUT:   %ProtectedInstance.ref: %Public.ProtectedInstance.cpp_overload_set.type = name_ref ProtectedInstance, imports.%Public.ProtectedInstance.cpp_overload_set.value [concrete = constants.%Public.ProtectedInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc71: <bound method> = bound_method %instance.ref.loc70, %ProtectedInstance.ref
// CHECK:STDOUT:   %.loc70.1: ref %Base = class_element_access %instance.ref.loc70, element0
// CHECK:STDOUT:   %.loc70.2: ref %Base = converted %instance.ref.loc70, %.loc70.1
// CHECK:STDOUT:   %Base.ProtectedInstance.call: init %empty_tuple.type = call imports.%Base.ProtectedInstance.decl(%.loc70.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%instance.param: ref %Public) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc89: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Public.ref.loc90: type = name_ref Public, imports.%Public.decl [concrete = constants.%Public]
// CHECK:STDOUT:   %PublicStatic.ref.loc92: %Public.PublicStatic.cpp_overload_set.type = name_ref PublicStatic, imports.%Public.PublicStatic.cpp_overload_set.value [concrete = constants.%Public.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.call.loc95: init %empty_tuple.type = call imports.%Base.PublicStatic.decl()
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %PublicStatic.ref.loc98: %Public.PublicStatic.cpp_overload_set.type = name_ref PublicStatic, imports.%Public.PublicStatic.cpp_overload_set.value [concrete = constants.%Public.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.call.loc101: init %empty_tuple.type = call imports.%Base.PublicStatic.decl()
// CHECK:STDOUT:   %instance.ref.loc103: ref %Public = name_ref instance, %instance
// CHECK:STDOUT:   %PublicInstance.ref.loc104: %Public.PublicInstance.cpp_overload_set.type = name_ref PublicInstance, imports.%Public.PublicInstance.cpp_overload_set.value [concrete = constants.%Public.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc104: <bound method> = bound_method %instance.ref.loc103, %PublicInstance.ref.loc104
// CHECK:STDOUT:   %.loc103.1: ref %Base = class_element_access %instance.ref.loc103, element0
// CHECK:STDOUT:   %.loc103.2: ref %Base = converted %instance.ref.loc103, %.loc103.1
// CHECK:STDOUT:   %Base.PublicInstance.call.loc107: init %empty_tuple.type = call imports.%Base.PublicInstance.decl(%.loc103.2)
// CHECK:STDOUT:   %instance.ref.loc109: ref %Public = name_ref instance, %instance
// CHECK:STDOUT:   %PublicInstance.ref.loc110: %Public.PublicInstance.cpp_overload_set.type = name_ref PublicInstance, imports.%Public.PublicInstance.cpp_overload_set.value [concrete = constants.%Public.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc110: <bound method> = bound_method %instance.ref.loc109, %PublicInstance.ref.loc110
// CHECK:STDOUT:   %.loc109.1: ref %Base = class_element_access %instance.ref.loc109, element0
// CHECK:STDOUT:   %.loc109.2: ref %Base = converted %instance.ref.loc109, %.loc109.1
// CHECK:STDOUT:   %Base.PublicInstance.call.loc113: init %empty_tuple.type = call imports.%Base.PublicInstance.decl(%.loc109.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_base_class_public_access_denied.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Public: type = class_type @Public [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .CallStatic = %Derived.CallStatic.decl
// CHECK:STDOUT:   .CallInstance = %Derived.CallInstance.decl
// CHECK:STDOUT:   .PrivateStatic = <poisoned>
// CHECK:STDOUT:   .ProtectedStatic = <poisoned>
// CHECK:STDOUT:   extend %Public.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.CallInstance(%instance.param: ref %Public) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %instance.ref: ref %Public = name_ref instance, %instance
// CHECK:STDOUT:   %PrivateInstance.ref: <error> = name_ref PrivateInstance, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- base_class_protected_access_allowed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Protected: type = class_type @Protected [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %Protected.PublicStatic.cpp_overload_set.type: type = cpp_overload_set_type @Protected.PublicStatic.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Protected.PublicStatic.cpp_overload_set.value: %Protected.PublicStatic.cpp_overload_set.type = cpp_overload_set_value @Protected.PublicStatic.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Base.PublicStatic.type: type = fn_type @Base.PublicStatic [concrete]
// CHECK:STDOUT:   %Base.PublicStatic: %Base.PublicStatic.type = struct_value () [concrete]
// CHECK:STDOUT:   %Protected.ProtectedStatic.cpp_overload_set.type: type = cpp_overload_set_type @Protected.ProtectedStatic.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Protected.ProtectedStatic.cpp_overload_set.value: %Protected.ProtectedStatic.cpp_overload_set.type = cpp_overload_set_value @Protected.ProtectedStatic.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Base.ProtectedStatic.type: type = fn_type @Base.ProtectedStatic [concrete]
// CHECK:STDOUT:   %Base.ProtectedStatic: %Base.ProtectedStatic.type = struct_value () [concrete]
// CHECK:STDOUT:   %Protected.PublicInstance.cpp_overload_set.type: type = cpp_overload_set_type @Protected.PublicInstance.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Protected.PublicInstance.cpp_overload_set.value: %Protected.PublicInstance.cpp_overload_set.type = cpp_overload_set_value @Protected.PublicInstance.cpp_overload_set [concrete]
// CHECK:STDOUT:   %pattern_type.a3a: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %Base.PublicInstance.type: type = fn_type @Base.PublicInstance [concrete]
// CHECK:STDOUT:   %Base.PublicInstance: %Base.PublicInstance.type = struct_value () [concrete]
// CHECK:STDOUT:   %Protected.ProtectedInstance.cpp_overload_set.type: type = cpp_overload_set_type @Protected.ProtectedInstance.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Protected.ProtectedInstance.cpp_overload_set.value: %Protected.ProtectedInstance.cpp_overload_set.type = cpp_overload_set_value @Protected.ProtectedInstance.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Base.ProtectedInstance.type: type = fn_type @Base.ProtectedInstance [concrete]
// CHECK:STDOUT:   %Base.ProtectedInstance: %Base.ProtectedInstance.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Protected = %Protected.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Protected.decl: type = class_decl @Protected [concrete = constants.%Protected] {} {}
// CHECK:STDOUT:   %Protected.PublicStatic.cpp_overload_set.value: %Protected.PublicStatic.cpp_overload_set.type = cpp_overload_set_value @Protected.PublicStatic.cpp_overload_set [concrete = constants.%Protected.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.decl: %Base.PublicStatic.type = fn_decl @Base.PublicStatic [concrete = constants.%Base.PublicStatic] {} {}
// CHECK:STDOUT:   %Protected.ProtectedStatic.cpp_overload_set.value: %Protected.ProtectedStatic.cpp_overload_set.type = cpp_overload_set_value @Protected.ProtectedStatic.cpp_overload_set [concrete = constants.%Protected.ProtectedStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedStatic.decl: %Base.ProtectedStatic.type = fn_decl @Base.ProtectedStatic [concrete = constants.%Base.ProtectedStatic] {} {}
// CHECK:STDOUT:   %Protected.PublicInstance.cpp_overload_set.value: %Protected.PublicInstance.cpp_overload_set.type = cpp_overload_set_value @Protected.PublicInstance.cpp_overload_set [concrete = constants.%Protected.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicInstance.decl: %Base.PublicInstance.type = fn_decl @Base.PublicInstance [concrete = constants.%Base.PublicInstance] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a3a = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.a3a = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Base = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Base = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Protected.ProtectedInstance.cpp_overload_set.value: %Protected.ProtectedInstance.cpp_overload_set.type = cpp_overload_set_value @Protected.ProtectedInstance.cpp_overload_set [concrete = constants.%Protected.ProtectedInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedInstance.decl: %Base.ProtectedInstance.type = fn_decl @Base.ProtectedInstance [concrete = constants.%Base.ProtectedInstance] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a3a = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.a3a = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Base = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Base = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .CallStatic = %Derived.CallStatic.decl
// CHECK:STDOUT:   .CallInstance = %Derived.CallInstance.decl
// CHECK:STDOUT:   .PublicStatic = <poisoned>
// CHECK:STDOUT:   .ProtectedStatic = <poisoned>
// CHECK:STDOUT:   extend %Protected.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.CallStatic() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Protected.ref.loc25: type = name_ref Protected, imports.%Protected.decl [concrete = constants.%Protected]
// CHECK:STDOUT:   %PublicStatic.ref.loc27: %Protected.PublicStatic.cpp_overload_set.type = name_ref PublicStatic, imports.%Protected.PublicStatic.cpp_overload_set.value [concrete = constants.%Protected.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.call.loc30: init %empty_tuple.type = call imports.%Base.PublicStatic.decl()
// CHECK:STDOUT:   %Self.ref.loc32: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:   %PublicStatic.ref.loc33: %Protected.PublicStatic.cpp_overload_set.type = name_ref PublicStatic, imports.%Protected.PublicStatic.cpp_overload_set.value [concrete = constants.%Protected.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.call.loc36: init %empty_tuple.type = call imports.%Base.PublicStatic.decl()
// CHECK:STDOUT:   %Cpp.ref.loc38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Protected.ref.loc39: type = name_ref Protected, imports.%Protected.decl [concrete = constants.%Protected]
// CHECK:STDOUT:   %ProtectedStatic.ref.loc41: %Protected.ProtectedStatic.cpp_overload_set.type = name_ref ProtectedStatic, imports.%Protected.ProtectedStatic.cpp_overload_set.value [concrete = constants.%Protected.ProtectedStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedStatic.call.loc44: init %empty_tuple.type = call imports.%Base.ProtectedStatic.decl()
// CHECK:STDOUT:   %Self.ref.loc46: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:   %ProtectedStatic.ref.loc47: %Protected.ProtectedStatic.cpp_overload_set.type = name_ref ProtectedStatic, imports.%Protected.ProtectedStatic.cpp_overload_set.value [concrete = constants.%Protected.ProtectedStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedStatic.call.loc50: init %empty_tuple.type = call imports.%Base.ProtectedStatic.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.CallInstance(%instance.param: ref %Protected) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %instance.ref.loc64: ref %Protected = name_ref instance, %instance
// CHECK:STDOUT:   %PublicInstance.ref: %Protected.PublicInstance.cpp_overload_set.type = name_ref PublicInstance, imports.%Protected.PublicInstance.cpp_overload_set.value [concrete = constants.%Protected.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc65: <bound method> = bound_method %instance.ref.loc64, %PublicInstance.ref
// CHECK:STDOUT:   %.loc64.1: ref %Base = class_element_access %instance.ref.loc64, element0
// CHECK:STDOUT:   %.loc64.2: ref %Base = converted %instance.ref.loc64, %.loc64.1
// CHECK:STDOUT:   %Base.PublicInstance.call: init %empty_tuple.type = call imports.%Base.PublicInstance.decl(%.loc64.2)
// CHECK:STDOUT:   %instance.ref.loc70: ref %Protected = name_ref instance, %instance
// CHECK:STDOUT:   %ProtectedInstance.ref: %Protected.ProtectedInstance.cpp_overload_set.type = name_ref ProtectedInstance, imports.%Protected.ProtectedInstance.cpp_overload_set.value [concrete = constants.%Protected.ProtectedInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc71: <bound method> = bound_method %instance.ref.loc70, %ProtectedInstance.ref
// CHECK:STDOUT:   %.loc70.1: ref %Base = class_element_access %instance.ref.loc70, element0
// CHECK:STDOUT:   %.loc70.2: ref %Base = converted %instance.ref.loc70, %.loc70.1
// CHECK:STDOUT:   %Base.ProtectedInstance.call: init %empty_tuple.type = call imports.%Base.ProtectedInstance.decl(%.loc70.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- base_class_public_protected_access_allowed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %PublicProtected: type = class_type @PublicProtected [concrete]
// CHECK:STDOUT:   %Protected: type = class_type @Protected [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %PublicProtected.PublicStatic.cpp_overload_set.type: type = cpp_overload_set_type @PublicProtected.PublicStatic.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicProtected.PublicStatic.cpp_overload_set.value: %PublicProtected.PublicStatic.cpp_overload_set.type = cpp_overload_set_value @PublicProtected.PublicStatic.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Base.PublicStatic.type: type = fn_type @Base.PublicStatic [concrete]
// CHECK:STDOUT:   %Base.PublicStatic: %Base.PublicStatic.type = struct_value () [concrete]
// CHECK:STDOUT:   %PublicProtected.ProtectedStatic.cpp_overload_set.type: type = cpp_overload_set_type @PublicProtected.ProtectedStatic.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicProtected.ProtectedStatic.cpp_overload_set.value: %PublicProtected.ProtectedStatic.cpp_overload_set.type = cpp_overload_set_value @PublicProtected.ProtectedStatic.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Base.ProtectedStatic.type: type = fn_type @Base.ProtectedStatic [concrete]
// CHECK:STDOUT:   %Base.ProtectedStatic: %Base.ProtectedStatic.type = struct_value () [concrete]
// CHECK:STDOUT:   %PublicProtected.PublicInstance.cpp_overload_set.type: type = cpp_overload_set_type @PublicProtected.PublicInstance.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicProtected.PublicInstance.cpp_overload_set.value: %PublicProtected.PublicInstance.cpp_overload_set.type = cpp_overload_set_value @PublicProtected.PublicInstance.cpp_overload_set [concrete]
// CHECK:STDOUT:   %pattern_type.a3a: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %Base.PublicInstance.type: type = fn_type @Base.PublicInstance [concrete]
// CHECK:STDOUT:   %Base.PublicInstance: %Base.PublicInstance.type = struct_value () [concrete]
// CHECK:STDOUT:   %PublicProtected.ProtectedInstance.cpp_overload_set.type: type = cpp_overload_set_type @PublicProtected.ProtectedInstance.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicProtected.ProtectedInstance.cpp_overload_set.value: %PublicProtected.ProtectedInstance.cpp_overload_set.type = cpp_overload_set_value @PublicProtected.ProtectedInstance.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Base.ProtectedInstance.type: type = fn_type @Base.ProtectedInstance [concrete]
// CHECK:STDOUT:   %Base.ProtectedInstance: %Base.ProtectedInstance.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .PublicProtected = %PublicProtected.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PublicProtected.decl: type = class_decl @PublicProtected [concrete = constants.%PublicProtected] {} {}
// CHECK:STDOUT:   %PublicProtected.PublicStatic.cpp_overload_set.value: %PublicProtected.PublicStatic.cpp_overload_set.type = cpp_overload_set_value @PublicProtected.PublicStatic.cpp_overload_set [concrete = constants.%PublicProtected.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.decl: %Base.PublicStatic.type = fn_decl @Base.PublicStatic [concrete = constants.%Base.PublicStatic] {} {}
// CHECK:STDOUT:   %PublicProtected.ProtectedStatic.cpp_overload_set.value: %PublicProtected.ProtectedStatic.cpp_overload_set.type = cpp_overload_set_value @PublicProtected.ProtectedStatic.cpp_overload_set [concrete = constants.%PublicProtected.ProtectedStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedStatic.decl: %Base.ProtectedStatic.type = fn_decl @Base.ProtectedStatic [concrete = constants.%Base.ProtectedStatic] {} {}
// CHECK:STDOUT:   %PublicProtected.PublicInstance.cpp_overload_set.value: %PublicProtected.PublicInstance.cpp_overload_set.type = cpp_overload_set_value @PublicProtected.PublicInstance.cpp_overload_set [concrete = constants.%PublicProtected.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicInstance.decl: %Base.PublicInstance.type = fn_decl @Base.PublicInstance [concrete = constants.%Base.PublicInstance] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a3a = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.a3a = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Base = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Base = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PublicProtected.ProtectedInstance.cpp_overload_set.value: %PublicProtected.ProtectedInstance.cpp_overload_set.type = cpp_overload_set_value @PublicProtected.ProtectedInstance.cpp_overload_set [concrete = constants.%PublicProtected.ProtectedInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedInstance.decl: %Base.ProtectedInstance.type = fn_decl @Base.ProtectedInstance [concrete = constants.%Base.ProtectedInstance] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a3a = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.a3a = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Base = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Base = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .CallStatic = %Derived.CallStatic.decl
// CHECK:STDOUT:   .CallInstance = %Derived.CallInstance.decl
// CHECK:STDOUT:   .PublicStatic = <poisoned>
// CHECK:STDOUT:   .ProtectedStatic = <poisoned>
// CHECK:STDOUT:   extend %PublicProtected.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.CallStatic() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicProtected.ref.loc25: type = name_ref PublicProtected, imports.%PublicProtected.decl [concrete = constants.%PublicProtected]
// CHECK:STDOUT:   %PublicStatic.ref.loc27: %PublicProtected.PublicStatic.cpp_overload_set.type = name_ref PublicStatic, imports.%PublicProtected.PublicStatic.cpp_overload_set.value [concrete = constants.%PublicProtected.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.call.loc30: init %empty_tuple.type = call imports.%Base.PublicStatic.decl()
// CHECK:STDOUT:   %Self.ref.loc32: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:   %PublicStatic.ref.loc33: %PublicProtected.PublicStatic.cpp_overload_set.type = name_ref PublicStatic, imports.%PublicProtected.PublicStatic.cpp_overload_set.value [concrete = constants.%PublicProtected.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.call.loc36: init %empty_tuple.type = call imports.%Base.PublicStatic.decl()
// CHECK:STDOUT:   %Cpp.ref.loc38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicProtected.ref.loc39: type = name_ref PublicProtected, imports.%PublicProtected.decl [concrete = constants.%PublicProtected]
// CHECK:STDOUT:   %ProtectedStatic.ref.loc41: %PublicProtected.ProtectedStatic.cpp_overload_set.type = name_ref ProtectedStatic, imports.%PublicProtected.ProtectedStatic.cpp_overload_set.value [concrete = constants.%PublicProtected.ProtectedStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedStatic.call.loc44: init %empty_tuple.type = call imports.%Base.ProtectedStatic.decl()
// CHECK:STDOUT:   %Self.ref.loc46: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:   %ProtectedStatic.ref.loc47: %PublicProtected.ProtectedStatic.cpp_overload_set.type = name_ref ProtectedStatic, imports.%PublicProtected.ProtectedStatic.cpp_overload_set.value [concrete = constants.%PublicProtected.ProtectedStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedStatic.call.loc50: init %empty_tuple.type = call imports.%Base.ProtectedStatic.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.CallInstance(%instance.param: ref %PublicProtected) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %instance.ref.loc64: ref %PublicProtected = name_ref instance, %instance
// CHECK:STDOUT:   %PublicInstance.ref: %PublicProtected.PublicInstance.cpp_overload_set.type = name_ref PublicInstance, imports.%PublicProtected.PublicInstance.cpp_overload_set.value [concrete = constants.%PublicProtected.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc65: <bound method> = bound_method %instance.ref.loc64, %PublicInstance.ref
// CHECK:STDOUT:   %.loc64.1: ref %Protected = class_element_access %instance.ref.loc64, element0
// CHECK:STDOUT:   %.loc64.2: ref %Base = class_element_access %.loc64.1, element0
// CHECK:STDOUT:   %.loc64.3: ref %Base = converted %instance.ref.loc64, %.loc64.2
// CHECK:STDOUT:   %Base.PublicInstance.call: init %empty_tuple.type = call imports.%Base.PublicInstance.decl(%.loc64.3)
// CHECK:STDOUT:   %instance.ref.loc70: ref %PublicProtected = name_ref instance, %instance
// CHECK:STDOUT:   %ProtectedInstance.ref: %PublicProtected.ProtectedInstance.cpp_overload_set.type = name_ref ProtectedInstance, imports.%PublicProtected.ProtectedInstance.cpp_overload_set.value [concrete = constants.%PublicProtected.ProtectedInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc71: <bound method> = bound_method %instance.ref.loc70, %ProtectedInstance.ref
// CHECK:STDOUT:   %.loc70.1: ref %Protected = class_element_access %instance.ref.loc70, element0
// CHECK:STDOUT:   %.loc70.2: ref %Base = class_element_access %.loc70.1, element0
// CHECK:STDOUT:   %.loc70.3: ref %Base = converted %instance.ref.loc70, %.loc70.2
// CHECK:STDOUT:   %Base.ProtectedInstance.call: init %empty_tuple.type = call imports.%Base.ProtectedInstance.decl(%.loc70.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
