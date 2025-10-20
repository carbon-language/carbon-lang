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
  let instance_data: i32 = s.instance_data;
  let static_data: i32 = Cpp.S.static_data;
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
  let instance_data: i32 = d.instance_data;
  let static_data: i32 = Derived.static_data;
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
  // CHECK:STDERR: fail_import_non_function_member_protected.carbon:[[@LINE+8]]:28: error: cannot access protected member `instance_data` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let instance_data: i32 = c.instance_data;
  // CHECK:STDERR:                            ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_protected.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./non_function_member_protected.h:2:7: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: class C {
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let instance_data: i32 = c.instance_data;
  // CHECK:STDERR: fail_import_non_function_member_protected.carbon:[[@LINE+7]]:26: error: cannot access protected member `static_data` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let static_data: i32 = Cpp.C.static_data;
  // CHECK:STDERR:                          ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_protected.carbon:[[@LINE+4]]:26: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   let static_data: i32 = Cpp.C.static_data;
  // CHECK:STDERR:                          ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let static_data: i32 = Cpp.C.static_data;
}

// --- fail_import_non_function_member_protected_extend.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_protected.h";

class Derived {
  extend base: Cpp.C;
}

fn F(d: Derived) {
  // CHECK:STDERR: fail_import_non_function_member_protected_extend.carbon:[[@LINE+8]]:28: error: cannot access protected member `instance_data` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let instance_data: i32 = d.instance_data;
  // CHECK:STDERR:                            ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_protected_extend.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./non_function_member_protected.h:2:7: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: class C {
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let instance_data: i32 = d.instance_data;
  // CHECK:STDERR: fail_import_non_function_member_protected_extend.carbon:[[@LINE+7]]:26: error: cannot access protected member `static_data` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let static_data: i32 = Derived.static_data;
  // CHECK:STDERR:                          ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_protected_extend.carbon:[[@LINE+4]]:26: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   let static_data: i32 = Derived.static_data;
  // CHECK:STDERR:                          ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let static_data: i32 = Derived.static_data;
}

// --- import_non_function_member_protected_extend_call_in_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_protected.h";

class Derived {
  extend base: Cpp.C;
  fn F[self: Self]() {
    //@dump-sem-ir-begin
    let instance_data: i32 = self.instance_data;
    let unqualified_static_data: i32 = static_data;
    let derived_static_data: i32 = Derived.static_data;
    let base_static_data: i32 = Cpp.C.static_data;
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
  // CHECK:STDERR: fail_import_non_function_member_private.carbon:[[@LINE+8]]:28: error: cannot access private member `instance_data` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let instance_data: i32 = c.instance_data;
  // CHECK:STDERR:                            ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_private.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./non_function_member_private.h:2:7: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: class C {
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let instance_data: i32 = c.instance_data;
  // CHECK:STDERR: fail_import_non_function_member_private.carbon:[[@LINE+7]]:26: error: cannot access private member `static_data` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let static_data: i32 = Cpp.C.static_data;
  // CHECK:STDERR:                          ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_private.carbon:[[@LINE+4]]:26: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   let static_data: i32 = Cpp.C.static_data;
  // CHECK:STDERR:                          ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let static_data: i32 = Cpp.C.static_data;
}

// --- fail_import_non_function_member_private_extend.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_private.h";

class Derived {
  extend base: Cpp.C;
  fn F[self: Self]() {
    // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE+8]]:30: error: cannot access private member `instance_data` of type `Cpp.C` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     let instance_data: i32 = self.instance_data;
    // CHECK:STDERR:                              ^~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE-8]]:10: in file included here [InCppInclude]
    // CHECK:STDERR: ./non_function_member_private.h:2:7: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR: class C {
    // CHECK:STDERR:       ^
    // CHECK:STDERR:
    let instance_data: i32 = self.instance_data;
  }
}

fn F(d: Derived) {
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE+8]]:28: error: cannot access private member `instance_data` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let instance_data: i32 = d.instance_data;
  // CHECK:STDERR:                            ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE-21]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./non_function_member_private.h:2:7: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: class C {
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let instance_data: i32 = d.instance_data;
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE+7]]:34: error: cannot access private member `static_data` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let derived_static_data: i32 = Derived.static_data;
  // CHECK:STDERR:                                  ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE+4]]:34: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   let derived_static_data: i32 = Derived.static_data;
  // CHECK:STDERR:                                  ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let derived_static_data: i32 = Derived.static_data;
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE+7]]:31: error: cannot access private member `static_data` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let base_static_data: i32 = Cpp.C.static_data;
  // CHECK:STDERR:                               ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE-4]]:34: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   let derived_static_data: i32 = Derived.static_data;
  // CHECK:STDERR:                                  ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let base_static_data: i32 = Cpp.C.static_data;
}

// --- todo_fail_import_non_function_member_private_extend_static_within_member_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_private.h";

class Derived {
  extend base: Cpp.C;
  fn F() {
    // TODO: `C.static_data` is private, so this should fail.
    let unqualified_static_data: i32 = static_data;
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
  // CHECK:STDERR: fail_import_function_member_protected.carbon:[[@LINE+5]]:3: error: cannot access protected member `instance_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   c->instance_fn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_function_member_protected.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  c->instance_fn();
  // CHECK:STDERR: fail_import_function_member_protected.carbon:[[@LINE+5]]:3: error: cannot access protected member `static_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.C.static_fn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_function_member_protected.carbon: note: declared here [ClassMemberDeclaration]
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
  fn F[self: Self]() {
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
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+5]]:5: error: cannot access private member `instance_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     self.instance_fn();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_function_member_private.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    self.instance_fn();
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+5]]:5: error: cannot access private member `static_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     static_fn();
    // CHECK:STDERR:     ^~~~~~~~~~~
    // CHECK:STDERR: fail_import_function_member_private.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    static_fn();
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+5]]:5: error: cannot access private member `static_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Derived.static_fn();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_function_member_private.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Derived.static_fn();
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+5]]:5: error: cannot access private member `static_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.C.static_fn();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_function_member_private.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.C.static_fn();
  }
}

fn F(c: Cpp.C*) {
  // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+5]]:3: error: cannot access private member `instance_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   c->instance_fn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_function_member_private.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  c->instance_fn();
  // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+5]]:3: error: cannot access private member `static_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.static_fn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_function_member_private.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.static_fn();
  // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+5]]:3: error: cannot access private member `static_fn` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.C.static_fn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_function_member_private.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.C.static_fn();
}

// ============================================================================
// Overload set
// ============================================================================

// --- overload_set.h

class C {
 public:
  C();
  static auto Overload() -> void;
 protected:
  C(int);
  static auto Overload(int a) -> void;
 private:
  C(int, int);
  static auto Overload(int a, int b) -> void;
};

// --- import_overload_set_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.C.C();
  Cpp.C.Overload();
  //@dump-sem-ir-end
}

// --- fail_import_overload_set_protected.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

fn F() {
  // CHECK:STDERR: fail_import_overload_set_protected.carbon:[[@LINE+5]]:3: error: cannot access protected member `C` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.C.C(1 as i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_protected.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.C.C(1 as i32);

  // CHECK:STDERR: fail_import_overload_set_protected.carbon:[[@LINE+5]]:3: error: cannot access protected member `Overload` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.C.Overload(1 as i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_protected.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.C.Overload(1 as i32);
}

// --- import_overload_set_protected_base.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

class D {
  extend base: Cpp.C;

  fn MakePublic() -> D {
    return {.base = C()};
  }

  fn MakeProtected() -> D {
    return {.base = C(1 as i32)};
  }

  fn CallPublic() {
    Overload();
  }

  fn CallProtected() {
    Overload(1 as i32);
  }
}

// --- fail_import_overload_set_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

fn F() {
  // CHECK:STDERR: fail_import_overload_set_private.carbon:[[@LINE+5]]:3: error: cannot access private member `C` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.C.C(1 as i32, 2 as i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_private.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.C.C(1 as i32, 2 as i32);

  // CHECK:STDERR: fail_import_overload_set_private.carbon:[[@LINE+5]]:3: error: cannot access private member `Overload` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.C.Overload(1 as i32, 2 as i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_private.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.C.Overload(1 as i32, 2 as i32);
}

// --- fail_import_overload_set_private_base.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

class D {
  extend base: Cpp.C;

  fn MakePrivate() -> D {
    // CHECK:STDERR: fail_import_overload_set_private_base.carbon:[[@LINE+5]]:21: error: cannot access private member `C` of type `Cpp.C` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     return {.base = C(1 as i32, 2 as i32)};
    // CHECK:STDERR:                     ^~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_overload_set_private_base.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    return {.base = C(1 as i32, 2 as i32)};
  }

  fn CallPrivate() {
    // CHECK:STDERR: fail_import_overload_set_private_base.carbon:[[@LINE+5]]:5: error: cannot access private member `Overload` of type `Cpp.C` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Overload(1 as i32, 2 as i32);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_overload_set_private_base.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Overload(1 as i32, 2 as i32);
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
  Cpp.Public.Overload();
  //@dump-sem-ir-end
}

// --- fail_import_overload_set_public_base_class_call_non_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

fn F() {
  // CHECK:STDERR: fail_import_overload_set_public_base_class_call_non_public.carbon:[[@LINE+5]]:3: error: cannot access protected member `Overload` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Public.Overload(1);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_public_base_class_call_non_public.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Public.Overload(1);
  // CHECK:STDERR: fail_import_overload_set_public_base_class_call_non_public.carbon:[[@LINE+5]]:3: error: cannot access private member `Overload` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Public.Overload(1, 2);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_public_base_class_call_non_public.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Public.Overload(1, 2);
}

// --- import_overload_set_public_base_class_derived_call_non_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

class Derived {
  extend base: Cpp.Public;

  fn F() {
    //@dump-sem-ir-begin
    Overload();
    Overload(1);
    //@dump-sem-ir-end
  }
}

// --- fail_import_overload_set_public_base_class_derived_call_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

class Derived {
  extend base: Cpp.Public;

  fn F() {
    // CHECK:STDERR: fail_import_overload_set_public_base_class_derived_call_private.carbon:[[@LINE+5]]:5: error: cannot access private member `Overload` of type `Cpp.Public` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Overload(1, 2);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_overload_set_public_base_class_derived_call_private.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Overload(1, 2);
  }
}

// --- fail_import_overload_set_protected_base_class.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

fn F() {
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon:[[@LINE+5]]:3: error: cannot access protected member `Overload` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Protected.Overload();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Protected.Overload();
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon:[[@LINE+5]]:3: error: cannot access protected member `Overload` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Protected.Overload(1);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Protected.Overload(1);
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon:[[@LINE+5]]:3: error: cannot access protected member `Overload` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Protected.Overload(1, 2);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Protected.Overload(1, 2);
}

// --- import_overload_set_protected_base_class_derived_call_non_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

class Derived {
  extend base: Cpp.Protected;

  fn F() {
    //@dump-sem-ir-begin
    Overload();
    Overload(1);
    //@dump-sem-ir-end
  }
}

// --- fail_import_overload_set_protected_base_class_derived_call_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

class Derived {
  extend base: Cpp.Protected;

  fn F() {
    // CHECK:STDERR: fail_import_overload_set_protected_base_class_derived_call_private.carbon:[[@LINE+5]]:5: error: cannot access private member `Overload` of type `Cpp.Protected` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Overload(1, 2);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_overload_set_protected_base_class_derived_call_private.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Overload(1, 2);
  }
}

// --- fail_import_overload_set_private_base_class.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

fn F() {
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon:[[@LINE+5]]:3: error: cannot access private member `Overload` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Private.Overload();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Private.Overload();
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon:[[@LINE+5]]:3: error: cannot access private member `Overload` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Private.Overload(1);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Private.Overload(1);
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon:[[@LINE+5]]:3: error: cannot access private member `Overload` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Private.Overload(1, 2);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Private.Overload(1, 2);
}

// --- fail_import_overload_set_private_base_class_derived.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

class Derived {
  extend base: Cpp.Private;

  fn F() {
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon:[[@LINE+5]]:5: error: cannot access private member `Overload` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Overload();
    // CHECK:STDERR:     ^~~~~~~~~~
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Overload();
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon:[[@LINE+5]]:5: error: cannot access private member `Overload` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Overload(1);
    // CHECK:STDERR:     ^~~~~~~~~~~
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Overload(1);
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon:[[@LINE+5]]:5: error: cannot access private member `Overload` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Overload(1, 2);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Overload(1, 2);
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

// --- fail_base_class_public_access_disallowed.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

class Derived {
  extend base: Cpp.Public;

  fn CallStatic() {
    // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.Public` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.Public.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.Public.PrivateStatic();
    // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.Public` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.PrivateStatic();
  }

  fn CallInstance(var instance: Cpp.Public) {
    //@dump-sem-ir-begin
    // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PrivateInstance` of type `Cpp.Public` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.PrivateInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.PrivateInstance();
    //@dump-sem-ir-end
  }
}

fn Call(var instance: Cpp.Public) {
  // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access protected member `ProtectedStatic` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Public.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Public.ProtectedStatic();
  // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access protected member `ProtectedStatic` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.ProtectedStatic();

  // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Public.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Public.PrivateStatic();
  // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PrivateStatic();

  // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access protected member `ProtectedInstance` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.ProtectedInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.ProtectedInstance();
  // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PrivateInstance` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PrivateInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
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

// --- fail_base_class_protected_access_disallowed.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

class Derived {
  extend base: Cpp.Protected;

  fn CallStatic() {
    // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.Protected.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.Protected.PrivateStatic();
    // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.PrivateStatic();
  }

  fn CallInstance(var instance: Cpp.Protected) {
    // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PrivateInstance` of type `Cpp.Protected` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.PrivateInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.PrivateInstance();
  }
}

fn Call(var instance: Cpp.Protected) {
  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access protected member `PublicStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Protected.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Protected.PublicStatic();
  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access protected member `PublicStatic` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PublicStatic();

  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access protected member `ProtectedStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Protected.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Protected.ProtectedStatic();
  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access protected member `ProtectedStatic` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.ProtectedStatic();

  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Protected.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Protected.PrivateStatic();
  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PrivateStatic();

  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access protected member `PublicInstance` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PublicInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.PublicInstance();
  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access protected member `ProtectedInstance` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.ProtectedInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.ProtectedInstance();
  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PrivateInstance` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PrivateInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.PrivateInstance();
}

// --- fail_base_class_private_access_disallowed.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

class Derived {
  extend base: Cpp.Private;

  fn CallStatic() {
    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PublicStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.Private.PublicStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.Private.PublicStatic();
    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PublicStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.PublicStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.PublicStatic();

    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `ProtectedStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.Private.ProtectedStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.Private.ProtectedStatic();
    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `ProtectedStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.ProtectedStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.ProtectedStatic();

    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.Private.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.Private.PrivateStatic();
    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.PrivateStatic();
  }

  fn CallInstance(var instance: Cpp.Private) {
    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PublicInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.PublicInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.PublicInstance();

    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `ProtectedInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.ProtectedInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.ProtectedInstance();

    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PrivateInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.PrivateInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.PrivateInstance();
  }
}

fn Call(var instance: Cpp.Private) {
  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PublicStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Private.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Private.PublicStatic();
  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PublicStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PublicStatic();

  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `ProtectedStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Private.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Private.ProtectedStatic();
  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `ProtectedStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.ProtectedStatic();

  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.Private.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Private.PrivateStatic();
  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PrivateStatic();

  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PublicInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PublicInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.PublicInstance();
  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `ProtectedInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.ProtectedInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.ProtectedInstance();
  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PrivateInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PrivateInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
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

// --- fail_base_class_public_protected_access_disallowed.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

class Derived {
  extend base: Cpp.PublicProtected;

  fn CallStatic() {
    // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.PublicProtected.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.PublicProtected.PrivateStatic();
    // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.PrivateStatic();
  }

  fn CallInstance(var instance: Cpp.PublicProtected) {
    // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PrivateInstance` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.PrivateInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.PrivateInstance();
  }
}

fn Call(var instance: Cpp.PublicProtected) {
  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access protected member `PublicStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.PublicProtected.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.PublicProtected.PublicStatic();
  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access protected member `PublicStatic` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PublicStatic();

  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access protected member `ProtectedStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.PublicProtected.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.PublicProtected.ProtectedStatic();
  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access protected member `ProtectedStatic` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.ProtectedStatic();

  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.PublicProtected.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.PublicProtected.PrivateStatic();
  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PrivateStatic();

  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access protected member `PublicInstance` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PublicInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.PublicInstance();
  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access protected member `ProtectedInstance` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.ProtectedInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.ProtectedInstance();
  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PrivateInstance` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PrivateInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_protected_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.PrivateInstance();
}

// --- fail_base_class_public_private_access_disallowed.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

class Derived {
  extend base: Cpp.PublicPrivate;

  fn CallStatic() {
    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PublicStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.PublicPrivate.PublicStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.PublicPrivate.PublicStatic();
    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PublicStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.PublicStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.PublicStatic();

    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `ProtectedStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.PublicPrivate.ProtectedStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.PublicPrivate.ProtectedStatic();
    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `ProtectedStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.ProtectedStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.ProtectedStatic();

    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Cpp.PublicPrivate.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.PublicPrivate.PrivateStatic();
    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PrivateStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     Self.PrivateStatic();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.PrivateStatic();
  }

  fn CallInstance(var instance: Cpp.PublicPrivate) {
    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PublicInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.PublicInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.PublicInstance();

    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `ProtectedInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.ProtectedInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.ProtectedInstance();

    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:5: error: cannot access private member `PrivateInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     instance.PrivateInstance();
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.PrivateInstance();
  }
}

fn Call(var instance: Cpp.PublicPrivate) {
  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PublicStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.PublicPrivate.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.PublicPrivate.PublicStatic();
  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PublicStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PublicStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PublicStatic();

  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `ProtectedStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.PublicPrivate.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.PublicPrivate.ProtectedStatic();
  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `ProtectedStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.ProtectedStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.ProtectedStatic();

  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.PublicPrivate.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.PublicPrivate.PrivateStatic();
  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PrivateStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Derived.PrivateStatic();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PrivateStatic();

  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PublicInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PublicInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.PublicInstance();
  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `ProtectedInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.ProtectedInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.ProtectedInstance();
  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon:[[@LINE+5]]:3: error: cannot access private member `PrivateInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   instance.PrivateInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_base_class_public_private_access_disallowed.carbon: note: declared here [ClassMemberDeclaration]
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
// CHECK:STDOUT:   %static_data.var: ref %i32 = var %static_data.var_patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %S) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %instance_data.patt: %pattern_type.7ce = binding_pattern instance_data [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %S = name_ref s, %s
// CHECK:STDOUT:   %instance_data.ref: %S.elem = name_ref instance_data, @S.%.1 [concrete = @S.%.1]
// CHECK:STDOUT:   %.loc8_29.1: ref %i32 = class_element_access %s.ref, element0
// CHECK:STDOUT:   %.loc8_29.2: %i32 = bind_value %.loc8_29.1
// CHECK:STDOUT:   %.loc8_22: type = splice_block %i32.loc8 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %instance_data: %i32 = bind_name instance_data, %.loc8_29.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %static_data.patt: %pattern_type.7ce = binding_pattern static_data [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc9: type = name_ref S, imports.%S.decl [concrete = constants.%S]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %static_data.ref: ref %i32 = name_ref static_data, imports.%static_data.var
// CHECK:STDOUT:   %.loc9_20: type = splice_block %i32.loc9 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_31: %i32 = bind_value %static_data.ref
// CHECK:STDOUT:   %static_data: %i32 = bind_name static_data, %.loc9_31
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
// CHECK:STDOUT:   %static_data.var: ref %i32 = var %static_data.var_patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%d.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %instance_data.patt: %pattern_type.7ce = binding_pattern instance_data [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %instance_data.ref: %S.elem = name_ref instance_data, @S.%.1 [concrete = @S.%.1]
// CHECK:STDOUT:   %.loc12_29.1: ref %S = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc12_29.2: ref %S = converted %d.ref, %.loc12_29.1
// CHECK:STDOUT:   %.loc12_29.3: ref %i32 = class_element_access %.loc12_29.2, element0
// CHECK:STDOUT:   %.loc12_22: type = splice_block %i32.loc12 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_29.4: %i32 = bind_value %.loc12_29.3
// CHECK:STDOUT:   %instance_data: %i32 = bind_name instance_data, %.loc12_29.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %static_data.patt: %pattern_type.7ce = binding_pattern static_data [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.ref.loc13: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %static_data.ref: ref %i32 = name_ref static_data, imports.%static_data.var
// CHECK:STDOUT:   %.loc13_20: type = splice_block %i32.loc13 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_33: %i32 = bind_value %static_data.ref
// CHECK:STDOUT:   %static_data: %i32 = bind_name static_data, %.loc13_33
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
// CHECK:STDOUT:   %static_data.var: ref %i32 = var %static_data.var_patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Cpp = <poisoned>
// CHECK:STDOUT:   .base = %.loc7
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
// CHECK:STDOUT:     %instance_data.patt: %pattern_type.7ce = binding_pattern instance_data [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %self.ref: %Derived = name_ref self, %self
// CHECK:STDOUT:   %instance_data.ref: %C.elem = name_ref instance_data, @C.%.1 [concrete = @C.%.1]
// CHECK:STDOUT:   %.loc10_34.1: ref %C = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc10_34.2: ref %C = converted %self.ref, %.loc10_34.1
// CHECK:STDOUT:   %.loc10_34.3: ref %i32 = class_element_access %.loc10_34.2, element0
// CHECK:STDOUT:   %.loc10_24: type = splice_block %i32.loc10 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_34.4: %i32 = bind_value %.loc10_34.3
// CHECK:STDOUT:   %instance_data: %i32 = bind_name instance_data, %.loc10_34.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %unqualified_static_data.patt: %pattern_type.7ce = binding_pattern unqualified_static_data [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %static_data.ref.loc11: ref %i32 = name_ref static_data, imports.%static_data.var
// CHECK:STDOUT:   %.loc11_34: type = splice_block %i32.loc11 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_40: %i32 = bind_value %static_data.ref.loc11
// CHECK:STDOUT:   %unqualified_static_data: %i32 = bind_name unqualified_static_data, %.loc11_40
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %derived_static_data.patt: %pattern_type.7ce = binding_pattern derived_static_data [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %static_data.ref.loc12: ref %i32 = name_ref static_data, imports.%static_data.var
// CHECK:STDOUT:   %.loc12_30: type = splice_block %i32.loc12 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_43: %i32 = bind_value %static_data.ref.loc12
// CHECK:STDOUT:   %derived_static_data: %i32 = bind_name derived_static_data, %.loc12_43
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %base_static_data.patt: %pattern_type.7ce = binding_pattern base_static_data [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %static_data.ref.loc13: ref %i32 = name_ref static_data, imports.%static_data.var
// CHECK:STDOUT:   %.loc13_27: type = splice_block %i32.loc13 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_38: %i32 = bind_value %static_data.ref.loc13
// CHECK:STDOUT:   %base_static_data: %i32 = bind_name base_static_data, %.loc13_38
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_function_member_public.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %S.instance_fn.cpp_overload_set.type: type = cpp_overload_set_type @S.instance_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %S.instance_fn.cpp_overload_set.value: %S.instance_fn.cpp_overload_set.type = cpp_overload_set_value @S.instance_fn.cpp_overload_set [concrete]
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
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.static_fn.cpp_overload_set.value: %S.static_fn.cpp_overload_set.type = cpp_overload_set_value @S.static_fn.cpp_overload_set [concrete = constants.%S.static_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %S.static_fn.decl: %S.static_fn.type = fn_decl @S.static_fn [concrete = constants.%S.static_fn] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %ptr.5c7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %ptr.5c7 = name_ref s, %s
// CHECK:STDOUT:   %.loc8: ref %S = deref %s.ref
// CHECK:STDOUT:   %instance_fn.ref: %S.instance_fn.cpp_overload_set.type = name_ref instance_fn, imports.%S.instance_fn.cpp_overload_set.value [concrete = constants.%S.instance_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8, %instance_fn.ref
// CHECK:STDOUT:   %addr: %ptr.5c7 = addr_of %.loc8
// CHECK:STDOUT:   %S.instance_fn.call: init %empty_tuple.type = call imports.%S.instance_fn.decl(%addr)
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %S.ref.loc9: type = name_ref S, imports.%S.decl [concrete = constants.%S]
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
// CHECK:STDOUT:   .Cpp = <poisoned>
// CHECK:STDOUT:   .base = %.loc7
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
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
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
// CHECK:STDOUT:   .Cpp = <poisoned>
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   .static_fn = <poisoned>
// CHECK:STDOUT:   .Derived = <poisoned>
// CHECK:STDOUT:   extend %C.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.F(%self.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %static_fn.ref.loc10: %C.static_fn.cpp_overload_set.type = name_ref static_fn, imports.%C.static_fn.cpp_overload_set.value [concrete = constants.%C.static_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %C.static_fn.call.loc10: init %empty_tuple.type = call imports.%C.static_fn.decl()
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %static_fn.ref.loc11: %C.static_fn.cpp_overload_set.type = name_ref static_fn, imports.%C.static_fn.cpp_overload_set.value [concrete = constants.%C.static_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %C.static_fn.call.loc11: init %empty_tuple.type = call imports.%C.static_fn.decl()
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %static_fn.ref.loc12: %C.static_fn.cpp_overload_set.type = name_ref static_fn, imports.%C.static_fn.cpp_overload_set.value [concrete = constants.%C.static_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %C.static_fn.call.loc12: init %empty_tuple.type = call imports.%C.static_fn.decl()
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
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.Overload.cpp_overload_set.type: type = cpp_overload_set_type @C.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.Overload.cpp_overload_set.value: %C.Overload.cpp_overload_set.type = cpp_overload_set_value @C.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.Overload.type: type = fn_type @C.Overload [concrete]
// CHECK:STDOUT:   %C.Overload: %C.Overload.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.841: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.Overload.cpp_overload_set.value: %C.Overload.cpp_overload_set.type = cpp_overload_set_value @C.Overload.cpp_overload_set [concrete = constants.%C.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %C.Overload.decl: %C.Overload.type = fn_decl @C.Overload [concrete = constants.%C.Overload] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc8_6: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc8_8: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_11.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc8_11.1: %ptr.d9e = addr_of %.loc8_11.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc8_11.1)
// CHECK:STDOUT:   %.loc8_11.2: init %C = in_place_init %C__carbon_thunk.call, %.loc8_11.1
// CHECK:STDOUT:   %.loc8_11.3: ref %C = temporary %.loc8_11.1, %.loc8_11.2
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc9: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Overload.ref: %C.Overload.cpp_overload_set.type = name_ref Overload, imports.%C.Overload.cpp_overload_set.value [concrete = constants.%C.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %C.Overload.call: init %empty_tuple.type = call imports.%C.Overload.decl()
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_11.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_11.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_11.2: %ptr.d9e = addr_of %.loc8_11.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_11.2)
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
// CHECK:STDOUT:   %C.Overload.type: type = fn_type @C.Overload [concrete]
// CHECK:STDOUT:   %C.Overload: %C.Overload.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .Public = %Public.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Public.decl: type = class_decl @Public [concrete = constants.%Public] {} {}
// CHECK:STDOUT:   %Public.Overload.cpp_overload_set.value: %Public.Overload.cpp_overload_set.type = cpp_overload_set_value @Public.Overload.cpp_overload_set [concrete = constants.%Public.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %C.Overload.decl: %C.Overload.type = fn_decl @C.Overload [concrete = constants.%C.Overload] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Public.ref: type = name_ref Public, imports.%Public.decl [concrete = constants.%Public]
// CHECK:STDOUT:   %Overload.ref: %Public.Overload.cpp_overload_set.type = name_ref Overload, imports.%Public.Overload.cpp_overload_set.value [concrete = constants.%Public.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %C.Overload.call: init %empty_tuple.type = call imports.%C.Overload.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overload_set_public_base_class_derived_call_non_private.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Public.Overload.cpp_overload_set.type: type = cpp_overload_set_type @Public.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Public.Overload.cpp_overload_set.value: %Public.Overload.cpp_overload_set.type = cpp_overload_set_value @Public.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.Overload.type.fd2af6.1: type = fn_type @C.Overload.1 [concrete]
// CHECK:STDOUT:   %C.Overload.5043eb.1: %C.Overload.type.fd2af6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.Overload.type.fd2af6.2: type = fn_type @C.Overload.2 [concrete]
// CHECK:STDOUT:   %C.Overload.5043eb.2: %C.Overload.type.fd2af6.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Public.Overload.cpp_overload_set.value: %Public.Overload.cpp_overload_set.type = cpp_overload_set_value @Public.Overload.cpp_overload_set [concrete = constants.%Public.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %C.Overload.decl.b29afa.1: %C.Overload.type.fd2af6.1 = fn_decl @C.Overload.1 [concrete = constants.%C.Overload.5043eb.1] {} {}
// CHECK:STDOUT:   %C.Overload.decl.b29afa.2: %C.Overload.type.fd2af6.2 = fn_decl @C.Overload.2 [concrete = constants.%C.Overload.5043eb.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Cpp = <poisoned>
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   .Overload = <poisoned>
// CHECK:STDOUT:   extend %Public.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Overload.ref.loc11: %Public.Overload.cpp_overload_set.type = name_ref Overload, imports.%Public.Overload.cpp_overload_set.value [concrete = constants.%Public.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %C.Overload.call.loc11: init %empty_tuple.type = call imports.%C.Overload.decl.b29afa.1()
// CHECK:STDOUT:   %Overload.ref.loc12: %Public.Overload.cpp_overload_set.type = name_ref Overload, imports.%Public.Overload.cpp_overload_set.value [concrete = constants.%Public.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc12_14.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_14.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_14.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_14.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_14.2: %i32 = converted %int_1, %.loc12_14.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %C.Overload.call.loc12: init %empty_tuple.type = call imports.%C.Overload.decl.b29afa.2(%.loc12_14.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overload_set_protected_base_class_derived_call_non_private.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Protected.Overload.cpp_overload_set.type: type = cpp_overload_set_type @Protected.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Protected.Overload.cpp_overload_set.value: %Protected.Overload.cpp_overload_set.type = cpp_overload_set_value @Protected.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.Overload.type.fd2af6.1: type = fn_type @C.Overload.1 [concrete]
// CHECK:STDOUT:   %C.Overload.5043eb.1: %C.Overload.type.fd2af6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.Overload.type.fd2af6.2: type = fn_type @C.Overload.2 [concrete]
// CHECK:STDOUT:   %C.Overload.5043eb.2: %C.Overload.type.fd2af6.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Protected.Overload.cpp_overload_set.value: %Protected.Overload.cpp_overload_set.type = cpp_overload_set_value @Protected.Overload.cpp_overload_set [concrete = constants.%Protected.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %C.Overload.decl.b29afa.1: %C.Overload.type.fd2af6.1 = fn_decl @C.Overload.1 [concrete = constants.%C.Overload.5043eb.1] {} {}
// CHECK:STDOUT:   %C.Overload.decl.b29afa.2: %C.Overload.type.fd2af6.2 = fn_decl @C.Overload.2 [concrete = constants.%C.Overload.5043eb.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Cpp = <poisoned>
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   .Overload = <poisoned>
// CHECK:STDOUT:   extend %Protected.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Overload.ref.loc11: %Protected.Overload.cpp_overload_set.type = name_ref Overload, imports.%Protected.Overload.cpp_overload_set.value [concrete = constants.%Protected.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %C.Overload.call.loc11: init %empty_tuple.type = call imports.%C.Overload.decl.b29afa.1()
// CHECK:STDOUT:   %Overload.ref.loc12: %Protected.Overload.cpp_overload_set.type = name_ref Overload, imports.%Protected.Overload.cpp_overload_set.value [concrete = constants.%Protected.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc12_14.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_14.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc12_14.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_14.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_14.2: %i32 = converted %int_1, %.loc12_14.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %C.Overload.call.loc12: init %empty_tuple.type = call imports.%C.Overload.decl.b29afa.2(%.loc12_14.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- base_class_public_access_allowed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Public: type = class_type @Public [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
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
// CHECK:STDOUT:   %ptr.fb2: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %Base.PublicInstance.type: type = fn_type @Base.PublicInstance [concrete]
// CHECK:STDOUT:   %Base.PublicInstance: %Base.PublicInstance.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.1e8: type = ptr_type %Public [concrete]
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
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Public.ProtectedInstance.cpp_overload_set.value: %Public.ProtectedInstance.cpp_overload_set.type = cpp_overload_set_value @Public.ProtectedInstance.cpp_overload_set [concrete = constants.%Public.ProtectedInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedInstance.decl: %Base.ProtectedInstance.type = fn_decl @Base.ProtectedInstance [concrete = constants.%Base.ProtectedInstance] {
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
// CHECK:STDOUT:   .Cpp = <poisoned>
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .CallStatic = %Derived.CallStatic.decl
// CHECK:STDOUT:   .CallInstance = %Derived.CallInstance.decl
// CHECK:STDOUT:   .PublicStatic = <poisoned>
// CHECK:STDOUT:   .ProtectedStatic = <poisoned>
// CHECK:STDOUT:   extend %Public.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.CallStatic() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Public.ref.loc12: type = name_ref Public, imports.%Public.decl [concrete = constants.%Public]
// CHECK:STDOUT:   %PublicStatic.ref.loc12: %Public.PublicStatic.cpp_overload_set.type = name_ref PublicStatic, imports.%Public.PublicStatic.cpp_overload_set.value [concrete = constants.%Public.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.call.loc12: init %empty_tuple.type = call imports.%Base.PublicStatic.decl()
// CHECK:STDOUT:   %Self.ref.loc13: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:   %PublicStatic.ref.loc13: %Public.PublicStatic.cpp_overload_set.type = name_ref PublicStatic, imports.%Public.PublicStatic.cpp_overload_set.value [concrete = constants.%Public.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.call.loc13: init %empty_tuple.type = call imports.%Base.PublicStatic.decl()
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Public.ref.loc16: type = name_ref Public, imports.%Public.decl [concrete = constants.%Public]
// CHECK:STDOUT:   %ProtectedStatic.ref.loc16: %Public.ProtectedStatic.cpp_overload_set.type = name_ref ProtectedStatic, imports.%Public.ProtectedStatic.cpp_overload_set.value [concrete = constants.%Public.ProtectedStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedStatic.call.loc16: init %empty_tuple.type = call imports.%Base.ProtectedStatic.decl()
// CHECK:STDOUT:   %Self.ref.loc17: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:   %ProtectedStatic.ref.loc17: %Public.ProtectedStatic.cpp_overload_set.type = name_ref ProtectedStatic, imports.%Public.ProtectedStatic.cpp_overload_set.value [concrete = constants.%Public.ProtectedStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedStatic.call.loc17: init %empty_tuple.type = call imports.%Base.ProtectedStatic.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.CallInstance(%instance.param: %Public) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %instance.ref.loc24: ref %Public = name_ref instance, %instance
// CHECK:STDOUT:   %PublicInstance.ref: %Public.PublicInstance.cpp_overload_set.type = name_ref PublicInstance, imports.%Public.PublicInstance.cpp_overload_set.value [concrete = constants.%Public.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc24: <bound method> = bound_method %instance.ref.loc24, %PublicInstance.ref
// CHECK:STDOUT:   %addr.loc24_5.1: %ptr.1e8 = addr_of %instance.ref.loc24
// CHECK:STDOUT:   %.loc24_5.1: ref %Public = deref %addr.loc24_5.1
// CHECK:STDOUT:   %.loc24_5.2: ref %Base = class_element_access %.loc24_5.1, element0
// CHECK:STDOUT:   %addr.loc24_5.2: %ptr.fb2 = addr_of %.loc24_5.2
// CHECK:STDOUT:   %.loc24_5.3: %ptr.fb2 = converted %addr.loc24_5.1, %addr.loc24_5.2
// CHECK:STDOUT:   %Base.PublicInstance.call: init %empty_tuple.type = call imports.%Base.PublicInstance.decl(%.loc24_5.3)
// CHECK:STDOUT:   %instance.ref.loc26: ref %Public = name_ref instance, %instance
// CHECK:STDOUT:   %ProtectedInstance.ref: %Public.ProtectedInstance.cpp_overload_set.type = name_ref ProtectedInstance, imports.%Public.ProtectedInstance.cpp_overload_set.value [concrete = constants.%Public.ProtectedInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc26: <bound method> = bound_method %instance.ref.loc26, %ProtectedInstance.ref
// CHECK:STDOUT:   %addr.loc26_5.1: %ptr.1e8 = addr_of %instance.ref.loc26
// CHECK:STDOUT:   %.loc26_5.1: ref %Public = deref %addr.loc26_5.1
// CHECK:STDOUT:   %.loc26_5.2: ref %Base = class_element_access %.loc26_5.1, element0
// CHECK:STDOUT:   %addr.loc26_5.2: %ptr.fb2 = addr_of %.loc26_5.2
// CHECK:STDOUT:   %.loc26_5.3: %ptr.fb2 = converted %addr.loc26_5.1, %addr.loc26_5.2
// CHECK:STDOUT:   %Base.ProtectedInstance.call: init %empty_tuple.type = call imports.%Base.ProtectedInstance.decl(%.loc26_5.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%instance.param: %Public) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Public.ref.loc33: type = name_ref Public, imports.%Public.decl [concrete = constants.%Public]
// CHECK:STDOUT:   %PublicStatic.ref.loc33: %Public.PublicStatic.cpp_overload_set.type = name_ref PublicStatic, imports.%Public.PublicStatic.cpp_overload_set.value [concrete = constants.%Public.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.call.loc33: init %empty_tuple.type = call imports.%Base.PublicStatic.decl()
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %PublicStatic.ref.loc34: %Public.PublicStatic.cpp_overload_set.type = name_ref PublicStatic, imports.%Public.PublicStatic.cpp_overload_set.value [concrete = constants.%Public.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.call.loc34: init %empty_tuple.type = call imports.%Base.PublicStatic.decl()
// CHECK:STDOUT:   %instance.ref.loc36: ref %Public = name_ref instance, %instance
// CHECK:STDOUT:   %PublicInstance.ref.loc36: %Public.PublicInstance.cpp_overload_set.type = name_ref PublicInstance, imports.%Public.PublicInstance.cpp_overload_set.value [concrete = constants.%Public.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc36: <bound method> = bound_method %instance.ref.loc36, %PublicInstance.ref.loc36
// CHECK:STDOUT:   %addr.loc36_3.1: %ptr.1e8 = addr_of %instance.ref.loc36
// CHECK:STDOUT:   %.loc36_3.1: ref %Public = deref %addr.loc36_3.1
// CHECK:STDOUT:   %.loc36_3.2: ref %Base = class_element_access %.loc36_3.1, element0
// CHECK:STDOUT:   %addr.loc36_3.2: %ptr.fb2 = addr_of %.loc36_3.2
// CHECK:STDOUT:   %.loc36_3.3: %ptr.fb2 = converted %addr.loc36_3.1, %addr.loc36_3.2
// CHECK:STDOUT:   %Base.PublicInstance.call.loc36: init %empty_tuple.type = call imports.%Base.PublicInstance.decl(%.loc36_3.3)
// CHECK:STDOUT:   %instance.ref.loc37: ref %Public = name_ref instance, %instance
// CHECK:STDOUT:   %PublicInstance.ref.loc37: %Public.PublicInstance.cpp_overload_set.type = name_ref PublicInstance, imports.%Public.PublicInstance.cpp_overload_set.value [concrete = constants.%Public.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc37: <bound method> = bound_method %instance.ref.loc37, %PublicInstance.ref.loc37
// CHECK:STDOUT:   %addr.loc37_3.1: %ptr.1e8 = addr_of %instance.ref.loc37
// CHECK:STDOUT:   %.loc37_3.1: ref %Public = deref %addr.loc37_3.1
// CHECK:STDOUT:   %.loc37_3.2: ref %Base = class_element_access %.loc37_3.1, element0
// CHECK:STDOUT:   %addr.loc37_3.2: %ptr.fb2 = addr_of %.loc37_3.2
// CHECK:STDOUT:   %.loc37_3.3: %ptr.fb2 = converted %addr.loc37_3.1, %addr.loc37_3.2
// CHECK:STDOUT:   %Base.PublicInstance.call.loc37: init %empty_tuple.type = call imports.%Base.PublicInstance.decl(%.loc37_3.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_base_class_public_access_disallowed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Public: type = class_type @Public [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Cpp = <poisoned>
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .CallStatic = %Derived.CallStatic.decl
// CHECK:STDOUT:   .CallInstance = %Derived.CallInstance.decl
// CHECK:STDOUT:   .PrivateStatic = <poisoned>
// CHECK:STDOUT:   .ProtectedStatic = <poisoned>
// CHECK:STDOUT:   extend %Public.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.CallInstance(%instance.param: %Public) {
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
// CHECK:STDOUT:   %Protected: type = class_type @Protected [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
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
// CHECK:STDOUT:   %ptr.fb2: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %Base.PublicInstance.type: type = fn_type @Base.PublicInstance [concrete]
// CHECK:STDOUT:   %Base.PublicInstance: %Base.PublicInstance.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.f97: type = ptr_type %Protected [concrete]
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
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Protected.ProtectedInstance.cpp_overload_set.value: %Protected.ProtectedInstance.cpp_overload_set.type = cpp_overload_set_value @Protected.ProtectedInstance.cpp_overload_set [concrete = constants.%Protected.ProtectedInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedInstance.decl: %Base.ProtectedInstance.type = fn_decl @Base.ProtectedInstance [concrete = constants.%Base.ProtectedInstance] {
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
// CHECK:STDOUT:   .Cpp = <poisoned>
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .CallStatic = %Derived.CallStatic.decl
// CHECK:STDOUT:   .CallInstance = %Derived.CallInstance.decl
// CHECK:STDOUT:   .PublicStatic = <poisoned>
// CHECK:STDOUT:   .ProtectedStatic = <poisoned>
// CHECK:STDOUT:   extend %Protected.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.CallStatic() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Protected.ref.loc12: type = name_ref Protected, imports.%Protected.decl [concrete = constants.%Protected]
// CHECK:STDOUT:   %PublicStatic.ref.loc12: %Protected.PublicStatic.cpp_overload_set.type = name_ref PublicStatic, imports.%Protected.PublicStatic.cpp_overload_set.value [concrete = constants.%Protected.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.call.loc12: init %empty_tuple.type = call imports.%Base.PublicStatic.decl()
// CHECK:STDOUT:   %Self.ref.loc13: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:   %PublicStatic.ref.loc13: %Protected.PublicStatic.cpp_overload_set.type = name_ref PublicStatic, imports.%Protected.PublicStatic.cpp_overload_set.value [concrete = constants.%Protected.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.call.loc13: init %empty_tuple.type = call imports.%Base.PublicStatic.decl()
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Protected.ref.loc16: type = name_ref Protected, imports.%Protected.decl [concrete = constants.%Protected]
// CHECK:STDOUT:   %ProtectedStatic.ref.loc16: %Protected.ProtectedStatic.cpp_overload_set.type = name_ref ProtectedStatic, imports.%Protected.ProtectedStatic.cpp_overload_set.value [concrete = constants.%Protected.ProtectedStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedStatic.call.loc16: init %empty_tuple.type = call imports.%Base.ProtectedStatic.decl()
// CHECK:STDOUT:   %Self.ref.loc17: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:   %ProtectedStatic.ref.loc17: %Protected.ProtectedStatic.cpp_overload_set.type = name_ref ProtectedStatic, imports.%Protected.ProtectedStatic.cpp_overload_set.value [concrete = constants.%Protected.ProtectedStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedStatic.call.loc17: init %empty_tuple.type = call imports.%Base.ProtectedStatic.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.CallInstance(%instance.param: %Protected) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %instance.ref.loc24: ref %Protected = name_ref instance, %instance
// CHECK:STDOUT:   %PublicInstance.ref: %Protected.PublicInstance.cpp_overload_set.type = name_ref PublicInstance, imports.%Protected.PublicInstance.cpp_overload_set.value [concrete = constants.%Protected.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc24: <bound method> = bound_method %instance.ref.loc24, %PublicInstance.ref
// CHECK:STDOUT:   %addr.loc24_5.1: %ptr.f97 = addr_of %instance.ref.loc24
// CHECK:STDOUT:   %.loc24_5.1: ref %Protected = deref %addr.loc24_5.1
// CHECK:STDOUT:   %.loc24_5.2: ref %Base = class_element_access %.loc24_5.1, element0
// CHECK:STDOUT:   %addr.loc24_5.2: %ptr.fb2 = addr_of %.loc24_5.2
// CHECK:STDOUT:   %.loc24_5.3: %ptr.fb2 = converted %addr.loc24_5.1, %addr.loc24_5.2
// CHECK:STDOUT:   %Base.PublicInstance.call: init %empty_tuple.type = call imports.%Base.PublicInstance.decl(%.loc24_5.3)
// CHECK:STDOUT:   %instance.ref.loc26: ref %Protected = name_ref instance, %instance
// CHECK:STDOUT:   %ProtectedInstance.ref: %Protected.ProtectedInstance.cpp_overload_set.type = name_ref ProtectedInstance, imports.%Protected.ProtectedInstance.cpp_overload_set.value [concrete = constants.%Protected.ProtectedInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc26: <bound method> = bound_method %instance.ref.loc26, %ProtectedInstance.ref
// CHECK:STDOUT:   %addr.loc26_5.1: %ptr.f97 = addr_of %instance.ref.loc26
// CHECK:STDOUT:   %.loc26_5.1: ref %Protected = deref %addr.loc26_5.1
// CHECK:STDOUT:   %.loc26_5.2: ref %Base = class_element_access %.loc26_5.1, element0
// CHECK:STDOUT:   %addr.loc26_5.2: %ptr.fb2 = addr_of %.loc26_5.2
// CHECK:STDOUT:   %.loc26_5.3: %ptr.fb2 = converted %addr.loc26_5.1, %addr.loc26_5.2
// CHECK:STDOUT:   %Base.ProtectedInstance.call: init %empty_tuple.type = call imports.%Base.ProtectedInstance.decl(%.loc26_5.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- base_class_public_protected_access_allowed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %PublicProtected: type = class_type @PublicProtected [concrete]
// CHECK:STDOUT:   %Protected: type = class_type @Protected [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
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
// CHECK:STDOUT:   %ptr.fb2: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %Base.PublicInstance.type: type = fn_type @Base.PublicInstance [concrete]
// CHECK:STDOUT:   %Base.PublicInstance: %Base.PublicInstance.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.dc4: type = ptr_type %PublicProtected [concrete]
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
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PublicProtected.ProtectedInstance.cpp_overload_set.value: %PublicProtected.ProtectedInstance.cpp_overload_set.type = cpp_overload_set_value @PublicProtected.ProtectedInstance.cpp_overload_set [concrete = constants.%PublicProtected.ProtectedInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedInstance.decl: %Base.ProtectedInstance.type = fn_decl @Base.ProtectedInstance [concrete = constants.%Base.ProtectedInstance] {
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
// CHECK:STDOUT:   .Cpp = <poisoned>
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .CallStatic = %Derived.CallStatic.decl
// CHECK:STDOUT:   .CallInstance = %Derived.CallInstance.decl
// CHECK:STDOUT:   .PublicStatic = <poisoned>
// CHECK:STDOUT:   .ProtectedStatic = <poisoned>
// CHECK:STDOUT:   extend %PublicProtected.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.CallStatic() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicProtected.ref.loc12: type = name_ref PublicProtected, imports.%PublicProtected.decl [concrete = constants.%PublicProtected]
// CHECK:STDOUT:   %PublicStatic.ref.loc12: %PublicProtected.PublicStatic.cpp_overload_set.type = name_ref PublicStatic, imports.%PublicProtected.PublicStatic.cpp_overload_set.value [concrete = constants.%PublicProtected.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.call.loc12: init %empty_tuple.type = call imports.%Base.PublicStatic.decl()
// CHECK:STDOUT:   %Self.ref.loc13: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:   %PublicStatic.ref.loc13: %PublicProtected.PublicStatic.cpp_overload_set.type = name_ref PublicStatic, imports.%PublicProtected.PublicStatic.cpp_overload_set.value [concrete = constants.%PublicProtected.PublicStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.PublicStatic.call.loc13: init %empty_tuple.type = call imports.%Base.PublicStatic.decl()
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicProtected.ref.loc16: type = name_ref PublicProtected, imports.%PublicProtected.decl [concrete = constants.%PublicProtected]
// CHECK:STDOUT:   %ProtectedStatic.ref.loc16: %PublicProtected.ProtectedStatic.cpp_overload_set.type = name_ref ProtectedStatic, imports.%PublicProtected.ProtectedStatic.cpp_overload_set.value [concrete = constants.%PublicProtected.ProtectedStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedStatic.call.loc16: init %empty_tuple.type = call imports.%Base.ProtectedStatic.decl()
// CHECK:STDOUT:   %Self.ref.loc17: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:   %ProtectedStatic.ref.loc17: %PublicProtected.ProtectedStatic.cpp_overload_set.type = name_ref ProtectedStatic, imports.%PublicProtected.ProtectedStatic.cpp_overload_set.value [concrete = constants.%PublicProtected.ProtectedStatic.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.ProtectedStatic.call.loc17: init %empty_tuple.type = call imports.%Base.ProtectedStatic.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.CallInstance(%instance.param: %PublicProtected) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %instance.ref.loc24: ref %PublicProtected = name_ref instance, %instance
// CHECK:STDOUT:   %PublicInstance.ref: %PublicProtected.PublicInstance.cpp_overload_set.type = name_ref PublicInstance, imports.%PublicProtected.PublicInstance.cpp_overload_set.value [concrete = constants.%PublicProtected.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc24: <bound method> = bound_method %instance.ref.loc24, %PublicInstance.ref
// CHECK:STDOUT:   %addr.loc24_5.1: %ptr.dc4 = addr_of %instance.ref.loc24
// CHECK:STDOUT:   %.loc24_5.1: ref %PublicProtected = deref %addr.loc24_5.1
// CHECK:STDOUT:   %.loc24_5.2: ref %Protected = class_element_access %.loc24_5.1, element0
// CHECK:STDOUT:   %.loc24_5.3: ref %Base = class_element_access %.loc24_5.2, element0
// CHECK:STDOUT:   %addr.loc24_5.2: %ptr.fb2 = addr_of %.loc24_5.3
// CHECK:STDOUT:   %.loc24_5.4: %ptr.fb2 = converted %addr.loc24_5.1, %addr.loc24_5.2
// CHECK:STDOUT:   %Base.PublicInstance.call: init %empty_tuple.type = call imports.%Base.PublicInstance.decl(%.loc24_5.4)
// CHECK:STDOUT:   %instance.ref.loc26: ref %PublicProtected = name_ref instance, %instance
// CHECK:STDOUT:   %ProtectedInstance.ref: %PublicProtected.ProtectedInstance.cpp_overload_set.type = name_ref ProtectedInstance, imports.%PublicProtected.ProtectedInstance.cpp_overload_set.value [concrete = constants.%PublicProtected.ProtectedInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc26: <bound method> = bound_method %instance.ref.loc26, %ProtectedInstance.ref
// CHECK:STDOUT:   %addr.loc26_5.1: %ptr.dc4 = addr_of %instance.ref.loc26
// CHECK:STDOUT:   %.loc26_5.1: ref %PublicProtected = deref %addr.loc26_5.1
// CHECK:STDOUT:   %.loc26_5.2: ref %Protected = class_element_access %.loc26_5.1, element0
// CHECK:STDOUT:   %.loc26_5.3: ref %Base = class_element_access %.loc26_5.2, element0
// CHECK:STDOUT:   %addr.loc26_5.2: %ptr.fb2 = addr_of %.loc26_5.3
// CHECK:STDOUT:   %.loc26_5.4: %ptr.fb2 = converted %addr.loc26_5.1, %addr.loc26_5.2
// CHECK:STDOUT:   %Base.ProtectedInstance.call: init %empty_tuple.type = call imports.%Base.ProtectedInstance.decl(%.loc26_5.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
