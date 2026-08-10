// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/import/access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/import/access.carbon

// ============================================================================
// Public non-function member
// ============================================================================

// --- non_function_member_public.h

struct NonFunctionMemberPublic {
  int instance_data;
  static int static_data;
};

// --- import_non_function_member_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_public.h";

fn F(s: Cpp.NonFunctionMemberPublic) {
  //@dump-sem-ir-begin
  let unused instance_data: i32 = s.instance_data;
  let unused static_data: i32 = Cpp.NonFunctionMemberPublic.static_data;
  //@dump-sem-ir-end
}

// --- import_non_function_member_public_extend.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_public.h";

class Derived {
  extend base: Cpp.NonFunctionMemberPublic;
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

class NonFunctionMemberProtected {
 protected:
  int instance_data;
  static int static_data;
};

// --- fail_import_non_function_member_protected.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_protected.h";

fn F(c: Cpp.NonFunctionMemberProtected) {
  // CHECK:STDERR: fail_import_non_function_member_protected.carbon:[[@LINE+4]]:35-49: error: cannot access protected member `instance_data` of type `Cpp.NonFunctionMemberProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: included from: fail_import_non_function_member_protected.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./non_function_member_protected.h:2:7: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  let unused instance_data: i32 = c.instance_data;
  // CHECK:STDERR: fail_import_non_function_member_protected.carbon:[[@LINE+3]]:33-74: error: cannot access protected member `static_data` of type `Cpp.NonFunctionMemberProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_import_non_function_member_protected.carbon:[[@LINE+2]]:33-74: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  let unused static_data: i32 = Cpp.NonFunctionMemberProtected.static_data;
}

// --- fail_import_non_function_member_protected_extend.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_protected.h";

class Derived {
  extend base: Cpp.NonFunctionMemberProtected;
}

fn F(d: Derived) {
  // CHECK:STDERR: fail_import_non_function_member_protected_extend.carbon:[[@LINE+4]]:35-49: error: cannot access protected member `instance_data` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR: included from: fail_import_non_function_member_protected_extend.carbon:[[@LINE-8]]:10 [InCppInclude]
  // CHECK:STDERR: ./non_function_member_protected.h:2:7: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  let unused instance_data: i32 = d.instance_data;
  // CHECK:STDERR: fail_import_non_function_member_protected_extend.carbon:[[@LINE+3]]:33-51: error: cannot access protected member `static_data` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_import_non_function_member_protected_extend.carbon:[[@LINE+2]]:33-51: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  let unused static_data: i32 = Derived.static_data;
}

// --- import_non_function_member_protected_extend_call_in_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_protected.h";

class Derived {
  extend base: Cpp.NonFunctionMemberProtected;
  fn F(self) {
    //@dump-sem-ir-begin
    let unused instance_data: i32 = self.instance_data;
    let unused unqualified_static_data: i32 = static_data;
    let unused derived_static_data: i32 = Derived.static_data;
    let unused base_static_data: i32 = Cpp.NonFunctionMemberProtected.static_data;
    //@dump-sem-ir-end
  }
}

// ============================================================================
// Private non-function member
// ============================================================================

// --- non_function_member_private.h

class NonFunctionMemberPrivate {
  int instance_data;
  static int static_data;
};

// --- fail_import_non_function_member_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_private.h";

fn F(c: Cpp.NonFunctionMemberPrivate) {
  // CHECK:STDERR: fail_import_non_function_member_private.carbon:[[@LINE+4]]:35-49: error: cannot access private member `instance_data` of type `Cpp.NonFunctionMemberPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: included from: fail_import_non_function_member_private.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./non_function_member_private.h:2:7: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  let unused instance_data: i32 = c.instance_data;
  // CHECK:STDERR: fail_import_non_function_member_private.carbon:[[@LINE+3]]:33-72: error: cannot access private member `static_data` of type `Cpp.NonFunctionMemberPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_import_non_function_member_private.carbon:[[@LINE+2]]:33-72: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  let unused static_data: i32 = Cpp.NonFunctionMemberPrivate.static_data;
}

// --- fail_import_non_function_member_private_extend.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_private.h";

class Derived {
  extend base: Cpp.NonFunctionMemberPrivate;
  fn F(self) {
    // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE+4]]:37-54: error: cannot access private member `instance_data` of type `Cpp.NonFunctionMemberPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: included from: fail_import_non_function_member_private_extend.carbon:[[@LINE-6]]:10 [InCppInclude]
    // CHECK:STDERR: ./non_function_member_private.h:2:7: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    let unused instance_data: i32 = self.instance_data;
  }
}

fn F(d: Derived) {
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE+4]]:35-49: error: cannot access private member `instance_data` of type `Cpp.NonFunctionMemberPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: included from: fail_import_non_function_member_private_extend.carbon:[[@LINE-15]]:10 [InCppInclude]
  // CHECK:STDERR: ./non_function_member_private.h:2:7: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  let unused instance_data: i32 = d.instance_data;
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE+3]]:41-59: error: cannot access private member `static_data` of type `Cpp.NonFunctionMemberPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE+2]]:41-59: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  let unused derived_static_data: i32 = Derived.static_data;
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE+3]]:38-77: error: cannot access private member `static_data` of type `Cpp.NonFunctionMemberPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE-2]]:41-59: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  let unused base_static_data: i32 = Cpp.NonFunctionMemberPrivate.static_data;
}

// --- fail_import_non_function_member_private_extend_static_within_member_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_private.h";

class Derived {
  extend base: Cpp.NonFunctionMemberPrivate;
  fn F() {
    // TODO: `NonFunctionMemberPrivate.static_data` is private, so this
    // should fail.
    // CHECK:STDERR: fail_import_non_function_member_private_extend_static_within_member_function.carbon:[[@LINE+3]]:47-57: error: cannot access private member `static_data` of type `Cpp.NonFunctionMemberPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_import_non_function_member_private_extend_static_within_member_function.carbon:[[@LINE+2]]:47-57: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    let unused unqualified_static_data: i32 = static_data;
  }
}

// ============================================================================
// Public function member
// ============================================================================

// --- function_member_public.h

struct FunctionMemberPublic {
  auto instance_fn() -> void;
  static auto static_fn() -> void;
};

// --- import_function_member_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "function_member_public.h";

fn F(s: Cpp.FunctionMemberPublic*) {
  //@dump-sem-ir-begin
  s->instance_fn();
  Cpp.FunctionMemberPublic.static_fn();
  //@dump-sem-ir-end
}

// ============================================================================
// Protected function member
// ============================================================================

// --- function_member_protected.h

class FunctionMemberProtected {
 protected:
  auto instance_fn() -> void;
  static auto static_fn() -> void;
};

// --- fail_import_function_member_protected.carbon

library "[[@TEST_NAME]]";

import Cpp library "function_member_protected.h";

fn F(c: Cpp.FunctionMemberProtected*) {
  // CHECK:STDERR: fail_import_function_member_protected.carbon:[[@LINE+3]]:3-16: error: cannot access protected member `instance_fn` of type `Cpp.FunctionMemberProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_import_function_member_protected.carbon:[[@LINE+2]]:3-16: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  c->instance_fn();
  // CHECK:STDERR: fail_import_function_member_protected.carbon:[[@LINE+3]]:3-39: error: cannot access protected member `static_fn` of type `Cpp.FunctionMemberProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_import_function_member_protected.carbon:[[@LINE+2]]:3-39: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.FunctionMemberProtected.static_fn();
}

// --- fail_todo_import_function_member_protected_extend_call_instance_in_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "function_member_protected.h";

class Derived {
  extend base: Cpp.FunctionMemberProtected;
  fn F(self) {
    //@dump-sem-ir-begin
    // TODO: It should be possible to call protected methods in the base class.
    // CHECK:STDERR: fail_todo_import_function_member_protected_extend_call_instance_in_member.carbon:[[@LINE+5]]:5-20: error: no matching function for call to 'instance_fn' [CppInteropParseError]
    // CHECK:STDERR: included from: fail_todo_import_function_member_protected_extend_call_instance_in_member.carbon:[[@LINE-8]]:10 [InCppInclude]
    // CHECK:STDERR: ./function_member_protected.h:4:8-18: note: candidate function not viable [CppInteropParseNote]
    // CHECK:STDERR: ./function_member_protected.h:4:19-20: note: cannot convert argument of incomplete type 'const Carbon::Derived' to 'FunctionMemberProtected' for object argument [CppInteropParseNoteReason]
    // CHECK:STDERR:
    self.instance_fn();
    //@dump-sem-ir-end
  }
}

// --- import_function_member_protected_extend_call_static_in_member.carbon

library "[[@TEST_NAME]]";

import Cpp library "function_member_protected.h";

class Derived {
  extend base: Cpp.FunctionMemberProtected;
  fn F(unused self) {
    //@dump-sem-ir-begin
    static_fn();
    Derived.static_fn();
    Cpp.FunctionMemberProtected.static_fn();
    //@dump-sem-ir-end
  }
}

// ============================================================================
// Private function member
// ============================================================================

// --- function_member_private.h

class FunctionMemberPrivate {
 private:
  auto instance_fn() -> void;
  static auto static_fn() -> void;
};

// --- fail_import_function_member_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "function_member_private.h";

class Derived {
  extend base: Cpp.FunctionMemberPrivate;
  fn F(self) {
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+3]]:5-20: error: cannot access private member `instance_fn` of type `Cpp.FunctionMemberPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+2]]:5-20: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    self.instance_fn();
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+3]]:5-13: error: cannot access private member `static_fn` of type `Cpp.FunctionMemberPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+2]]:5-13: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    static_fn();
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+3]]:5-21: error: cannot access private member `static_fn` of type `Cpp.FunctionMemberPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE-2]]:5-13: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Derived.static_fn();
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+3]]:5-39: error: cannot access private member `static_fn` of type `Cpp.FunctionMemberPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE-6]]:5-13: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.FunctionMemberPrivate.static_fn();
  }
}

fn F(c: Cpp.FunctionMemberPrivate*) {
  // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+3]]:3-16: error: cannot access private member `instance_fn` of type `Cpp.FunctionMemberPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE-18]]:5-20: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  c->instance_fn();
  // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+3]]:3-19: error: cannot access private member `static_fn` of type `Cpp.FunctionMemberPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE-18]]:5-13: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.static_fn();
  // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+3]]:3-37: error: cannot access private member `static_fn` of type `Cpp.FunctionMemberPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE-22]]:5-13: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.FunctionMemberPrivate.static_fn();
}

// ============================================================================
// Overload set
// ============================================================================

// --- overload_set.h

struct PublicCall {};
struct ProtectedCall {};
struct PrivateCall {};

class OverloadSet {
 public:
  OverloadSet(PublicCall x);
  static auto Overload(PublicCall x) -> void;
 protected:
  OverloadSet(ProtectedCall x);
  static auto Overload(ProtectedCall x) -> void;
 private:
  OverloadSet(PrivateCall x);
  static auto Overload(PrivateCall x) -> void;
};

// --- import_overload_set_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.OverloadSet.OverloadSet(Cpp.PublicCall.PublicCall());
  Cpp.OverloadSet.Overload(Cpp.PublicCall.PublicCall());
  //@dump-sem-ir-end
}

// --- fail_import_overload_set_protected.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

fn F() {
  // CHECK:STDERR: fail_import_overload_set_protected.carbon:[[@LINE+4]]:3-64: error: cannot access protected member `OverloadSet` of type `Cpp.OverloadSet` [ClassInvalidMemberAccess]
  // CHECK:STDERR: included from: fail_import_overload_set_protected.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./overload_set.h:11:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.OverloadSet.OverloadSet(Cpp.ProtectedCall.ProtectedCall());
  // CHECK:STDERR: fail_import_overload_set_protected.carbon:[[@LINE+4]]:3-61: error: cannot access protected member `Overload` of type `Cpp.OverloadSet` [ClassInvalidMemberAccess]
  // CHECK:STDERR: included from: fail_import_overload_set_protected.carbon:[[@LINE-9]]:10 [InCppInclude]
  // CHECK:STDERR: ./overload_set.h:12:15: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.OverloadSet.Overload(Cpp.ProtectedCall.ProtectedCall());
}

// --- import_overload_set_protected_base.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

class D {
  extend base: Cpp.OverloadSet;

  fn MakePublic() -> D {
    //@dump-sem-ir-begin
    return {.base = OverloadSet(Cpp.PublicCall.PublicCall())};
    //@dump-sem-ir-end
  }

  fn MakeProtected() -> D {
    //@dump-sem-ir-begin
    return {.base = OverloadSet(Cpp.ProtectedCall.ProtectedCall())};
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
  // CHECK:STDERR: fail_import_overload_set_private.carbon:[[@LINE+4]]:3-60: error: cannot access private member `OverloadSet` of type `Cpp.OverloadSet` [ClassInvalidMemberAccess]
  // CHECK:STDERR: included from: fail_import_overload_set_private.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./overload_set.h:14:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.OverloadSet.OverloadSet(Cpp.PrivateCall.PrivateCall());
  // CHECK:STDERR: fail_import_overload_set_private.carbon:[[@LINE+4]]:3-57: error: cannot access private member `Overload` of type `Cpp.OverloadSet` [ClassInvalidMemberAccess]
  // CHECK:STDERR: included from: fail_import_overload_set_private.carbon:[[@LINE-9]]:10 [InCppInclude]
  // CHECK:STDERR: ./overload_set.h:15:15: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.OverloadSet.Overload(Cpp.PrivateCall.PrivateCall());
}

// --- fail_import_overload_set_private_base.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

class D {
  extend base: Cpp.OverloadSet;

  fn MakePrivate() -> D {
    // CHECK:STDERR: fail_import_overload_set_private_base.carbon:[[@LINE+4]]:21-62: error: cannot access private member `OverloadSet` of type `Cpp.OverloadSet` [ClassInvalidMemberAccess]
    // CHECK:STDERR: included from: fail_import_overload_set_private_base.carbon:[[@LINE-7]]:10 [InCppInclude]
    // CHECK:STDERR: ./overload_set.h:14:3: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    return {.base = OverloadSet(Cpp.PrivateCall.PrivateCall())};
  }

  fn CallPrivate() {
    // CHECK:STDERR: fail_import_overload_set_private_base.carbon:[[@LINE+4]]:5-43: error: cannot access private member `Overload` of type `Cpp.OverloadSet` [ClassInvalidMemberAccess]
    // CHECK:STDERR: included from: fail_import_overload_set_private_base.carbon:[[@LINE-15]]:10 [InCppInclude]
    // CHECK:STDERR: ./overload_set.h:15:15: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Overload(Cpp.PrivateCall.PrivateCall());
  }
}

// --- overload_set_base_class.h

#include "overload_set.h"

class OverloadPublic : public OverloadSet {};
class OverloadProtected : protected OverloadSet {};
class OverloadPrivate : private OverloadSet {};

// --- import_overload_set_public_base_class_call_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.OverloadPublic.Overload(Cpp.PublicCall.PublicCall());
  //@dump-sem-ir-end
}

// --- fail_import_overload_set_public_base_class_call_non_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

fn F() {
  // CHECK:STDERR: fail_import_overload_set_public_base_class_call_non_public.carbon:[[@LINE+5]]:3-64: error: cannot access protected member `Overload` of type `Cpp.OverloadPublic` [ClassInvalidMemberAccess]
  // CHECK:STDERR: included from: fail_import_overload_set_public_base_class_call_non_public.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: included from: ./overload_set_base_class.h:2:10 [InCppInclude]
  // CHECK:STDERR: ./overload_set.h:12:15: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.OverloadPublic.Overload(Cpp.ProtectedCall.ProtectedCall());
  // CHECK:STDERR: fail_import_overload_set_public_base_class_call_non_public.carbon:[[@LINE+5]]:3-60: error: cannot access private member `Overload` of type `Cpp.OverloadPublic` [ClassInvalidMemberAccess]
  // CHECK:STDERR: included from: fail_import_overload_set_public_base_class_call_non_public.carbon:[[@LINE-10]]:10 [InCppInclude]
  // CHECK:STDERR: included from: ./overload_set_base_class.h:2:10 [InCppInclude]
  // CHECK:STDERR: ./overload_set.h:15:15: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.OverloadPublic.Overload(Cpp.PrivateCall.PrivateCall());
}

// --- import_overload_set_public_base_class_derived_call_non_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

class Derived {
  extend base: Cpp.OverloadPublic;

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
  extend base: Cpp.OverloadPublic;

  fn F() {
    // CHECK:STDERR: fail_import_overload_set_public_base_class_derived_call_private.carbon:[[@LINE+5]]:5-43: error: cannot access private member `Overload` of type `Cpp.OverloadPublic` [ClassInvalidMemberAccess]
    // CHECK:STDERR: included from: fail_import_overload_set_public_base_class_derived_call_private.carbon:[[@LINE-7]]:10 [InCppInclude]
    // CHECK:STDERR: included from: ./overload_set_base_class.h:2:10 [InCppInclude]
    // CHECK:STDERR: ./overload_set.h:15:15: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Overload(Cpp.PrivateCall.PrivateCall());
  }
}

// --- fail_import_overload_set_protected_base_class.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

fn F() {
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon:[[@LINE+3]]:3-32: error: cannot access protected member `Overload` of type `Cpp.OverloadProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon:[[@LINE+2]]:3-32: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.OverloadProtected.Overload(Cpp.PublicCall.PublicCall());
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon:[[@LINE+3]]:3-32: error: cannot access protected member `Overload` of type `Cpp.OverloadProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon:[[@LINE-2]]:3-32: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.OverloadProtected.Overload(Cpp.ProtectedCall.ProtectedCall());
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon:[[@LINE+3]]:3-32: error: cannot access protected member `Overload` of type `Cpp.OverloadProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_import_overload_set_protected_base_class.carbon:[[@LINE-6]]:3-32: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.OverloadProtected.Overload(Cpp.PrivateCall.PrivateCall());
}

// --- import_overload_set_protected_base_class_derived_call_non_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

class Derived {
  extend base: Cpp.OverloadProtected;

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
  extend base: Cpp.OverloadProtected;

  fn F() {
    // CHECK:STDERR: fail_import_overload_set_protected_base_class_derived_call_private.carbon:[[@LINE+5]]:5-43: error: cannot access private member `Overload` of type `Cpp.OverloadProtected` [ClassInvalidMemberAccess]
    // CHECK:STDERR: included from: fail_import_overload_set_protected_base_class_derived_call_private.carbon:[[@LINE-7]]:10 [InCppInclude]
    // CHECK:STDERR: included from: ./overload_set_base_class.h:2:10 [InCppInclude]
    // CHECK:STDERR: ./overload_set.h:15:15: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Overload(Cpp.PrivateCall.PrivateCall());
  }
}

// --- fail_import_overload_set_private_base_class.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

fn F() {
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon:[[@LINE+3]]:3-30: error: cannot access private member `Overload` of type `Cpp.OverloadPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon:[[@LINE+2]]:3-30: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.OverloadPrivate.Overload(Cpp.PublicCall.PublicCall());
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon:[[@LINE+3]]:3-30: error: cannot access private member `Overload` of type `Cpp.OverloadPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon:[[@LINE-2]]:3-30: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.OverloadPrivate.Overload(Cpp.ProtectedCall.ProtectedCall());
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon:[[@LINE+3]]:3-30: error: cannot access private member `Overload` of type `Cpp.OverloadPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_import_overload_set_private_base_class.carbon:[[@LINE-6]]:3-30: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.OverloadPrivate.Overload(Cpp.PrivateCall.PrivateCall());
}

// --- fail_import_overload_set_private_base_class_derived.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set_base_class.h";

class Derived {
  extend base: Cpp.OverloadPrivate;

  fn F() {
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon:[[@LINE+3]]:5-12: error: cannot access private member `Overload` of type `Cpp.OverloadPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon:[[@LINE+2]]:5-12: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Overload(Cpp.PublicCall.PublicCall());
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon:[[@LINE+3]]:5-12: error: cannot access private member `Overload` of type `Cpp.OverloadPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon:[[@LINE-2]]:5-12: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Overload(Cpp.ProtectedCall.ProtectedCall());
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon:[[@LINE+3]]:5-12: error: cannot access private member `Overload` of type `Cpp.OverloadPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_import_overload_set_private_base_class_derived.carbon:[[@LINE-6]]:5-12: note: declared here [ClassMemberDeclaration]
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
    // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+3]]:5-28: error: cannot access private member `PrivateStatic` of type `Cpp.Public` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+2]]:5-28: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.Public.PrivateStatic();
    // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+3]]:5-22: error: cannot access private member `PrivateStatic` of type `Cpp.Public` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE-2]]:5-28: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.PrivateStatic();
  }

  fn CallInstance(var instance: Cpp.Public) {
    //@dump-sem-ir-begin
    // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+3]]:5-28: error: cannot access private member `PrivateInstance` of type `Cpp.Public` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+2]]:5-28: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.PrivateInstance();
    //@dump-sem-ir-end
  }
}

fn Call(var instance: Cpp.Public) {
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+3]]:3-28: error: cannot access protected member `ProtectedStatic` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+2]]:3-28: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Public.ProtectedStatic();
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+3]]:3-25: error: cannot access protected member `ProtectedStatic` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE-2]]:3-28: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.ProtectedStatic();

  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+3]]:3-26: error: cannot access private member `PrivateStatic` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE-28]]:5-28: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Public.PrivateStatic();
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+3]]:3-23: error: cannot access private member `PrivateStatic` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE-32]]:5-28: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PrivateStatic();

  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+3]]:3-28: error: cannot access protected member `ProtectedInstance` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+2]]:3-28: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.ProtectedInstance();
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE+3]]:3-26: error: cannot access private member `PrivateInstance` of type `Cpp.Public` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_access_denied.carbon:[[@LINE-29]]:5-28: note: declared here [ClassMemberDeclaration]
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
    // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+3]]:5-31: error: cannot access private member `PrivateStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+2]]:5-31: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.Protected.PrivateStatic();
    // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+3]]:5-22: error: cannot access private member `PrivateStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE-2]]:5-31: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.PrivateStatic();
  }

  fn CallInstance(var instance: Cpp.Protected) {
    // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+3]]:5-28: error: cannot access private member `PrivateInstance` of type `Cpp.Protected` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+2]]:5-28: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.PrivateInstance();
  }
}

fn Call(var instance: Cpp.Protected) {
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+3]]:3-28: error: cannot access protected member `PublicStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+2]]:3-28: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Protected.PublicStatic();
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+3]]:3-22: error: cannot access protected member `PublicStatic` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE-2]]:3-28: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PublicStatic();

  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+3]]:3-31: error: cannot access protected member `ProtectedStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+2]]:3-31: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Protected.ProtectedStatic();
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+3]]:3-25: error: cannot access protected member `ProtectedStatic` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE-2]]:3-31: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.ProtectedStatic();

  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+3]]:3-29: error: cannot access private member `PrivateStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE-35]]:5-31: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Protected.PrivateStatic();
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+3]]:3-23: error: cannot access private member `PrivateStatic` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE-39]]:5-31: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PrivateStatic();

  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+3]]:3-25: error: cannot access protected member `PublicInstance` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+2]]:3-25: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.PublicInstance();
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+3]]:3-28: error: cannot access protected member `ProtectedInstance` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+2]]:3-28: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.ProtectedInstance();
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE+3]]:3-26: error: cannot access private member `PrivateInstance` of type `Cpp.Protected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_protected_access_denied.carbon:[[@LINE-41]]:5-28: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.PrivateInstance();
}

// --- fail_base_class_private_access_denied.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

class Derived {
  extend base: Cpp.Private;

  fn CallStatic() {
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:5-28: error: cannot access private member `PublicStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+2]]:5-28: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.Private.PublicStatic();
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:5-21: error: cannot access private member `PublicStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-2]]:5-28: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.PublicStatic();

    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:5-31: error: cannot access private member `ProtectedStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+2]]:5-31: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.Private.ProtectedStatic();
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:5-24: error: cannot access private member `ProtectedStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-2]]:5-31: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.ProtectedStatic();

    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:5-29: error: cannot access private member `PrivateStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+2]]:5-29: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.Private.PrivateStatic();
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:5-22: error: cannot access private member `PrivateStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-2]]:5-29: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.PrivateStatic();
  }

  fn CallInstance(var instance: Cpp.Private) {
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:5-27: error: cannot access private member `PublicInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+2]]:5-27: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.PublicInstance();

    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:5-30: error: cannot access private member `ProtectedInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+2]]:5-30: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.ProtectedInstance();

    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:5-28: error: cannot access private member `PrivateInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+2]]:5-28: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.PrivateInstance();
  }
}

fn Call(var instance: Cpp.Private) {
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:3-26: error: cannot access private member `PublicStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-45]]:5-28: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Private.PublicStatic();
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:3-22: error: cannot access private member `PublicStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-49]]:5-28: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PublicStatic();

  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:3-29: error: cannot access private member `ProtectedStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-45]]:5-31: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Private.ProtectedStatic();
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:3-25: error: cannot access private member `ProtectedStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-49]]:5-31: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.ProtectedStatic();

  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:3-27: error: cannot access private member `PrivateStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-45]]:5-29: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.Private.PrivateStatic();
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:3-23: error: cannot access private member `PrivateStatic` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-49]]:5-29: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PrivateStatic();

  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:3-25: error: cannot access private member `PublicInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-43]]:5-27: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.PublicInstance();
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:3-28: error: cannot access private member `ProtectedInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-42]]:5-30: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.ProtectedInstance();
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE+3]]:3-26: error: cannot access private member `PrivateInstance` of type `Cpp.Private` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_private_access_denied.carbon:[[@LINE-41]]:5-28: note: declared here [ClassMemberDeclaration]
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
    // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+3]]:5-37: error: cannot access private member `PrivateStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+2]]:5-37: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.PublicProtected.PrivateStatic();
    // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+3]]:5-22: error: cannot access private member `PrivateStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE-2]]:5-37: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.PrivateStatic();
  }

  fn CallInstance(var instance: Cpp.PublicProtected) {
    // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+3]]:5-28: error: cannot access private member `PrivateInstance` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+2]]:5-28: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.PrivateInstance();
  }
}

fn Call(var instance: Cpp.PublicProtected) {
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+3]]:3-34: error: cannot access protected member `PublicStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+2]]:3-34: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.PublicProtected.PublicStatic();
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+3]]:3-22: error: cannot access protected member `PublicStatic` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE-2]]:3-34: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PublicStatic();

  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+3]]:3-37: error: cannot access protected member `ProtectedStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+2]]:3-37: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.PublicProtected.ProtectedStatic();
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+3]]:3-25: error: cannot access protected member `ProtectedStatic` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE-2]]:3-37: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.ProtectedStatic();

  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+3]]:3-35: error: cannot access private member `PrivateStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE-35]]:5-37: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.PublicProtected.PrivateStatic();
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+3]]:3-23: error: cannot access private member `PrivateStatic` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE-39]]:5-37: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PrivateStatic();

  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+3]]:3-25: error: cannot access protected member `PublicInstance` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+2]]:3-25: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.PublicInstance();
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+3]]:3-28: error: cannot access protected member `ProtectedInstance` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+2]]:3-28: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.ProtectedInstance();
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE+3]]:3-26: error: cannot access private member `PrivateInstance` of type `Cpp.PublicProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_protected_access_denied.carbon:[[@LINE-41]]:5-28: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.PrivateInstance();
}

// --- fail_base_class_public_private_access_denied.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

class Derived {
  extend base: Cpp.PublicPrivate;

  fn CallStatic() {
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:5-34: error: cannot access private member `PublicStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+2]]:5-34: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.PublicPrivate.PublicStatic();
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:5-21: error: cannot access private member `PublicStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-2]]:5-34: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.PublicStatic();

    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:5-37: error: cannot access private member `ProtectedStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+2]]:5-37: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.PublicPrivate.ProtectedStatic();
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:5-24: error: cannot access private member `ProtectedStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-2]]:5-37: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.ProtectedStatic();

    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:5-35: error: cannot access private member `PrivateStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+2]]:5-35: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Cpp.PublicPrivate.PrivateStatic();
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:5-22: error: cannot access private member `PrivateStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-2]]:5-35: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    Self.PrivateStatic();
  }

  fn CallInstance(var instance: Cpp.PublicPrivate) {
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:5-27: error: cannot access private member `PublicInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+2]]:5-27: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.PublicInstance();

    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:5-30: error: cannot access private member `ProtectedInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+2]]:5-30: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.ProtectedInstance();

    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:5-28: error: cannot access private member `PrivateInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
    // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+2]]:5-28: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    instance.PrivateInstance();
  }
}

fn Call(var instance: Cpp.PublicPrivate) {
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:3-32: error: cannot access private member `PublicStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-45]]:5-34: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.PublicPrivate.PublicStatic();
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:3-22: error: cannot access private member `PublicStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-49]]:5-34: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PublicStatic();

  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:3-35: error: cannot access private member `ProtectedStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-45]]:5-37: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.PublicPrivate.ProtectedStatic();
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:3-25: error: cannot access private member `ProtectedStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-49]]:5-37: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.ProtectedStatic();

  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:3-33: error: cannot access private member `PrivateStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-45]]:5-35: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.PublicPrivate.PrivateStatic();
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:3-23: error: cannot access private member `PrivateStatic` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-49]]:5-35: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Derived.PrivateStatic();

  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:3-25: error: cannot access private member `PublicInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-43]]:5-27: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.PublicInstance();
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:3-28: error: cannot access private member `ProtectedInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-42]]:5-30: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.ProtectedInstance();
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE+3]]:3-26: error: cannot access private member `PrivateInstance` of type `Cpp.PublicPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR: fail_base_class_public_private_access_denied.carbon:[[@LINE-41]]:5-28: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  instance.PrivateInstance();
}

// CHECK:STDOUT: --- import_non_function_member_public.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonFunctionMemberPublic: type = class_type @NonFunctionMemberPublic [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %NonFunctionMemberPublic.elem: type = unbound_element_type %NonFunctionMemberPublic, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %instance_data.patt: %pattern_type.6b6 = value_binding_pattern instance_data [concrete]
// CHECK:STDOUT:   %static_data.patt.405: %pattern_type.6b6 = value_binding_pattern static_data [concrete]
// CHECK:STDOUT:   %static_data.patt.617: %pattern_type.6b6 = ref_binding_pattern static_data [concrete]
// CHECK:STDOUT:   %static_data.var_patt: %pattern_type.6b6 = var_pattern %static_data.patt.617 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonFunctionMemberPublic = %NonFunctionMemberPublic.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonFunctionMemberPublic.decl: type = class_decl @NonFunctionMemberPublic [concrete = constants.%NonFunctionMemberPublic] {} {}
// CHECK:STDOUT:   %static_data.patt: %pattern_type.6b6 = ref_binding_pattern static_data [concrete = constants.%static_data.patt.617]
// CHECK:STDOUT:   %static_data.var_patt: %pattern_type.6b6 = var_pattern %static_data.patt [concrete = constants.%static_data.var_patt]
// CHECK:STDOUT:   %static_data.var: ref %i32 = var_storage %static_data.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %NonFunctionMemberPublic) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %NonFunctionMemberPublic = name_ref s, %s
// CHECK:STDOUT:   %instance_data.ref: %NonFunctionMemberPublic.elem = name_ref instance_data, @NonFunctionMemberPublic.%.1 [concrete = @NonFunctionMemberPublic.%.1]
// CHECK:STDOUT:   %.loc8_36.1: ref %i32 = class_element_access %s.ref, element0
// CHECK:STDOUT:   %.loc8_36.2: %i32 = acquire_value %.loc8_36.1
// CHECK:STDOUT:   %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %instance_data: %i32 = wrapper_binding instance_data, %.loc8_36.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %instance_data.patt: %pattern_type.6b6 = value_binding_pattern instance_data [concrete = constants.%instance_data.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %NonFunctionMemberPublic.ref.loc9: type = name_ref NonFunctionMemberPublic, imports.%NonFunctionMemberPublic.decl [concrete = constants.%NonFunctionMemberPublic]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %static_data.ref: ref %i32 = name_ref static_data, imports.%static_data.var [concrete = imports.%static_data.var]
// CHECK:STDOUT:   %.loc9: %i32 = acquire_value %static_data.ref
// CHECK:STDOUT:   %i32.loc9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %static_data: %i32 = wrapper_binding static_data, %.loc9
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %static_data.patt: %pattern_type.6b6 = value_binding_pattern static_data [concrete = constants.%static_data.patt.405]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_function_member_public_extend.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %NonFunctionMemberPublic: type = class_type @NonFunctionMemberPublic [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %NonFunctionMemberPublic.elem: type = unbound_element_type %NonFunctionMemberPublic, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %instance_data.patt: %pattern_type.6b6 = value_binding_pattern instance_data [concrete]
// CHECK:STDOUT:   %static_data.patt.405: %pattern_type.6b6 = value_binding_pattern static_data [concrete]
// CHECK:STDOUT:   %static_data.patt.617: %pattern_type.6b6 = ref_binding_pattern static_data [concrete]
// CHECK:STDOUT:   %static_data.var_patt: %pattern_type.6b6 = var_pattern %static_data.patt.617 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %static_data.patt: %pattern_type.6b6 = ref_binding_pattern static_data [concrete = constants.%static_data.patt.617]
// CHECK:STDOUT:   %static_data.var_patt: %pattern_type.6b6 = var_pattern %static_data.patt [concrete = constants.%static_data.var_patt]
// CHECK:STDOUT:   %static_data.var: ref %i32 = var_storage %static_data.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%d.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %instance_data.ref: %NonFunctionMemberPublic.elem = name_ref instance_data, @NonFunctionMemberPublic.%.1 [concrete = @NonFunctionMemberPublic.%.1]
// CHECK:STDOUT:   %.loc12_36.1: %NonFunctionMemberPublic = as_compatible %d.ref
// CHECK:STDOUT:   %.loc12_36.2: ref %NonFunctionMemberPublic = class_element_access %.loc12_36.1, element0
// CHECK:STDOUT:   %.loc12_36.3: ref %NonFunctionMemberPublic = converted %d.ref, %.loc12_36.2
// CHECK:STDOUT:   %.loc12_36.4: ref %i32 = class_element_access %.loc12_36.3, element0
// CHECK:STDOUT:   %.loc12_36.5: %i32 = acquire_value %.loc12_36.4
// CHECK:STDOUT:   %i32.loc12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %instance_data: %i32 = wrapper_binding instance_data, %.loc12_36.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %instance_data.patt: %pattern_type.6b6 = value_binding_pattern instance_data [concrete = constants.%instance_data.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.ref.loc13: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %static_data.ref: ref %i32 = name_ref static_data, imports.%static_data.var [concrete = imports.%static_data.var]
// CHECK:STDOUT:   %.loc13: %i32 = acquire_value %static_data.ref
// CHECK:STDOUT:   %i32.loc13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %static_data: %i32 = wrapper_binding static_data, %.loc13
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %static_data.patt: %pattern_type.6b6 = value_binding_pattern static_data [concrete = constants.%static_data.patt.405]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_non_function_member_protected_extend_call_in_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %NonFunctionMemberProtected: type = class_type @NonFunctionMemberProtected [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %NonFunctionMemberProtected.elem: type = unbound_element_type %NonFunctionMemberProtected, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %instance_data.patt: %pattern_type.6b6 = value_binding_pattern instance_data [concrete]
// CHECK:STDOUT:   %unqualified_static_data.patt: %pattern_type.6b6 = value_binding_pattern unqualified_static_data [concrete]
// CHECK:STDOUT:   %static_data.patt: %pattern_type.6b6 = ref_binding_pattern static_data [concrete]
// CHECK:STDOUT:   %static_data.var_patt: %pattern_type.6b6 = var_pattern %static_data.patt [concrete]
// CHECK:STDOUT:   %derived_static_data.patt: %pattern_type.6b6 = value_binding_pattern derived_static_data [concrete]
// CHECK:STDOUT:   %base_static_data.patt: %pattern_type.6b6 = value_binding_pattern base_static_data [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NonFunctionMemberProtected = %NonFunctionMemberProtected.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonFunctionMemberProtected.decl: type = class_decl @NonFunctionMemberProtected [concrete = constants.%NonFunctionMemberProtected] {} {}
// CHECK:STDOUT:   %static_data.patt: %pattern_type.6b6 = ref_binding_pattern static_data [concrete = constants.%static_data.patt]
// CHECK:STDOUT:   %static_data.var_patt: %pattern_type.6b6 = var_pattern %static_data.patt [concrete = constants.%static_data.var_patt]
// CHECK:STDOUT:   %static_data.var: ref %i32 = var_storage %static_data.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   .instance_data = <poisoned>
// CHECK:STDOUT:   .static_data = <poisoned>
// CHECK:STDOUT:   .Derived = <poisoned>
// CHECK:STDOUT:   extend %NonFunctionMemberProtected.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.F(%self.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Derived = name_ref self, %self
// CHECK:STDOUT:   %instance_data.ref: %NonFunctionMemberProtected.elem = name_ref instance_data, @NonFunctionMemberProtected.%.1 [concrete = @NonFunctionMemberProtected.%.1]
// CHECK:STDOUT:   %.loc10_41.1: %NonFunctionMemberProtected = as_compatible %self.ref
// CHECK:STDOUT:   %.loc10_41.2: ref %NonFunctionMemberProtected = class_element_access %.loc10_41.1, element0
// CHECK:STDOUT:   %.loc10_41.3: ref %NonFunctionMemberProtected = converted %self.ref, %.loc10_41.2
// CHECK:STDOUT:   %.loc10_41.4: ref %i32 = class_element_access %.loc10_41.3, element0
// CHECK:STDOUT:   %.loc10_41.5: %i32 = acquire_value %.loc10_41.4
// CHECK:STDOUT:   %i32.loc10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %instance_data: %i32 = wrapper_binding instance_data, %.loc10_41.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %instance_data.patt: %pattern_type.6b6 = value_binding_pattern instance_data [concrete = constants.%instance_data.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %static_data.ref.loc11: ref %i32 = name_ref static_data, imports.%static_data.var [concrete = imports.%static_data.var]
// CHECK:STDOUT:   %.loc11: %i32 = acquire_value %static_data.ref.loc11
// CHECK:STDOUT:   %i32.loc11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %unqualified_static_data: %i32 = wrapper_binding unqualified_static_data, %.loc11
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %unqualified_static_data.patt: %pattern_type.6b6 = value_binding_pattern unqualified_static_data [concrete = constants.%unqualified_static_data.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %static_data.ref.loc12: ref %i32 = name_ref static_data, imports.%static_data.var [concrete = imports.%static_data.var]
// CHECK:STDOUT:   %.loc12: %i32 = acquire_value %static_data.ref.loc12
// CHECK:STDOUT:   %i32.loc12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %derived_static_data: %i32 = wrapper_binding derived_static_data, %.loc12
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %derived_static_data.patt: %pattern_type.6b6 = value_binding_pattern derived_static_data [concrete = constants.%derived_static_data.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %NonFunctionMemberProtected.ref: type = name_ref NonFunctionMemberProtected, imports.%NonFunctionMemberProtected.decl [concrete = constants.%NonFunctionMemberProtected]
// CHECK:STDOUT:   %static_data.ref.loc13: ref %i32 = name_ref static_data, imports.%static_data.var [concrete = imports.%static_data.var]
// CHECK:STDOUT:   %.loc13: %i32 = acquire_value %static_data.ref.loc13
// CHECK:STDOUT:   %i32.loc13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %base_static_data: %i32 = wrapper_binding base_static_data, %.loc13
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %base_static_data.patt: %pattern_type.6b6 = value_binding_pattern base_static_data [concrete = constants.%base_static_data.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_function_member_public.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %FunctionMemberPublic: type = class_type @FunctionMemberPublic [concrete]
// CHECK:STDOUT:   %ptr.9a9: type = ptr_type %FunctionMemberPublic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %FunctionMemberPublic.instance_fn.cpp_overload_set.type: type = cpp_overload_set_type @FunctionMemberPublic.instance_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %FunctionMemberPublic.instance_fn.cpp_overload_set.value: %FunctionMemberPublic.instance_fn.cpp_overload_set.type = cpp_overload_set_value @FunctionMemberPublic.instance_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %FunctionMemberPublic.instance_fn.type: type = fn_type @FunctionMemberPublic.instance_fn [concrete]
// CHECK:STDOUT:   %FunctionMemberPublic.instance_fn: %FunctionMemberPublic.instance_fn.type = struct_value () [concrete]
// CHECK:STDOUT:   %FunctionMemberPublic.static_fn.cpp_overload_set.type: type = cpp_overload_set_type @FunctionMemberPublic.static_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %FunctionMemberPublic.static_fn.cpp_overload_set.value: %FunctionMemberPublic.static_fn.cpp_overload_set.type = cpp_overload_set_value @FunctionMemberPublic.static_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %FunctionMemberPublic.static_fn.type: type = fn_type @FunctionMemberPublic.static_fn [concrete]
// CHECK:STDOUT:   %FunctionMemberPublic.static_fn: %FunctionMemberPublic.static_fn.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .FunctionMemberPublic = %FunctionMemberPublic.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FunctionMemberPublic.decl: type = class_decl @FunctionMemberPublic [concrete = constants.%FunctionMemberPublic] {} {}
// CHECK:STDOUT:   %FunctionMemberPublic.instance_fn.cpp_overload_set.value: %FunctionMemberPublic.instance_fn.cpp_overload_set.type = cpp_overload_set_value @FunctionMemberPublic.instance_fn.cpp_overload_set [concrete = constants.%FunctionMemberPublic.instance_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %FunctionMemberPublic.instance_fn.decl: %FunctionMemberPublic.instance_fn.type = fn_decl @FunctionMemberPublic.instance_fn [concrete = constants.%FunctionMemberPublic.instance_fn] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FunctionMemberPublic.static_fn.cpp_overload_set.value: %FunctionMemberPublic.static_fn.cpp_overload_set.type = cpp_overload_set_value @FunctionMemberPublic.static_fn.cpp_overload_set [concrete = constants.%FunctionMemberPublic.static_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %FunctionMemberPublic.static_fn.decl: %FunctionMemberPublic.static_fn.type = fn_decl @FunctionMemberPublic.static_fn [concrete = constants.%FunctionMemberPublic.static_fn] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %ptr.9a9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %ptr.9a9 = name_ref s, %s
// CHECK:STDOUT:   %.loc8: ref %FunctionMemberPublic = deref %s.ref
// CHECK:STDOUT:   %instance_fn.ref: %FunctionMemberPublic.instance_fn.cpp_overload_set.type = name_ref instance_fn, imports.%FunctionMemberPublic.instance_fn.cpp_overload_set.value [concrete = constants.%FunctionMemberPublic.instance_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8, %instance_fn.ref
// CHECK:STDOUT:   %FunctionMemberPublic.instance_fn.call: init %empty_tuple.type = call imports.%FunctionMemberPublic.instance_fn.decl(%.loc8)
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %FunctionMemberPublic.ref.loc9: type = name_ref FunctionMemberPublic, imports.%FunctionMemberPublic.decl [concrete = constants.%FunctionMemberPublic]
// CHECK:STDOUT:   %static_fn.ref: %FunctionMemberPublic.static_fn.cpp_overload_set.type = name_ref static_fn, imports.%FunctionMemberPublic.static_fn.cpp_overload_set.value [concrete = constants.%FunctionMemberPublic.static_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %FunctionMemberPublic.static_fn.call: init %empty_tuple.type = call imports.%FunctionMemberPublic.static_fn.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_function_member_protected_extend_call_instance_in_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %FunctionMemberProtected.instance_fn.cpp_overload_set.type: type = cpp_overload_set_type @FunctionMemberProtected.instance_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %FunctionMemberProtected.instance_fn.cpp_overload_set.value: %FunctionMemberProtected.instance_fn.cpp_overload_set.type = cpp_overload_set_value @FunctionMemberProtected.instance_fn.cpp_overload_set [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %FunctionMemberProtected.instance_fn.cpp_overload_set.value: %FunctionMemberProtected.instance_fn.cpp_overload_set.type = cpp_overload_set_value @FunctionMemberProtected.instance_fn.cpp_overload_set [concrete = constants.%FunctionMemberProtected.instance_fn.cpp_overload_set.value]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   .instance_fn = <poisoned>
// CHECK:STDOUT:   extend %FunctionMemberProtected.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.F(%self.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Derived = name_ref self, %self
// CHECK:STDOUT:   %instance_fn.ref: %FunctionMemberProtected.instance_fn.cpp_overload_set.type = name_ref instance_fn, imports.%FunctionMemberProtected.instance_fn.cpp_overload_set.value [concrete = constants.%FunctionMemberProtected.instance_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %self.ref, %instance_fn.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_function_member_protected_extend_call_static_in_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %FunctionMemberProtected: type = class_type @FunctionMemberProtected [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %FunctionMemberProtected.static_fn.cpp_overload_set.type: type = cpp_overload_set_type @FunctionMemberProtected.static_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %FunctionMemberProtected.static_fn.cpp_overload_set.value: %FunctionMemberProtected.static_fn.cpp_overload_set.type = cpp_overload_set_value @FunctionMemberProtected.static_fn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %FunctionMemberProtected.static_fn.type: type = fn_type @FunctionMemberProtected.static_fn [concrete]
// CHECK:STDOUT:   %FunctionMemberProtected.static_fn: %FunctionMemberProtected.static_fn.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .FunctionMemberProtected = %FunctionMemberProtected.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %FunctionMemberProtected.decl: type = class_decl @FunctionMemberProtected [concrete = constants.%FunctionMemberProtected] {} {}
// CHECK:STDOUT:   %FunctionMemberProtected.static_fn.cpp_overload_set.value: %FunctionMemberProtected.static_fn.cpp_overload_set.type = cpp_overload_set_value @FunctionMemberProtected.static_fn.cpp_overload_set [concrete = constants.%FunctionMemberProtected.static_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %FunctionMemberProtected.static_fn.decl: %FunctionMemberProtected.static_fn.type = fn_decl @FunctionMemberProtected.static_fn [concrete = constants.%FunctionMemberProtected.static_fn] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   .static_fn = <poisoned>
// CHECK:STDOUT:   .Derived = <poisoned>
// CHECK:STDOUT:   extend %FunctionMemberProtected.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.F(%self.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %static_fn.ref.loc10: %FunctionMemberProtected.static_fn.cpp_overload_set.type = name_ref static_fn, imports.%FunctionMemberProtected.static_fn.cpp_overload_set.value [concrete = constants.%FunctionMemberProtected.static_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %FunctionMemberProtected.static_fn.call.loc10: init %empty_tuple.type = call imports.%FunctionMemberProtected.static_fn.decl()
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %static_fn.ref.loc11: %FunctionMemberProtected.static_fn.cpp_overload_set.type = name_ref static_fn, imports.%FunctionMemberProtected.static_fn.cpp_overload_set.value [concrete = constants.%FunctionMemberProtected.static_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %FunctionMemberProtected.static_fn.call.loc11: init %empty_tuple.type = call imports.%FunctionMemberProtected.static_fn.decl()
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %FunctionMemberProtected.ref: type = name_ref FunctionMemberProtected, imports.%FunctionMemberProtected.decl [concrete = constants.%FunctionMemberProtected]
// CHECK:STDOUT:   %static_fn.ref.loc12: %FunctionMemberProtected.static_fn.cpp_overload_set.type = name_ref static_fn, imports.%FunctionMemberProtected.static_fn.cpp_overload_set.value [concrete = constants.%FunctionMemberProtected.static_fn.cpp_overload_set.value]
// CHECK:STDOUT:   %FunctionMemberProtected.static_fn.call.loc12: init %empty_tuple.type = call imports.%FunctionMemberProtected.static_fn.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overload_set_public.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %OverloadSet: type = class_type @OverloadSet [concrete]
// CHECK:STDOUT:   %OverloadSet.OverloadSet.cpp_overload_set.type: type = cpp_overload_set_type @OverloadSet.OverloadSet.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OverloadSet.OverloadSet.cpp_overload_set.value: %OverloadSet.OverloadSet.cpp_overload_set.type = cpp_overload_set_value @OverloadSet.OverloadSet.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall: type = class_type @PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.type: type = cpp_overload_set_type @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d0a: type = ptr_type %PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.type: type = fn_type @PublicCall__carbon_thunk [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk: %PublicCall__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.4a0: type = ptr_type %OverloadSet [concrete]
// CHECK:STDOUT:   %OverloadSet__carbon_thunk.type: type = fn_type @OverloadSet__carbon_thunk [concrete]
// CHECK:STDOUT:   %OverloadSet__carbon_thunk: %OverloadSet__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %OverloadSet.cpp_destructor.type: type = fn_type @OverloadSet.cpp_destructor [concrete]
// CHECK:STDOUT:   %OverloadSet.cpp_destructor: %OverloadSet.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor.type: type = fn_type @PublicCall.cpp_destructor [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor: %PublicCall.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %OverloadSet.Overload.cpp_overload_set.type: type = cpp_overload_set_type @OverloadSet.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OverloadSet.Overload.cpp_overload_set.value: %OverloadSet.Overload.cpp_overload_set.type = cpp_overload_set_value @OverloadSet.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.type: type = fn_type @Overload__carbon_thunk [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk: %Overload__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OverloadSet = %OverloadSet.decl
// CHECK:STDOUT:     .PublicCall = %PublicCall.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OverloadSet.decl: type = class_decl @OverloadSet [concrete = constants.%OverloadSet] {} {}
// CHECK:STDOUT:   %OverloadSet.OverloadSet.cpp_overload_set.value: %OverloadSet.OverloadSet.cpp_overload_set.type = cpp_overload_set_value @OverloadSet.OverloadSet.cpp_overload_set [concrete = constants.%OverloadSet.OverloadSet.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall.decl: type = class_decl @PublicCall [concrete = constants.%PublicCall] {} {}
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.decl: %PublicCall__carbon_thunk.type = fn_decl @PublicCall__carbon_thunk [concrete = constants.%PublicCall__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OverloadSet__carbon_thunk.decl: %OverloadSet__carbon_thunk.type = fn_decl @OverloadSet__carbon_thunk [concrete = constants.%OverloadSet__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OverloadSet.Overload.cpp_overload_set.value: %OverloadSet.Overload.cpp_overload_set.type = cpp_overload_set_value @OverloadSet.Overload.cpp_overload_set [concrete = constants.%OverloadSet.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Overload__carbon_thunk.decl: %Overload__carbon_thunk.type = fn_decl @Overload__carbon_thunk [concrete = constants.%Overload__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %OverloadSet.ref.loc8_6: type = name_ref OverloadSet, imports.%OverloadSet.decl [concrete = constants.%OverloadSet]
// CHECK:STDOUT:   %OverloadSet.ref.loc8_18: %OverloadSet.OverloadSet.cpp_overload_set.type = name_ref OverloadSet, imports.%OverloadSet.OverloadSet.cpp_overload_set.value [concrete = constants.%OverloadSet.OverloadSet.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc8_31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicCall.ref.loc8_34: type = name_ref PublicCall, imports.%PublicCall.decl [concrete = constants.%PublicCall]
// CHECK:STDOUT:   %PublicCall.ref.loc8_45: %PublicCall.PublicCall.cpp_overload_set.type = name_ref PublicCall, imports.%PublicCall.PublicCall.cpp_overload_set.value [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_57.1: ref %PublicCall = temporary_storage
// CHECK:STDOUT:   %addr.loc8_57: %ptr.d0a = addr_of %.loc8_57.1
// CHECK:STDOUT:   %PublicCall__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%PublicCall__carbon_thunk.decl(%addr.loc8_57)
// CHECK:STDOUT:   %.loc8_57.2: init %PublicCall to %.loc8_57.1 = mark_in_place_init %PublicCall__carbon_thunk.call.loc8
// CHECK:STDOUT:   %.loc8_58.1: ref %OverloadSet = temporary_storage
// CHECK:STDOUT:   %.loc8_57.3: ref %PublicCall = temporary %.loc8_57.1, %.loc8_57.2
// CHECK:STDOUT:   %.loc8_57.4: %PublicCall = acquire_value %.loc8_57.3
// CHECK:STDOUT:   %.loc8_57.5: ref %PublicCall = value_as_ref %.loc8_57.4
// CHECK:STDOUT:   %addr.loc8_58.1: %ptr.d0a = addr_of %.loc8_57.5
// CHECK:STDOUT:   %addr.loc8_58.2: %ptr.4a0 = addr_of %.loc8_58.1
// CHECK:STDOUT:   %OverloadSet__carbon_thunk.call: init %empty_tuple.type = call imports.%OverloadSet__carbon_thunk.decl(%addr.loc8_58.1, %addr.loc8_58.2)
// CHECK:STDOUT:   %.loc8_58.2: init %OverloadSet to %.loc8_58.1 = mark_in_place_init %OverloadSet__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_58.3: ref %OverloadSet = temporary %.loc8_58.1, %.loc8_58.2
// CHECK:STDOUT:   %OverloadSet.cpp_destructor.bound: <bound method> = bound_method %.loc8_58.3, constants.%OverloadSet.cpp_destructor
// CHECK:STDOUT:   %OverloadSet.cpp_destructor.call: init %empty_tuple.type = call %OverloadSet.cpp_destructor.bound(%.loc8_58.3)
// CHECK:STDOUT:   %PublicCall.cpp_destructor.bound.loc8: <bound method> = bound_method %.loc8_57.3, constants.%PublicCall.cpp_destructor
// CHECK:STDOUT:   %PublicCall.cpp_destructor.call.loc8: init %empty_tuple.type = call %PublicCall.cpp_destructor.bound.loc8(%.loc8_57.3)
// CHECK:STDOUT:   %Cpp.ref.loc9_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %OverloadSet.ref.loc9: type = name_ref OverloadSet, imports.%OverloadSet.decl [concrete = constants.%OverloadSet]
// CHECK:STDOUT:   %Overload.ref: %OverloadSet.Overload.cpp_overload_set.type = name_ref Overload, imports.%OverloadSet.Overload.cpp_overload_set.value [concrete = constants.%OverloadSet.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc9_28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicCall.ref.loc9_31: type = name_ref PublicCall, imports.%PublicCall.decl [concrete = constants.%PublicCall]
// CHECK:STDOUT:   %PublicCall.ref.loc9_42: %PublicCall.PublicCall.cpp_overload_set.type = name_ref PublicCall, imports.%PublicCall.PublicCall.cpp_overload_set.value [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc9_54.1: ref %PublicCall = temporary_storage
// CHECK:STDOUT:   %addr.loc9_54: %ptr.d0a = addr_of %.loc9_54.1
// CHECK:STDOUT:   %PublicCall__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%PublicCall__carbon_thunk.decl(%addr.loc9_54)
// CHECK:STDOUT:   %.loc9_54.2: init %PublicCall to %.loc9_54.1 = mark_in_place_init %PublicCall__carbon_thunk.call.loc9
// CHECK:STDOUT:   %.loc9_54.3: ref %PublicCall = temporary %.loc9_54.1, %.loc9_54.2
// CHECK:STDOUT:   %.loc9_54.4: %PublicCall = acquire_value %.loc9_54.3
// CHECK:STDOUT:   %.loc9_54.5: ref %PublicCall = value_as_ref %.loc9_54.4
// CHECK:STDOUT:   %addr.loc9_55: %ptr.d0a = addr_of %.loc9_54.5
// CHECK:STDOUT:   %Overload__carbon_thunk.call: init %empty_tuple.type = call imports.%Overload__carbon_thunk.decl(%addr.loc9_55)
// CHECK:STDOUT:   %PublicCall.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_54.3, constants.%PublicCall.cpp_destructor
// CHECK:STDOUT:   %PublicCall.cpp_destructor.call.loc9: init %empty_tuple.type = call %PublicCall.cpp_destructor.bound.loc9(%.loc9_54.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overload_set_protected_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %OverloadSet: type = class_type @OverloadSet [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.base.d55: type = struct_type {.base: %OverloadSet} [concrete]
// CHECK:STDOUT:   %OverloadSet.OverloadSet.cpp_overload_set.type: type = cpp_overload_set_type @OverloadSet.OverloadSet.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OverloadSet.OverloadSet.cpp_overload_set.value: %OverloadSet.OverloadSet.cpp_overload_set.type = cpp_overload_set_value @OverloadSet.OverloadSet.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall: type = class_type @PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.type: type = cpp_overload_set_type @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d0a: type = ptr_type %PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.type: type = fn_type @PublicCall__carbon_thunk [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk: %PublicCall__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.4a0: type = ptr_type %OverloadSet [concrete]
// CHECK:STDOUT:   %OverloadSet__carbon_thunk.type.c7f6eb.1: type = fn_type @OverloadSet__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %OverloadSet__carbon_thunk.e50482.1: %OverloadSet__carbon_thunk.type.c7f6eb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %.244: type = partial_type %OverloadSet [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor.type: type = fn_type @PublicCall.cpp_destructor [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor: %PublicCall.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ProtectedCall: type = class_type @ProtectedCall [concrete]
// CHECK:STDOUT:   %ProtectedCall.ProtectedCall.cpp_overload_set.type: type = cpp_overload_set_type @ProtectedCall.ProtectedCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ProtectedCall.ProtectedCall.cpp_overload_set.value: %ProtectedCall.ProtectedCall.cpp_overload_set.type = cpp_overload_set_value @ProtectedCall.ProtectedCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.bc1: type = ptr_type %ProtectedCall [concrete]
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.type: type = fn_type @ProtectedCall__carbon_thunk [concrete]
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk: %ProtectedCall__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %OverloadSet__carbon_thunk.type.c7f6eb.2: type = fn_type @OverloadSet__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %OverloadSet__carbon_thunk.e50482.2: %OverloadSet__carbon_thunk.type.c7f6eb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.type: type = fn_type @ProtectedCall.cpp_destructor [concrete]
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor: %ProtectedCall.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %OverloadSet.Overload.cpp_overload_set.type: type = cpp_overload_set_type @OverloadSet.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OverloadSet.Overload.cpp_overload_set.value: %OverloadSet.Overload.cpp_overload_set.type = cpp_overload_set_value @OverloadSet.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.type.3dd392.1: type = fn_type @Overload__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.1c804c.1: %Overload__carbon_thunk.type.3dd392.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.type.3dd392.2: type = fn_type @Overload__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.1c804c.2: %Overload__carbon_thunk.type.3dd392.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OverloadSet = %OverloadSet.decl
// CHECK:STDOUT:     .PublicCall = %PublicCall.decl
// CHECK:STDOUT:     .ProtectedCall = %ProtectedCall.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OverloadSet.decl: type = class_decl @OverloadSet [concrete = constants.%OverloadSet] {} {}
// CHECK:STDOUT:   %OverloadSet.OverloadSet.cpp_overload_set.value: %OverloadSet.OverloadSet.cpp_overload_set.type = cpp_overload_set_value @OverloadSet.OverloadSet.cpp_overload_set [concrete = constants.%OverloadSet.OverloadSet.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall.decl: type = class_decl @PublicCall [concrete = constants.%PublicCall] {} {}
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.decl: %PublicCall__carbon_thunk.type = fn_decl @PublicCall__carbon_thunk [concrete = constants.%PublicCall__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OverloadSet__carbon_thunk.decl.b67c1b.1: %OverloadSet__carbon_thunk.type.c7f6eb.1 = fn_decl @OverloadSet__carbon_thunk.1 [concrete = constants.%OverloadSet__carbon_thunk.e50482.1] {
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
// CHECK:STDOUT:   %OverloadSet__carbon_thunk.decl.b67c1b.2: %OverloadSet__carbon_thunk.type.c7f6eb.2 = fn_decl @OverloadSet__carbon_thunk.2 [concrete = constants.%OverloadSet__carbon_thunk.e50482.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OverloadSet.Overload.cpp_overload_set.value: %OverloadSet.Overload.cpp_overload_set.type = cpp_overload_set_value @OverloadSet.Overload.cpp_overload_set [concrete = constants.%OverloadSet.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Overload__carbon_thunk.decl.c2f84f.1: %Overload__carbon_thunk.type.3dd392.1 = fn_decl @Overload__carbon_thunk.1 [concrete = constants.%Overload__carbon_thunk.1c804c.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Overload__carbon_thunk.decl.c2f84f.2: %Overload__carbon_thunk.type.3dd392.2 = fn_decl @Overload__carbon_thunk.2 [concrete = constants.%Overload__carbon_thunk.1c804c.2] {
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
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .MakePublic = %D.MakePublic.decl
// CHECK:STDOUT:   .MakeProtected = %D.MakeProtected.decl
// CHECK:STDOUT:   .CallPublic = %D.CallPublic.decl
// CHECK:STDOUT:   .CallProtected = %D.CallProtected.decl
// CHECK:STDOUT:   .OverloadSet = <poisoned>
// CHECK:STDOUT:   .Overload = <poisoned>
// CHECK:STDOUT:   extend %OverloadSet.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.MakePublic() -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %OverloadSet.ref: %OverloadSet.OverloadSet.cpp_overload_set.type = name_ref OverloadSet, imports.%OverloadSet.OverloadSet.cpp_overload_set.value [concrete = constants.%OverloadSet.OverloadSet.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicCall.ref.loc11_36: type = name_ref PublicCall, imports.%PublicCall.decl [concrete = constants.%PublicCall]
// CHECK:STDOUT:   %PublicCall.ref.loc11_47: %PublicCall.PublicCall.cpp_overload_set.type = name_ref PublicCall, imports.%PublicCall.PublicCall.cpp_overload_set.value [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc11_59.1: ref %PublicCall = temporary_storage
// CHECK:STDOUT:   %addr.loc11_59: %ptr.d0a = addr_of %.loc11_59.1
// CHECK:STDOUT:   %PublicCall__carbon_thunk.call: init %empty_tuple.type = call imports.%PublicCall__carbon_thunk.decl(%addr.loc11_59)
// CHECK:STDOUT:   %.loc11_59.2: init %PublicCall to %.loc11_59.1 = mark_in_place_init %PublicCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc11_61.1: ref %.244 = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc11_59.3: ref %PublicCall = temporary %.loc11_59.1, %.loc11_59.2
// CHECK:STDOUT:   %.loc11_59.4: %PublicCall = acquire_value %.loc11_59.3
// CHECK:STDOUT:   %.loc11_59.5: ref %PublicCall = value_as_ref %.loc11_59.4
// CHECK:STDOUT:   %addr.loc11_60.1: %ptr.d0a = addr_of %.loc11_59.5
// CHECK:STDOUT:   %addr.loc11_60.2: %ptr.4a0 = addr_of %.loc11_61.1
// CHECK:STDOUT:   %OverloadSet__carbon_thunk.call: init %empty_tuple.type = call imports.%OverloadSet__carbon_thunk.decl.b67c1b.1(%addr.loc11_60.1, %addr.loc11_60.2)
// CHECK:STDOUT:   %.loc11_60: init %OverloadSet to %.loc11_61.1 = mark_in_place_init %OverloadSet__carbon_thunk.call
// CHECK:STDOUT:   %.loc11_61.2: %struct_type.base.d55 = struct_literal (%.loc11_60)
// CHECK:STDOUT:   %.loc11_61.3: init %.244 = as_compatible %.loc11_60
// CHECK:STDOUT:   %.loc11_61.4: init %.244 = converted %.loc11_60, %.loc11_61.3
// CHECK:STDOUT:   %.loc11_61.5: init %OverloadSet = as_compatible %.loc11_61.4
// CHECK:STDOUT:   %.loc11_61.6: init %D to %return.param = class_init (%.loc11_61.5)
// CHECK:STDOUT:   %.loc11_62: init %D = converted %.loc11_61.2, %.loc11_61.6
// CHECK:STDOUT:   %PublicCall.cpp_destructor.bound: <bound method> = bound_method %.loc11_59.3, constants.%PublicCall.cpp_destructor
// CHECK:STDOUT:   %PublicCall.cpp_destructor.call: init %empty_tuple.type = call %PublicCall.cpp_destructor.bound(%.loc11_59.3)
// CHECK:STDOUT:   return %.loc11_62 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.MakeProtected() -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %OverloadSet.ref: %OverloadSet.OverloadSet.cpp_overload_set.type = name_ref OverloadSet, imports.%OverloadSet.OverloadSet.cpp_overload_set.value [concrete = constants.%OverloadSet.OverloadSet.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ProtectedCall.ref.loc17_36: type = name_ref ProtectedCall, imports.%ProtectedCall.decl [concrete = constants.%ProtectedCall]
// CHECK:STDOUT:   %ProtectedCall.ref.loc17_50: %ProtectedCall.ProtectedCall.cpp_overload_set.type = name_ref ProtectedCall, imports.%ProtectedCall.ProtectedCall.cpp_overload_set.value [concrete = constants.%ProtectedCall.ProtectedCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc17_65.1: ref %ProtectedCall = temporary_storage
// CHECK:STDOUT:   %addr.loc17_65: %ptr.bc1 = addr_of %.loc17_65.1
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.call: init %empty_tuple.type = call imports.%ProtectedCall__carbon_thunk.decl(%addr.loc17_65)
// CHECK:STDOUT:   %.loc17_65.2: init %ProtectedCall to %.loc17_65.1 = mark_in_place_init %ProtectedCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc17_67.1: ref %.244 = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc17_65.3: ref %ProtectedCall = temporary %.loc17_65.1, %.loc17_65.2
// CHECK:STDOUT:   %.loc17_65.4: %ProtectedCall = acquire_value %.loc17_65.3
// CHECK:STDOUT:   %.loc17_65.5: ref %ProtectedCall = value_as_ref %.loc17_65.4
// CHECK:STDOUT:   %addr.loc17_66.1: %ptr.bc1 = addr_of %.loc17_65.5
// CHECK:STDOUT:   %addr.loc17_66.2: %ptr.4a0 = addr_of %.loc17_67.1
// CHECK:STDOUT:   %OverloadSet__carbon_thunk.call: init %empty_tuple.type = call imports.%OverloadSet__carbon_thunk.decl.b67c1b.2(%addr.loc17_66.1, %addr.loc17_66.2)
// CHECK:STDOUT:   %.loc17_66: init %OverloadSet to %.loc17_67.1 = mark_in_place_init %OverloadSet__carbon_thunk.call
// CHECK:STDOUT:   %.loc17_67.2: %struct_type.base.d55 = struct_literal (%.loc17_66)
// CHECK:STDOUT:   %.loc17_67.3: init %.244 = as_compatible %.loc17_66
// CHECK:STDOUT:   %.loc17_67.4: init %.244 = converted %.loc17_66, %.loc17_67.3
// CHECK:STDOUT:   %.loc17_67.5: init %OverloadSet = as_compatible %.loc17_67.4
// CHECK:STDOUT:   %.loc17_67.6: init %D to %return.param = class_init (%.loc17_67.5)
// CHECK:STDOUT:   %.loc17_68: init %D = converted %.loc17_67.2, %.loc17_67.6
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.bound: <bound method> = bound_method %.loc17_65.3, constants.%ProtectedCall.cpp_destructor
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.call: init %empty_tuple.type = call %ProtectedCall.cpp_destructor.bound(%.loc17_65.3)
// CHECK:STDOUT:   return %.loc17_68 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.CallPublic() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Overload.ref: %OverloadSet.Overload.cpp_overload_set.type = name_ref Overload, imports.%OverloadSet.Overload.cpp_overload_set.value [concrete = constants.%OverloadSet.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicCall.ref.loc23_17: type = name_ref PublicCall, imports.%PublicCall.decl [concrete = constants.%PublicCall]
// CHECK:STDOUT:   %PublicCall.ref.loc23_28: %PublicCall.PublicCall.cpp_overload_set.type = name_ref PublicCall, imports.%PublicCall.PublicCall.cpp_overload_set.value [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc23_40.1: ref %PublicCall = temporary_storage
// CHECK:STDOUT:   %addr.loc23_40: %ptr.d0a = addr_of %.loc23_40.1
// CHECK:STDOUT:   %PublicCall__carbon_thunk.call: init %empty_tuple.type = call imports.%PublicCall__carbon_thunk.decl(%addr.loc23_40)
// CHECK:STDOUT:   %.loc23_40.2: init %PublicCall to %.loc23_40.1 = mark_in_place_init %PublicCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc23_40.3: ref %PublicCall = temporary %.loc23_40.1, %.loc23_40.2
// CHECK:STDOUT:   %.loc23_40.4: %PublicCall = acquire_value %.loc23_40.3
// CHECK:STDOUT:   %.loc23_40.5: ref %PublicCall = value_as_ref %.loc23_40.4
// CHECK:STDOUT:   %addr.loc23_41: %ptr.d0a = addr_of %.loc23_40.5
// CHECK:STDOUT:   %Overload__carbon_thunk.call: init %empty_tuple.type = call imports.%Overload__carbon_thunk.decl.c2f84f.1(%addr.loc23_41)
// CHECK:STDOUT:   %PublicCall.cpp_destructor.bound: <bound method> = bound_method %.loc23_40.3, constants.%PublicCall.cpp_destructor
// CHECK:STDOUT:   %PublicCall.cpp_destructor.call: init %empty_tuple.type = call %PublicCall.cpp_destructor.bound(%.loc23_40.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.CallProtected() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Overload.ref: %OverloadSet.Overload.cpp_overload_set.type = name_ref Overload, imports.%OverloadSet.Overload.cpp_overload_set.value [concrete = constants.%OverloadSet.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ProtectedCall.ref.loc29_17: type = name_ref ProtectedCall, imports.%ProtectedCall.decl [concrete = constants.%ProtectedCall]
// CHECK:STDOUT:   %ProtectedCall.ref.loc29_31: %ProtectedCall.ProtectedCall.cpp_overload_set.type = name_ref ProtectedCall, imports.%ProtectedCall.ProtectedCall.cpp_overload_set.value [concrete = constants.%ProtectedCall.ProtectedCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc29_46.1: ref %ProtectedCall = temporary_storage
// CHECK:STDOUT:   %addr.loc29_46: %ptr.bc1 = addr_of %.loc29_46.1
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.call: init %empty_tuple.type = call imports.%ProtectedCall__carbon_thunk.decl(%addr.loc29_46)
// CHECK:STDOUT:   %.loc29_46.2: init %ProtectedCall to %.loc29_46.1 = mark_in_place_init %ProtectedCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc29_46.3: ref %ProtectedCall = temporary %.loc29_46.1, %.loc29_46.2
// CHECK:STDOUT:   %.loc29_46.4: %ProtectedCall = acquire_value %.loc29_46.3
// CHECK:STDOUT:   %.loc29_46.5: ref %ProtectedCall = value_as_ref %.loc29_46.4
// CHECK:STDOUT:   %addr.loc29_47: %ptr.bc1 = addr_of %.loc29_46.5
// CHECK:STDOUT:   %Overload__carbon_thunk.call: init %empty_tuple.type = call imports.%Overload__carbon_thunk.decl.c2f84f.2(%addr.loc29_47)
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.bound: <bound method> = bound_method %.loc29_46.3, constants.%ProtectedCall.cpp_destructor
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.call: init %empty_tuple.type = call %ProtectedCall.cpp_destructor.bound(%.loc29_46.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overload_set_public_base_class_call_public.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %OverloadPublic: type = class_type @OverloadPublic [concrete]
// CHECK:STDOUT:   %OverloadPublic.Overload.cpp_overload_set.type: type = cpp_overload_set_type @OverloadPublic.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OverloadPublic.Overload.cpp_overload_set.value: %OverloadPublic.Overload.cpp_overload_set.type = cpp_overload_set_value @OverloadPublic.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall: type = class_type @PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.type: type = cpp_overload_set_type @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d0a: type = ptr_type %PublicCall [concrete]
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
// CHECK:STDOUT:     .OverloadPublic = %OverloadPublic.decl
// CHECK:STDOUT:     .PublicCall = %PublicCall.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OverloadPublic.decl: type = class_decl @OverloadPublic [concrete = constants.%OverloadPublic] {} {}
// CHECK:STDOUT:   %OverloadPublic.Overload.cpp_overload_set.value: %OverloadPublic.Overload.cpp_overload_set.type = cpp_overload_set_value @OverloadPublic.Overload.cpp_overload_set [concrete = constants.%OverloadPublic.Overload.cpp_overload_set.value]
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
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %OverloadPublic.ref: type = name_ref OverloadPublic, imports.%OverloadPublic.decl [concrete = constants.%OverloadPublic]
// CHECK:STDOUT:   %Overload.ref: %OverloadPublic.Overload.cpp_overload_set.type = name_ref Overload, imports.%OverloadPublic.Overload.cpp_overload_set.value [concrete = constants.%OverloadPublic.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc8_31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicCall.ref.loc8_34: type = name_ref PublicCall, imports.%PublicCall.decl [concrete = constants.%PublicCall]
// CHECK:STDOUT:   %PublicCall.ref.loc8_45: %PublicCall.PublicCall.cpp_overload_set.type = name_ref PublicCall, imports.%PublicCall.PublicCall.cpp_overload_set.value [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_57.1: ref %PublicCall = temporary_storage
// CHECK:STDOUT:   %addr.loc8_57: %ptr.d0a = addr_of %.loc8_57.1
// CHECK:STDOUT:   %PublicCall__carbon_thunk.call: init %empty_tuple.type = call imports.%PublicCall__carbon_thunk.decl(%addr.loc8_57)
// CHECK:STDOUT:   %.loc8_57.2: init %PublicCall to %.loc8_57.1 = mark_in_place_init %PublicCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_57.3: ref %PublicCall = temporary %.loc8_57.1, %.loc8_57.2
// CHECK:STDOUT:   %.loc8_57.4: %PublicCall = acquire_value %.loc8_57.3
// CHECK:STDOUT:   %.loc8_57.5: ref %PublicCall = value_as_ref %.loc8_57.4
// CHECK:STDOUT:   %addr.loc8_58: %ptr.d0a = addr_of %.loc8_57.5
// CHECK:STDOUT:   %Overload__carbon_thunk.call: init %empty_tuple.type = call imports.%Overload__carbon_thunk.decl(%addr.loc8_58)
// CHECK:STDOUT:   %PublicCall.cpp_destructor.bound: <bound method> = bound_method %.loc8_57.3, constants.%PublicCall.cpp_destructor
// CHECK:STDOUT:   %PublicCall.cpp_destructor.call: init %empty_tuple.type = call %PublicCall.cpp_destructor.bound(%.loc8_57.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overload_set_public_base_class_derived_call_non_private.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %OverloadPublic: type = class_type @OverloadPublic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %OverloadPublic.Overload.cpp_overload_set.type: type = cpp_overload_set_type @OverloadPublic.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OverloadPublic.Overload.cpp_overload_set.value: %OverloadPublic.Overload.cpp_overload_set.type = cpp_overload_set_value @OverloadPublic.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall: type = class_type @PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.type: type = cpp_overload_set_type @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d0a: type = ptr_type %PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.type: type = fn_type @PublicCall__carbon_thunk [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk: %PublicCall__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.type.3dd392.1: type = fn_type @Overload__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.1c804c.1: %Overload__carbon_thunk.type.3dd392.1 = struct_value () [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor.type: type = fn_type @PublicCall.cpp_destructor [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor: %PublicCall.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ProtectedCall: type = class_type @ProtectedCall [concrete]
// CHECK:STDOUT:   %ProtectedCall.ProtectedCall.cpp_overload_set.type: type = cpp_overload_set_type @ProtectedCall.ProtectedCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ProtectedCall.ProtectedCall.cpp_overload_set.value: %ProtectedCall.ProtectedCall.cpp_overload_set.type = cpp_overload_set_value @ProtectedCall.ProtectedCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.bc1: type = ptr_type %ProtectedCall [concrete]
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.type: type = fn_type @ProtectedCall__carbon_thunk [concrete]
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk: %ProtectedCall__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.type.3dd392.2: type = fn_type @Overload__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.1c804c.2: %Overload__carbon_thunk.type.3dd392.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.type: type = fn_type @ProtectedCall.cpp_destructor [concrete]
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor: %ProtectedCall.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OverloadPublic = %OverloadPublic.decl
// CHECK:STDOUT:     .PublicCall = %PublicCall.decl
// CHECK:STDOUT:     .ProtectedCall = %ProtectedCall.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OverloadPublic.decl: type = class_decl @OverloadPublic [concrete = constants.%OverloadPublic] {} {}
// CHECK:STDOUT:   %OverloadPublic.Overload.cpp_overload_set.value: %OverloadPublic.Overload.cpp_overload_set.type = cpp_overload_set_value @OverloadPublic.Overload.cpp_overload_set [concrete = constants.%OverloadPublic.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall.decl: type = class_decl @PublicCall [concrete = constants.%PublicCall] {} {}
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.decl: %PublicCall__carbon_thunk.type = fn_decl @PublicCall__carbon_thunk [concrete = constants.%PublicCall__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Overload__carbon_thunk.decl.c2f84f.1: %Overload__carbon_thunk.type.3dd392.1 = fn_decl @Overload__carbon_thunk.1 [concrete = constants.%Overload__carbon_thunk.1c804c.1] {
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
// CHECK:STDOUT:   %Overload__carbon_thunk.decl.c2f84f.2: %Overload__carbon_thunk.type.3dd392.2 = fn_decl @Overload__carbon_thunk.2 [concrete = constants.%Overload__carbon_thunk.1c804c.2] {
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
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   .Overload = <poisoned>
// CHECK:STDOUT:   extend %OverloadPublic.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Overload.ref.loc11: %OverloadPublic.Overload.cpp_overload_set.type = name_ref Overload, imports.%OverloadPublic.Overload.cpp_overload_set.value [concrete = constants.%OverloadPublic.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicCall.ref.loc11_17: type = name_ref PublicCall, imports.%PublicCall.decl [concrete = constants.%PublicCall]
// CHECK:STDOUT:   %PublicCall.ref.loc11_28: %PublicCall.PublicCall.cpp_overload_set.type = name_ref PublicCall, imports.%PublicCall.PublicCall.cpp_overload_set.value [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc11_40.1: ref %PublicCall = temporary_storage
// CHECK:STDOUT:   %addr.loc11_40: %ptr.d0a = addr_of %.loc11_40.1
// CHECK:STDOUT:   %PublicCall__carbon_thunk.call: init %empty_tuple.type = call imports.%PublicCall__carbon_thunk.decl(%addr.loc11_40)
// CHECK:STDOUT:   %.loc11_40.2: init %PublicCall to %.loc11_40.1 = mark_in_place_init %PublicCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc11_40.3: ref %PublicCall = temporary %.loc11_40.1, %.loc11_40.2
// CHECK:STDOUT:   %.loc11_40.4: %PublicCall = acquire_value %.loc11_40.3
// CHECK:STDOUT:   %.loc11_40.5: ref %PublicCall = value_as_ref %.loc11_40.4
// CHECK:STDOUT:   %addr.loc11_41: %ptr.d0a = addr_of %.loc11_40.5
// CHECK:STDOUT:   %Overload__carbon_thunk.call.loc11: init %empty_tuple.type = call imports.%Overload__carbon_thunk.decl.c2f84f.1(%addr.loc11_41)
// CHECK:STDOUT:   %PublicCall.cpp_destructor.bound: <bound method> = bound_method %.loc11_40.3, constants.%PublicCall.cpp_destructor
// CHECK:STDOUT:   %PublicCall.cpp_destructor.call: init %empty_tuple.type = call %PublicCall.cpp_destructor.bound(%.loc11_40.3)
// CHECK:STDOUT:   %Overload.ref.loc12: %OverloadPublic.Overload.cpp_overload_set.type = name_ref Overload, imports.%OverloadPublic.Overload.cpp_overload_set.value [concrete = constants.%OverloadPublic.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ProtectedCall.ref.loc12_17: type = name_ref ProtectedCall, imports.%ProtectedCall.decl [concrete = constants.%ProtectedCall]
// CHECK:STDOUT:   %ProtectedCall.ref.loc12_31: %ProtectedCall.ProtectedCall.cpp_overload_set.type = name_ref ProtectedCall, imports.%ProtectedCall.ProtectedCall.cpp_overload_set.value [concrete = constants.%ProtectedCall.ProtectedCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc12_46.1: ref %ProtectedCall = temporary_storage
// CHECK:STDOUT:   %addr.loc12_46: %ptr.bc1 = addr_of %.loc12_46.1
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.call: init %empty_tuple.type = call imports.%ProtectedCall__carbon_thunk.decl(%addr.loc12_46)
// CHECK:STDOUT:   %.loc12_46.2: init %ProtectedCall to %.loc12_46.1 = mark_in_place_init %ProtectedCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc12_46.3: ref %ProtectedCall = temporary %.loc12_46.1, %.loc12_46.2
// CHECK:STDOUT:   %.loc12_46.4: %ProtectedCall = acquire_value %.loc12_46.3
// CHECK:STDOUT:   %.loc12_46.5: ref %ProtectedCall = value_as_ref %.loc12_46.4
// CHECK:STDOUT:   %addr.loc12_47: %ptr.bc1 = addr_of %.loc12_46.5
// CHECK:STDOUT:   %Overload__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%Overload__carbon_thunk.decl.c2f84f.2(%addr.loc12_47)
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.bound: <bound method> = bound_method %.loc12_46.3, constants.%ProtectedCall.cpp_destructor
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.call: init %empty_tuple.type = call %ProtectedCall.cpp_destructor.bound(%.loc12_46.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overload_set_protected_base_class_derived_call_non_private.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %OverloadProtected: type = class_type @OverloadProtected [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %OverloadProtected.Overload.cpp_overload_set.type: type = cpp_overload_set_type @OverloadProtected.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %OverloadProtected.Overload.cpp_overload_set.value: %OverloadProtected.Overload.cpp_overload_set.type = cpp_overload_set_value @OverloadProtected.Overload.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall: type = class_type @PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.type: type = cpp_overload_set_type @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d0a: type = ptr_type %PublicCall [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.type: type = fn_type @PublicCall__carbon_thunk [concrete]
// CHECK:STDOUT:   %PublicCall__carbon_thunk: %PublicCall__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.type.3dd392.1: type = fn_type @Overload__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.1c804c.1: %Overload__carbon_thunk.type.3dd392.1 = struct_value () [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor.type: type = fn_type @PublicCall.cpp_destructor [concrete]
// CHECK:STDOUT:   %PublicCall.cpp_destructor: %PublicCall.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ProtectedCall: type = class_type @ProtectedCall [concrete]
// CHECK:STDOUT:   %ProtectedCall.ProtectedCall.cpp_overload_set.type: type = cpp_overload_set_type @ProtectedCall.ProtectedCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ProtectedCall.ProtectedCall.cpp_overload_set.value: %ProtectedCall.ProtectedCall.cpp_overload_set.type = cpp_overload_set_value @ProtectedCall.ProtectedCall.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.bc1: type = ptr_type %ProtectedCall [concrete]
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.type: type = fn_type @ProtectedCall__carbon_thunk [concrete]
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk: %ProtectedCall__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.type.3dd392.2: type = fn_type @Overload__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %Overload__carbon_thunk.1c804c.2: %Overload__carbon_thunk.type.3dd392.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.type: type = fn_type @ProtectedCall.cpp_destructor [concrete]
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor: %ProtectedCall.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OverloadProtected = %OverloadProtected.decl
// CHECK:STDOUT:     .PublicCall = %PublicCall.decl
// CHECK:STDOUT:     .ProtectedCall = %ProtectedCall.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %OverloadProtected.decl: type = class_decl @OverloadProtected [concrete = constants.%OverloadProtected] {} {}
// CHECK:STDOUT:   %OverloadProtected.Overload.cpp_overload_set.value: %OverloadProtected.Overload.cpp_overload_set.type = cpp_overload_set_value @OverloadProtected.Overload.cpp_overload_set [concrete = constants.%OverloadProtected.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall.decl: type = class_decl @PublicCall [concrete = constants.%PublicCall] {} {}
// CHECK:STDOUT:   %PublicCall.PublicCall.cpp_overload_set.value: %PublicCall.PublicCall.cpp_overload_set.type = cpp_overload_set_value @PublicCall.PublicCall.cpp_overload_set [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %PublicCall__carbon_thunk.decl: %PublicCall__carbon_thunk.type = fn_decl @PublicCall__carbon_thunk [concrete = constants.%PublicCall__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Overload__carbon_thunk.decl.c2f84f.1: %Overload__carbon_thunk.type.3dd392.1 = fn_decl @Overload__carbon_thunk.1 [concrete = constants.%Overload__carbon_thunk.1c804c.1] {
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
// CHECK:STDOUT:   %Overload__carbon_thunk.decl.c2f84f.2: %Overload__carbon_thunk.type.3dd392.2 = fn_decl @Overload__carbon_thunk.2 [concrete = constants.%Overload__carbon_thunk.1c804c.2] {
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
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   .Overload = <poisoned>
// CHECK:STDOUT:   extend %OverloadProtected.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Overload.ref.loc11: %OverloadProtected.Overload.cpp_overload_set.type = name_ref Overload, imports.%OverloadProtected.Overload.cpp_overload_set.value [concrete = constants.%OverloadProtected.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PublicCall.ref.loc11_17: type = name_ref PublicCall, imports.%PublicCall.decl [concrete = constants.%PublicCall]
// CHECK:STDOUT:   %PublicCall.ref.loc11_28: %PublicCall.PublicCall.cpp_overload_set.type = name_ref PublicCall, imports.%PublicCall.PublicCall.cpp_overload_set.value [concrete = constants.%PublicCall.PublicCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc11_40.1: ref %PublicCall = temporary_storage
// CHECK:STDOUT:   %addr.loc11_40: %ptr.d0a = addr_of %.loc11_40.1
// CHECK:STDOUT:   %PublicCall__carbon_thunk.call: init %empty_tuple.type = call imports.%PublicCall__carbon_thunk.decl(%addr.loc11_40)
// CHECK:STDOUT:   %.loc11_40.2: init %PublicCall to %.loc11_40.1 = mark_in_place_init %PublicCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc11_40.3: ref %PublicCall = temporary %.loc11_40.1, %.loc11_40.2
// CHECK:STDOUT:   %.loc11_40.4: %PublicCall = acquire_value %.loc11_40.3
// CHECK:STDOUT:   %.loc11_40.5: ref %PublicCall = value_as_ref %.loc11_40.4
// CHECK:STDOUT:   %addr.loc11_41: %ptr.d0a = addr_of %.loc11_40.5
// CHECK:STDOUT:   %Overload__carbon_thunk.call.loc11: init %empty_tuple.type = call imports.%Overload__carbon_thunk.decl.c2f84f.1(%addr.loc11_41)
// CHECK:STDOUT:   %PublicCall.cpp_destructor.bound: <bound method> = bound_method %.loc11_40.3, constants.%PublicCall.cpp_destructor
// CHECK:STDOUT:   %PublicCall.cpp_destructor.call: init %empty_tuple.type = call %PublicCall.cpp_destructor.bound(%.loc11_40.3)
// CHECK:STDOUT:   %Overload.ref.loc12: %OverloadProtected.Overload.cpp_overload_set.type = name_ref Overload, imports.%OverloadProtected.Overload.cpp_overload_set.value [concrete = constants.%OverloadProtected.Overload.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ProtectedCall.ref.loc12_17: type = name_ref ProtectedCall, imports.%ProtectedCall.decl [concrete = constants.%ProtectedCall]
// CHECK:STDOUT:   %ProtectedCall.ref.loc12_31: %ProtectedCall.ProtectedCall.cpp_overload_set.type = name_ref ProtectedCall, imports.%ProtectedCall.ProtectedCall.cpp_overload_set.value [concrete = constants.%ProtectedCall.ProtectedCall.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc12_46.1: ref %ProtectedCall = temporary_storage
// CHECK:STDOUT:   %addr.loc12_46: %ptr.bc1 = addr_of %.loc12_46.1
// CHECK:STDOUT:   %ProtectedCall__carbon_thunk.call: init %empty_tuple.type = call imports.%ProtectedCall__carbon_thunk.decl(%addr.loc12_46)
// CHECK:STDOUT:   %.loc12_46.2: init %ProtectedCall to %.loc12_46.1 = mark_in_place_init %ProtectedCall__carbon_thunk.call
// CHECK:STDOUT:   %.loc12_46.3: ref %ProtectedCall = temporary %.loc12_46.1, %.loc12_46.2
// CHECK:STDOUT:   %.loc12_46.4: %ProtectedCall = acquire_value %.loc12_46.3
// CHECK:STDOUT:   %.loc12_46.5: ref %ProtectedCall = value_as_ref %.loc12_46.4
// CHECK:STDOUT:   %addr.loc12_47: %ptr.bc1 = addr_of %.loc12_46.5
// CHECK:STDOUT:   %Overload__carbon_thunk.call.loc12: init %empty_tuple.type = call imports.%Overload__carbon_thunk.decl.c2f84f.2(%addr.loc12_47)
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.bound: <bound method> = bound_method %.loc12_46.3, constants.%ProtectedCall.cpp_destructor
// CHECK:STDOUT:   %ProtectedCall.cpp_destructor.call: init %empty_tuple.type = call %ProtectedCall.cpp_destructor.bound(%.loc12_46.3)
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
// CHECK:STDOUT: fn @Derived.CallInstance(%instance.param: ref %Public) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %instance.ref.loc24: ref %Public = name_ref instance, %instance
// CHECK:STDOUT:   %PublicInstance.ref: %Public.PublicInstance.cpp_overload_set.type = name_ref PublicInstance, imports.%Public.PublicInstance.cpp_overload_set.value [concrete = constants.%Public.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc24: <bound method> = bound_method %instance.ref.loc24, %PublicInstance.ref
// CHECK:STDOUT:   %.loc24_5.1: ref %Base = as_compatible %instance.ref.loc24
// CHECK:STDOUT:   %.loc24_5.2: ref %Base = class_element_access %.loc24_5.1, element0
// CHECK:STDOUT:   %.loc24_5.3: ref %Base = converted %instance.ref.loc24, %.loc24_5.2
// CHECK:STDOUT:   %Base.PublicInstance.call: init %empty_tuple.type = call imports.%Base.PublicInstance.decl(%.loc24_5.3)
// CHECK:STDOUT:   %instance.ref.loc26: ref %Public = name_ref instance, %instance
// CHECK:STDOUT:   %ProtectedInstance.ref: %Public.ProtectedInstance.cpp_overload_set.type = name_ref ProtectedInstance, imports.%Public.ProtectedInstance.cpp_overload_set.value [concrete = constants.%Public.ProtectedInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc26: <bound method> = bound_method %instance.ref.loc26, %ProtectedInstance.ref
// CHECK:STDOUT:   %.loc26_5.1: ref %Base = as_compatible %instance.ref.loc26
// CHECK:STDOUT:   %.loc26_5.2: ref %Base = class_element_access %.loc26_5.1, element0
// CHECK:STDOUT:   %.loc26_5.3: ref %Base = converted %instance.ref.loc26, %.loc26_5.2
// CHECK:STDOUT:   %Base.ProtectedInstance.call: init %empty_tuple.type = call imports.%Base.ProtectedInstance.decl(%.loc26_5.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%instance.param: ref %Public) {
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
// CHECK:STDOUT:   %.loc36_3.1: ref %Base = as_compatible %instance.ref.loc36
// CHECK:STDOUT:   %.loc36_3.2: ref %Base = class_element_access %.loc36_3.1, element0
// CHECK:STDOUT:   %.loc36_3.3: ref %Base = converted %instance.ref.loc36, %.loc36_3.2
// CHECK:STDOUT:   %Base.PublicInstance.call.loc36: init %empty_tuple.type = call imports.%Base.PublicInstance.decl(%.loc36_3.3)
// CHECK:STDOUT:   %instance.ref.loc37: ref %Public = name_ref instance, %instance
// CHECK:STDOUT:   %PublicInstance.ref.loc37: %Public.PublicInstance.cpp_overload_set.type = name_ref PublicInstance, imports.%Public.PublicInstance.cpp_overload_set.value [concrete = constants.%Public.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc37: <bound method> = bound_method %instance.ref.loc37, %PublicInstance.ref.loc37
// CHECK:STDOUT:   %.loc37_3.1: ref %Base = as_compatible %instance.ref.loc37
// CHECK:STDOUT:   %.loc37_3.2: ref %Base = class_element_access %.loc37_3.1, element0
// CHECK:STDOUT:   %.loc37_3.3: ref %Base = converted %instance.ref.loc37, %.loc37_3.2
// CHECK:STDOUT:   %Base.PublicInstance.call.loc37: init %empty_tuple.type = call imports.%Base.PublicInstance.decl(%.loc37_3.3)
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
// CHECK:STDOUT:   .base = %.loc7
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
// CHECK:STDOUT: fn @Derived.CallInstance(%instance.param: ref %Protected) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %instance.ref.loc24: ref %Protected = name_ref instance, %instance
// CHECK:STDOUT:   %PublicInstance.ref: %Protected.PublicInstance.cpp_overload_set.type = name_ref PublicInstance, imports.%Protected.PublicInstance.cpp_overload_set.value [concrete = constants.%Protected.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc24: <bound method> = bound_method %instance.ref.loc24, %PublicInstance.ref
// CHECK:STDOUT:   %.loc24_5.1: ref %Base = as_compatible %instance.ref.loc24
// CHECK:STDOUT:   %.loc24_5.2: ref %Base = class_element_access %.loc24_5.1, element0
// CHECK:STDOUT:   %.loc24_5.3: ref %Base = converted %instance.ref.loc24, %.loc24_5.2
// CHECK:STDOUT:   %Base.PublicInstance.call: init %empty_tuple.type = call imports.%Base.PublicInstance.decl(%.loc24_5.3)
// CHECK:STDOUT:   %instance.ref.loc26: ref %Protected = name_ref instance, %instance
// CHECK:STDOUT:   %ProtectedInstance.ref: %Protected.ProtectedInstance.cpp_overload_set.type = name_ref ProtectedInstance, imports.%Protected.ProtectedInstance.cpp_overload_set.value [concrete = constants.%Protected.ProtectedInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc26: <bound method> = bound_method %instance.ref.loc26, %ProtectedInstance.ref
// CHECK:STDOUT:   %.loc26_5.1: ref %Base = as_compatible %instance.ref.loc26
// CHECK:STDOUT:   %.loc26_5.2: ref %Base = class_element_access %.loc26_5.1, element0
// CHECK:STDOUT:   %.loc26_5.3: ref %Base = converted %instance.ref.loc26, %.loc26_5.2
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
// CHECK:STDOUT: fn @Derived.CallInstance(%instance.param: ref %PublicProtected) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %instance.ref.loc24: ref %PublicProtected = name_ref instance, %instance
// CHECK:STDOUT:   %PublicInstance.ref: %PublicProtected.PublicInstance.cpp_overload_set.type = name_ref PublicInstance, imports.%PublicProtected.PublicInstance.cpp_overload_set.value [concrete = constants.%PublicProtected.PublicInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc24: <bound method> = bound_method %instance.ref.loc24, %PublicInstance.ref
// CHECK:STDOUT:   %.loc24_5.1: ref %Base = as_compatible %instance.ref.loc24
// CHECK:STDOUT:   %.loc24_5.2: ref %Protected = class_element_access %.loc24_5.1, element0
// CHECK:STDOUT:   %.loc24_5.3: ref %Base = class_element_access %.loc24_5.2, element0
// CHECK:STDOUT:   %.loc24_5.4: ref %Base = converted %instance.ref.loc24, %.loc24_5.3
// CHECK:STDOUT:   %Base.PublicInstance.call: init %empty_tuple.type = call imports.%Base.PublicInstance.decl(%.loc24_5.4)
// CHECK:STDOUT:   %instance.ref.loc26: ref %PublicProtected = name_ref instance, %instance
// CHECK:STDOUT:   %ProtectedInstance.ref: %PublicProtected.ProtectedInstance.cpp_overload_set.type = name_ref ProtectedInstance, imports.%PublicProtected.ProtectedInstance.cpp_overload_set.value [concrete = constants.%PublicProtected.ProtectedInstance.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc26: <bound method> = bound_method %instance.ref.loc26, %ProtectedInstance.ref
// CHECK:STDOUT:   %.loc26_5.1: ref %Base = as_compatible %instance.ref.loc26
// CHECK:STDOUT:   %.loc26_5.2: ref %Protected = class_element_access %.loc26_5.1, element0
// CHECK:STDOUT:   %.loc26_5.3: ref %Base = class_element_access %.loc26_5.2, element0
// CHECK:STDOUT:   %.loc26_5.4: ref %Base = converted %instance.ref.loc26, %.loc26_5.3
// CHECK:STDOUT:   %Base.ProtectedInstance.call: init %empty_tuple.type = call imports.%Base.ProtectedInstance.decl(%.loc26_5.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
