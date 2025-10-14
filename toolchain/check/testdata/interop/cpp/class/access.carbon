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
  static auto foo() -> void;
 protected:
  C(int);
  static auto foo(int a) -> void;
 private:
  C(int, int);
  static auto foo(int a, int b) -> void;
};

// --- import_overload_set_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.C.C();
  Cpp.C.foo();
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

  // CHECK:STDERR: fail_import_overload_set_protected.carbon:[[@LINE+5]]:3: error: cannot access protected member `foo` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.C.foo(1 as i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_protected.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.C.foo(1 as i32);
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
    foo();
  }

  fn CallProtected() {
    foo(1 as i32);
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

  // CHECK:STDERR: fail_import_overload_set_private.carbon:[[@LINE+5]]:3: error: cannot access private member `foo` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.C.foo(1 as i32, 2 as i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_overload_set_private.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.C.foo(1 as i32, 2 as i32);
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
    // CHECK:STDERR: fail_import_overload_set_private_base.carbon:[[@LINE+5]]:5: error: cannot access private member `foo` of type `Cpp.C` [ClassInvalidMemberAccess]
    // CHECK:STDERR:     foo(1 as i32, 2 as i32);
    // CHECK:STDERR:     ^~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR: fail_import_overload_set_private_base.carbon: note: declared here [ClassMemberDeclaration]
    // CHECK:STDERR:
    foo(1 as i32, 2 as i32);
  }
}

// ============================================================================
// Base class
// ============================================================================

// --- base_class.h

struct Base {
  static auto foo() -> void;
};

class DerivedPublic : public Base {};
class DerivedProtected : protected Base {};
class DerivedPrivate : private Base {};

// --- import_base_class_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.DerivedPublic.foo();
  //@dump-sem-ir-end
}

// --- fail_import_base_class_protected.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

fn F() {
  // CHECK:STDERR: fail_import_base_class_protected.carbon:[[@LINE+5]]:3: error: cannot access protected member `foo` of type `Cpp.DerivedProtected` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.DerivedProtected.foo();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_base_class_protected.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.DerivedProtected.foo();
}

// --- use_base_class_protected_from_derived.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

class Derived {
  extend base: Cpp.DerivedProtected;

  fn F() {
    // OK, we can access a protected member of our base class.
    Cpp.DerivedProtected.foo();
  }
}

// --- fail_import_base_class_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "base_class.h";

fn F() {
  // CHECK:STDERR: fail_import_base_class_private.carbon:[[@LINE+5]]:3: error: cannot access private member `foo` of type `Cpp.DerivedPrivate` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   Cpp.DerivedPrivate.foo();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_base_class_private.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  Cpp.DerivedPrivate.foo();
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
// CHECK:STDOUT:   %C.foo.cpp_overload_set.type: type = cpp_overload_set_type @C.foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.foo.cpp_overload_set.value: %C.foo.cpp_overload_set.type = cpp_overload_set_value @C.foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.foo.type: type = fn_type @C.foo [concrete]
// CHECK:STDOUT:   %C.foo: %C.foo.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %C.foo.cpp_overload_set.value: %C.foo.cpp_overload_set.type = cpp_overload_set_value @C.foo.cpp_overload_set [concrete = constants.%C.foo.cpp_overload_set.value]
// CHECK:STDOUT:   %C.foo.decl: %C.foo.type = fn_decl @C.foo [concrete = constants.%C.foo] {} {}
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
// CHECK:STDOUT:   %foo.ref: %C.foo.cpp_overload_set.type = name_ref foo, imports.%C.foo.cpp_overload_set.value [concrete = constants.%C.foo.cpp_overload_set.value]
// CHECK:STDOUT:   %C.foo.call: init %empty_tuple.type = call imports.%C.foo.decl()
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_11.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_11.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_11.2: %ptr.d9e = addr_of %.loc8_11.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc8_11.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_base_class_public.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DerivedPublic: type = class_type @DerivedPublic [concrete]
// CHECK:STDOUT:   %DerivedPublic.foo.cpp_overload_set.type: type = cpp_overload_set_type @DerivedPublic.foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %DerivedPublic.foo.cpp_overload_set.value: %DerivedPublic.foo.cpp_overload_set.type = cpp_overload_set_value @DerivedPublic.foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Base.foo.type: type = fn_type @Base.foo [concrete]
// CHECK:STDOUT:   %Base.foo: %Base.foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DerivedPublic = %DerivedPublic.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DerivedPublic.decl: type = class_decl @DerivedPublic [concrete = constants.%DerivedPublic] {} {}
// CHECK:STDOUT:   %DerivedPublic.foo.cpp_overload_set.value: %DerivedPublic.foo.cpp_overload_set.type = cpp_overload_set_value @DerivedPublic.foo.cpp_overload_set [concrete = constants.%DerivedPublic.foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.foo.decl: %Base.foo.type = fn_decl @Base.foo [concrete = constants.%Base.foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %DerivedPublic.ref: type = name_ref DerivedPublic, imports.%DerivedPublic.decl [concrete = constants.%DerivedPublic]
// CHECK:STDOUT:   %foo.ref: %DerivedPublic.foo.cpp_overload_set.type = name_ref foo, imports.%DerivedPublic.foo.cpp_overload_set.value [concrete = constants.%DerivedPublic.foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Base.foo.call: init %empty_tuple.type = call imports.%Base.foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
