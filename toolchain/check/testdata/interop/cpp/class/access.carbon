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

// TODO: Tests marked as `fail_todo_5891_` to fixed as a follow-up of https://github.com/carbon-language/carbon-lang/pull/5891.

// ============================================================================
// Public non-function member
// ============================================================================

// --- non_function_member_public.h

struct S {
  int x;
};

// --- import_non_function_member_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_public.h";

fn F(s: Cpp.S) {
  //@dump-sem-ir-begin
  let x: i32 = s.x;
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
  let x: i32 = d.x;
  //@dump-sem-ir-end
}

// ============================================================================
// Protected non-function member
// ============================================================================

// --- non_function_member_protected.h

class C {
 protected:
  int x;
};

// --- fail_import_non_function_member_protected.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_protected.h";

fn F(c: Cpp.C) {
  // CHECK:STDERR: fail_import_non_function_member_protected.carbon:[[@LINE+8]]:16: error: cannot access protected member `x` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let x: i32 = c.x;
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR: fail_import_non_function_member_protected.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./non_function_member_protected.h:2:7: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: class C {
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let x: i32 = c.x;
}

// --- fail_import_non_function_member_protected_extend.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_protected.h";

class Derived {
  extend base: Cpp.C;
}

fn F(d: Derived) {
  // CHECK:STDERR: fail_import_non_function_member_protected_extend.carbon:[[@LINE+8]]:16: error: cannot access protected member `x` of type `Derived` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let x: i32 = d.x;
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR: fail_import_non_function_member_protected_extend.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./non_function_member_protected.h:2:7: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: class C {
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let x: i32 = d.x;
}

// ============================================================================
// Private non-function member
// ============================================================================

// --- non_function_member_private.h

class C {
  int x;
};

// --- fail_import_non_function_member_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_private.h";

fn F(c: Cpp.C) {
  // CHECK:STDERR: fail_import_non_function_member_private.carbon:[[@LINE+8]]:16: error: cannot access private member `x` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let x: i32 = c.x;
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR: fail_import_non_function_member_private.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./non_function_member_private.h:2:7: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: class C {
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let x: i32 = c.x;
}

// --- fail_import_non_function_member_private_extend.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_function_member_private.h";

class Derived {
  extend base: Cpp.C;
}

fn F(d: Derived) {
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE+8]]:16: error: cannot access private member `x` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let x: i32 = d.x;
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR: fail_import_non_function_member_private_extend.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./non_function_member_private.h:2:7: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR: class C {
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let x: i32 = d.x;
}

// ============================================================================
// Public function member
// ============================================================================

// --- function_member_public.h

struct S {
  auto foo() -> void;
};

// --- import_function_member_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "function_member_public.h";

fn F(s: Cpp.S*) {
  //@dump-sem-ir-begin
  s->foo();
  //@dump-sem-ir-end
}

// ============================================================================
// Protected function member
// ============================================================================

// --- function_member_protected.h

class C {
 protected:
  auto foo() -> void;
};

// --- fail_import_function_member_protected.carbon

library "[[@TEST_NAME]]";

import Cpp library "function_member_protected.h";

fn F(c: Cpp.C*) {
  // CHECK:STDERR: fail_import_function_member_protected.carbon:[[@LINE+5]]:3: error: cannot access protected member `foo` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   c->foo();
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_import_function_member_protected.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  c->foo();
}

// ============================================================================
// Private function member
// ============================================================================

// --- function_member_private.h

class C {
 private:
  auto foo() -> void;
};

// --- fail_import_function_member_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "function_member_private.h";

fn F(c: Cpp.C*) {
  // CHECK:STDERR: fail_import_function_member_private.carbon:[[@LINE+5]]:3: error: cannot access private member `foo` of type `Cpp.C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   c->foo();
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_import_function_member_private.carbon: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:
  c->foo();
}

// ============================================================================
// Overload set
// ============================================================================

// --- overload_set.h

class C {
 public:
  static auto foo() -> void;
 protected:
  static auto foo(int a) -> void;
 private:
  static auto foo(int a, int b) -> void;
};

// --- fail_todo_5891_import_overload_set_public.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_5891_import_overload_set_public.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: Overloaded set with mixed access` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.C.foo();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_import_overload_set_public.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.C.foo();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  Cpp.C.foo();
  //@dump-sem-ir-end
}

// --- fail_todo_5891_import_overload_set_protected.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

fn F() {
  // CHECK:STDERR: fail_todo_5891_import_overload_set_protected.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: Overloaded set with mixed access` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.C.foo(1 as i32);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_import_overload_set_protected.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.C.foo(1 as i32);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  Cpp.C.foo(1 as i32);
}

// --- fail_todo_5891_import_overload_set_private.carbon

library "[[@TEST_NAME]]";

import Cpp library "overload_set.h";

fn F() {
  // CHECK:STDERR: fail_todo_5891_import_overload_set_private.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: Overloaded set with mixed access` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.C.foo(1 as i32, 2 as i32);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_5891_import_overload_set_private.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.C.foo(1 as i32, 2 as i32);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  Cpp.C.foo(1 as i32, 2 as i32);
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %S) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.ref: %S = name_ref s, %s
// CHECK:STDOUT:   %x.ref: %S.elem = name_ref x, @S.%.1 [concrete = @S.%.1]
// CHECK:STDOUT:   %.loc8_17.1: ref %i32 = class_element_access %s.ref, element0
// CHECK:STDOUT:   %.loc8_17.2: %i32 = bind_value %.loc8_17.1
// CHECK:STDOUT:   %.loc8_10: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: %i32 = bind_name x, %.loc8_17.2
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%d.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %x.ref: %S.elem = name_ref x, @S.%.1 [concrete = @S.%.1]
// CHECK:STDOUT:   %.loc12_17.1: ref %S = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc12_17.2: ref %S = converted %d.ref, %.loc12_17.1
// CHECK:STDOUT:   %.loc12_17.3: ref %i32 = class_element_access %.loc12_17.2, element0
// CHECK:STDOUT:   %.loc12_10: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_17.4: %i32 = bind_value %.loc12_17.3
// CHECK:STDOUT:   %x: %i32 = bind_name x, %.loc12_17.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_function_member_public.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %ptr.5c7: type = ptr_type %S [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.177: type = cpp_overload_set_type @S.foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.177 = struct_value () [concrete]
// CHECK:STDOUT:   %S.foo.type: type = fn_type @S.foo [concrete]
// CHECK:STDOUT:   %S.foo: %S.foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %.dcb: %.177 = cpp_overload_set_value @S.foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %S.foo.decl: %S.foo.type = fn_decl @S.foo [concrete = constants.%S.foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%s.param: %ptr.5c7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: %ptr.5c7 = name_ref s, %s
// CHECK:STDOUT:   %.loc8: ref %S = deref %s.ref
// CHECK:STDOUT:   %foo.ref: %.177 = name_ref foo, imports.%.dcb [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8, %foo.ref
// CHECK:STDOUT:   %addr: %ptr.5c7 = addr_of %.loc8
// CHECK:STDOUT:   %S.foo.call: init %empty_tuple.type = call imports.%S.foo.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_5891_import_overload_set_public.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %foo.ref: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_base_class_public.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DerivedPublic: type = class_type @DerivedPublic [concrete]
// CHECK:STDOUT:   %.5a6: type = cpp_overload_set_type @Base.foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.5a6 = struct_value () [concrete]
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
// CHECK:STDOUT:   %.607: %.5a6 = cpp_overload_set_value @Base.foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Base.foo.decl: %Base.foo.type = fn_decl @Base.foo [concrete = constants.%Base.foo] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %DerivedPublic.ref: type = name_ref DerivedPublic, imports.%DerivedPublic.decl [concrete = constants.%DerivedPublic]
// CHECK:STDOUT:   %foo.ref: %.5a6 = name_ref foo, imports.%.607 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Base.foo.call: init %empty_tuple.type = call imports.%Base.foo.decl()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
