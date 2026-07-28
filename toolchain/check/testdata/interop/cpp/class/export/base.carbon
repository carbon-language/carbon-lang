// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/export/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/export/base.carbon

// --- derived_to_base.carbon
library "[[@TEST_NAME]]";

import Cpp;

base class A {}

class B {
  extend base: A;
}

inline Cpp '''
Carbon::A* Convert(Carbon::B* p) {
  return p;
}
''';

// --- name_lookup.carbon
library "[[@TEST_NAME]]";

import Cpp;

base class A {
  fn F();
  fn G(n: i32);
}

class B {
  extend base: A;
  fn G();
}

inline Cpp '''
void Qualified() {
  Carbon::B::F();
  Carbon::B::G();
}

void MemberAccess(Carbon::A *a, Carbon::B *b) {
  b->F();
  b->G();
}
''';

// --- fail_derive_from_final.carbon
library "[[@TEST_NAME]]";

import Cpp;

class Final {}

inline Cpp '''
// CHECK:STDERR: fail_derive_from_final.carbon:[[@LINE+7]]:12: error: base 'Final' is marked 'final' [CppInteropParseError]
// CHECK:STDERR:    15 | struct A : Carbon::Final {};
// CHECK:STDERR:       |            ^
// CHECK:STDERR: fail_derive_from_final.carbon:[[@LINE-6]]:13: note: 'Final' declared here [CppInteropParseNote]
// CHECK:STDERR:     5 | class Final {}
// CHECK:STDERR:       |             ^
// CHECK:STDERR:
struct A : Carbon::Final {};
''';

// --- fail_abstract.carbon
library "[[@TEST_NAME]]";

import Cpp;

// TODO: Add a way to give a class a virtual destructor without C++ interop.
inline Cpp '''
struct VirtualDestructor {
  virtual ~VirtualDestructor() {}
};
''';

// CHECK:STDERR: fail_abstract.carbon:[[@LINE+4]]:1: error: cannot access member of interface `Core.Destroy` in type `Abstract` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: abstract class Abstract {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
abstract class Abstract {
  extend base: Cpp.VirtualDestructor;
}

inline Cpp '''
// CHECK:STDERR: fail_abstract.carbon:[[@LINE+7]]:18: error: variable type 'Carbon::Abstract' is an abstract class [CppInteropParseError]
// CHECK:STDERR:    28 | Carbon::Abstract x;
// CHECK:STDERR:       |                  ^
// CHECK:STDERR: fail_abstract.carbon:[[@LINE-8]]:25: note: unimplemented pure virtual method '~Abstract' in 'Abstract' [CppInteropParseNote]
// CHECK:STDERR:    16 | abstract class Abstract {
// CHECK:STDERR:       |                         ^
// CHECK:STDERR:
Carbon::Abstract x;
''';

// --- fail_todo_abstract_nonvirtual_dtor.carbon
library "[[@TEST_NAME]]";

import Cpp;

// TODO: Find a way to export this to C++ as an abstract class, despite not
// having a vptr.
// CHECK:STDERR: fail_todo_abstract_nonvirtual_dtor.carbon:[[@LINE+8]]:1: error: semantics TODO: `exporting abstract class with no abstract methods and non-virtual destructor to C++` [SemanticsTodo]
// CHECK:STDERR: abstract class Abstract {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_abstract_nonvirtual_dtor.carbon:[[@LINE+4]]:1: error: cannot access member of interface `Core.Destroy` in type `Abstract` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: abstract class Abstract {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
abstract class Abstract {}

inline Cpp '''
class Derived : public Carbon::Abstract {};
''';

// --- fail_todo_abstract_nonvirtual_dtor_but_virtual_fns.carbon
library "[[@TEST_NAME]]";

import Cpp;

// TODO: We should support generating a base subobject destructor for an
// abstract class, even though we refuse to generate a complete object
// destructor.
// CHECK:STDERR: fail_todo_abstract_nonvirtual_dtor_but_virtual_fns.carbon:[[@LINE+4]]:1: error: cannot access member of interface `Core.Destroy` in type `Abstract` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: abstract class Abstract {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
abstract class Abstract {
  abstract fn F(self);
}

inline Cpp '''
class Derived : public Carbon::Abstract {
};
''';

// --- fail_abstract_method.carbon
library "[[@TEST_NAME]]";

import Cpp;

// CHECK:STDERR: fail_abstract_method.carbon:[[@LINE+4]]:1: error: cannot access member of interface `Core.Destroy` in type `Abstract` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: abstract class Abstract {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
abstract class Abstract {
  abstract fn F(self);
}

inline Cpp '''
// CHECK:STDERR: fail_abstract_method.carbon:[[@LINE+7]]:18: error: variable type 'Carbon::Abstract' is an abstract class [CppInteropParseError]
// CHECK:STDERR:    21 | Carbon::Abstract x;
// CHECK:STDERR:       |                  ^
// CHECK:STDERR: fail_abstract_method.carbon:[[@LINE-7]]:22: note: unimplemented pure virtual method 'F' in 'Abstract' [CppInteropParseNote]
// CHECK:STDERR:    10 |   abstract fn F(self);
// CHECK:STDERR:       |                      ^
// CHECK:STDERR:
Carbon::Abstract x;
''';

// --- fail_todo_abstract_method_overridden_in_cpp.carbon
library "[[@TEST_NAME]]";

import Cpp;

// CHECK:STDERR: fail_todo_abstract_method_overridden_in_cpp.carbon:[[@LINE+4]]:1: error: cannot access member of interface `Core.Destroy` in type `Abstract` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: abstract class Abstract {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
abstract class Abstract {
  abstract fn F(ref self);
}

// A C++ class that overrides the abstract method can be instantiated.
inline Cpp '''
struct Derived : Carbon::Abstract {
  void F() & override {}
};
Derived d;
''';

// --- override_virtual_dtor.carbon
library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''
struct A {
  virtual ~A() = 0;
};
''';

class B {
  extend base: Cpp.A;
}

inline Cpp '''
Carbon::B b;
''';
