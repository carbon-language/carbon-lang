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

base class LookupBase {
  fn F();
  fn G(n: i32);
}

class LookupDerived {
  extend base: LookupBase;
  fn G();
}

inline Cpp '''
void Qualified() {
  Carbon::LookupDerived::F();
  Carbon::LookupDerived::G();
}

void MemberAccess(Carbon::LookupBase *a, Carbon::LookupDerived *b) {
  b->F();
  b->G();
}
''';

// --- fail_derive_from_final.carbon
library "[[@TEST_NAME]]";

import Cpp;

class FinalClass {}

inline Cpp '''
// CHECK:STDERR: fail_derive_from_final.carbon:[[@LINE+3]]:12-17: error: base 'FinalClass' is marked 'final' [CppInteropParseError]
// CHECK:STDERR: fail_derive_from_final.carbon:[[@LINE-4]]:18: note: 'FinalClass' declared here [CppInteropParseNote]
// CHECK:STDERR:
struct A : Carbon::FinalClass {};
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

// CHECK:STDERR: fail_abstract.carbon:[[@LINE+2]]:1-34: error: cannot access member of interface `Core.Destroy` in type `AbstractWithVdtor` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR:
abstract class AbstractWithVdtor {
  extend base: Cpp.VirtualDestructor;
}

inline Cpp '''
// CHECK:STDERR: fail_abstract.carbon:[[@LINE+3]]:27: error: variable type 'Carbon::AbstractWithVdtor' is an abstract class [CppInteropParseError]
// CHECK:STDERR: fail_abstract.carbon:[[@LINE-6]]:34: note: unimplemented pure virtual method '~AbstractWithVdtor' in 'AbstractWithVdtor' [CppInteropParseNote]
// CHECK:STDERR:
Carbon::AbstractWithVdtor x;
''';

// --- fail_todo_abstract_nonvirtual_dtor.carbon
library "[[@TEST_NAME]]";

import Cpp;

// TODO: Find a way to export this to C++ as an abstract class, despite not
// having a vptr.
// CHECK:STDERR: fail_todo_abstract_nonvirtual_dtor.carbon:[[@LINE+4]]:1-39: error: semantics TODO: `exporting abstract class with no abstract methods and non-virtual destructor to C++` [SemanticsTodo]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_abstract_nonvirtual_dtor.carbon:[[@LINE+2]]:1-39: error: cannot access member of interface `Core.Destroy` in type `AbstractNonvirtualDtor` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR:
abstract class AbstractNonvirtualDtor {}

inline Cpp '''
class DerivedNonvirtualDtor : public Carbon::AbstractNonvirtualDtor {};
''';

// --- fail_todo_abstract_nonvirtual_dtor_but_virtual_fns.carbon
library "[[@TEST_NAME]]";

import Cpp;

// TODO: We should support generating a base subobject destructor for an
// abstract class, even though we refuse to generate a complete object
// destructor.
// CHECK:STDERR: fail_todo_abstract_nonvirtual_dtor_but_virtual_fns.carbon:[[@LINE+2]]:1-41: error: cannot access member of interface `Core.Destroy` in type `AbstractWithMethodNoVptr` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR:
abstract class AbstractWithMethodNoVptr {
  abstract fn F(self);
}

inline Cpp '''
class DerivedWithMethodNoVptr : public Carbon::AbstractWithMethodNoVptr {
};
''';

// --- fail_variable_with_abstract_method.carbon
library "[[@TEST_NAME]]";

import Cpp;

// TODO: We should support generating a base subobject destructor for an
// abstract class, even though we refuse to generate a complete object
// destructor.
// CHECK:STDERR: fail_variable_with_abstract_method.carbon:[[@LINE+2]]:1-35: error: cannot access member of interface `Core.Destroy` in type `AbstractWithMethod` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR:
abstract class AbstractWithMethod {
  abstract fn F(self);
}

inline Cpp '''
// CHECK:STDERR: fail_variable_with_abstract_method.carbon:[[@LINE+3]]:28-37: error: variable type 'Carbon::AbstractWithMethod' is an abstract class [CppInteropParseError]
// CHECK:STDERR: fail_variable_with_abstract_method.carbon:[[@LINE-5]]:22: note: unimplemented pure virtual method 'F' in 'AbstractWithMethod' [CppInteropParseNote]
// CHECK:STDERR:
Carbon::AbstractWithMethod x_abstract;
''';

// --- fail_todo_abstract_method_overridden_in_cpp.carbon
library "[[@TEST_NAME]]";

import Cpp;

// TODO: We should support generating a base subobject destructor for an
// abstract class, even though we refuse to generate a complete object
// destructor.
// CHECK:STDERR: fail_todo_abstract_method_overridden_in_cpp.carbon:[[@LINE+2]]:1-35: error: cannot access member of interface `Core.Destroy` in type `AbstractToOverride` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR:
abstract class AbstractToOverride {
  abstract fn F(ref self);
}

// A C++ class that overrides the abstract method can be instantiated.
inline Cpp '''
struct DerivedOverriding : Carbon::AbstractToOverride {
  void F() & override {}
};
DerivedOverriding d;
''';

// --- override_virtual_dtor.carbon
library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''
struct BaseWithPureVdtor {
  virtual ~BaseWithPureVdtor() = 0;
};
''';

class DerivedWithVdtor {
  extend base: Cpp.BaseWithPureVdtor;
}

inline Cpp '''
Carbon::DerivedWithVdtor b;
''';
