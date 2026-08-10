// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/export/class.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/export/class.carbon

// --- static_members.carbon
library "[[@TEST_NAME]]";

import Cpp;

class StaticMembers {
  fn F() {}
}

inline Cpp '''
void CallStaticMember() {
  Carbon::StaticMembers::F();
}
''';

// --- nested.carbon
library "[[@TEST_NAME]]";

import Cpp;

class OuterA {
  class InnerB {
    class NestedC {
    }
  }
}

inline Cpp '''
void TestNested() {
  Carbon::OuterA::InnerB::NestedC *nested_p = nullptr;
}
''';

// --- incomplete.carbon
library "[[@TEST_NAME]]";

import Cpp;

class IncompleteA;

inline Cpp '''
Carbon::IncompleteA *incomplete_p = nullptr;
''';

// --- fail_incomplete_concrete.carbon
library "[[@TEST_NAME]]";

import Cpp;

// CHECK:STDERR: fail_incomplete_concrete.carbon:[[@LINE+2]]:1-26: error: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR:
class IncompleteConcreteA;

inline Cpp '''
// CHECK:STDERR: fail_incomplete_concrete.carbon:[[@LINE+3]]:29-49: error: variable has incomplete type 'Carbon::IncompleteConcreteA' [CppInteropParseError]
// CHECK:STDERR: fail_incomplete_concrete.carbon:[[@LINE-4]]:26: note: forward declaration of 'Carbon::IncompleteConcreteA' [CppInteropParseNote]
// CHECK:STDERR:
Carbon::IncompleteConcreteA incomplete_concrete_a;
''';

// --- fail_todo_monomorphization_failure.carbon
library "[[@TEST_NAME]]";

import Cpp;

// CHECK:STDERR: fail_todo_monomorphization_failure.carbon:[[@LINE+2]]:1-17: error: semantics TODO: `binding maps to a non-type template parameter` [SemanticsTodo]
// CHECK:STDERR:
class B(N: i32) {
  var a: array(i8, N);
}

// TODO: Once we allow this, we should produce an "invalid array bound" error
// during monomorphization triggered by the C++ code.
alias T = B(-1);

inline Cpp '''
// CHECK:STDERR: fail_todo_monomorphization_failure.carbon:[[@LINE+4]]:1: error: semantics TODO: `interop with unsupported type` [SemanticsTodo]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_monomorphization_failure.carbon:[[@LINE+2]]:9: error: no type named 'T' in namespace 'Carbon' [CppInteropParseError]
// CHECK:STDERR:
Carbon::T t;
''';
