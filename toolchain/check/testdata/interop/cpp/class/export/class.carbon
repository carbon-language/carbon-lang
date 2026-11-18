// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
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

// CHECK:STDERR:        ╭─┤ fail_incomplete_concrete.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ class IncompleteConcreteA;
// CHECK:STDERR:        · ━━━━━━━━━━━━━┳━━━━━━━━━━━━
// CHECK:STDERR:        ·              ┗━━┫ error: class was forward declared here
// CHECK:STDERR:        ·                 ┃        [ClassForwardDeclaredHere]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
class IncompleteConcreteA;

inline Cpp '''
// CHECK:STDERR:        ╭─┤ fail_incomplete_concrete.carbon:[[@LINE+15]]:29
// CHECK:STDERR:        │ class IncompleteConcreteA;
// CHECK:STDERR:        ·                          ┬
// CHECK:STDERR:        ·                          ╰──┤ forward declaration of
// CHECK:STDERR:        ·                             │ 'Carbon::Main::IncompleteConcreteA'
// CHECK:STDERR:        ·                             │ [CppInteropParseNote]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ Carbon::IncompleteConcreteA incomplete_concrete_a;
// CHECK:STDERR:        ·                             ┳
// CHECK:STDERR:        ·                             ┗━━┫ error: variable has incomplete type
// CHECK:STDERR:        ·                                ┃        'Carbon::IncompleteConcreteA' (aka
// CHECK:STDERR:        ·                                ┃        'Carbon::Main::IncompleteConcreteA')
// CHECK:STDERR:        ·                                ┃        [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
Carbon::IncompleteConcreteA incomplete_concrete_a;
''';

// --- fail_todo_monomorphization_failure.carbon
library "[[@TEST_NAME]]";

import Cpp;

// CHECK:STDERR:        ╭─┤ fail_todo_monomorphization_failure.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ class B(N: i32) {
// CHECK:STDERR:        · ━━━━━┳━━━━━━━━━━━
// CHECK:STDERR:        ·      ┗━━┫ error: semantics TODO: `binding maps to a non-type template parameter`
// CHECK:STDERR:        ·         ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
class B(N: i32) {
  var a: array(i8, N);
}

// TODO: Once we allow this, we should produce an "invalid array bound" error
// during monomorphization triggered by the C++ code.
alias T = B(-1);

inline Cpp '''
// CHECK:STDERR:        ╭─┤ fail_todo_monomorphization_failure.carbon:[[@LINE+13]]:1
// CHECK:STDERR: ->     │ Carbon::T t;
// CHECK:STDERR:        · ┳
// CHECK:STDERR:        · ┗━━┫ error: semantics TODO: `interop with unsupported type` [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_monomorphization_failure.carbon:[[@LINE+7]]:9
// CHECK:STDERR: ->     │ Carbon::T t;
// CHECK:STDERR:        · ───────╴╻
// CHECK:STDERR:        ·         ┗━━┫ error: no type named 'T' in namespace 'Carbon'
// CHECK:STDERR:        ·            ┃        [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
Carbon::T t;
''';
