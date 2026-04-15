// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/reverse/class.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/reverse/class.carbon

// --- static_members.carbon
library "[[@TEST_NAME]]";

import Cpp;

class A {
  fn F() {}
}

inline Cpp '''
void G() {
  Carbon::A::F();
}
''';

// --- nested.carbon
library "[[@TEST_NAME]]";

import Cpp;

class A {
  class B {
    class C {
    }
  }
}

inline Cpp '''
void G() {
  Carbon::A::B::C *p = nullptr;
}
''';

// --- fail_incomplete_concrete.carbon
library "[[@TEST_NAME]]";

import Cpp;

// CHECK:STDERR: fail_incomplete_concrete.carbon:[[@LINE+4]]:1: error: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class A;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
class A;

inline Cpp '''
// CHECK:STDERR: fail_incomplete_concrete.carbon:[[@LINE+4]]:11: error: variable has incomplete type 'Carbon::A' [CppInteropParseError]
// CHECK:STDERR:    16 | Carbon::A a;
// CHECK:STDERR:       |           ^
// CHECK:STDERR:
Carbon::A a;
''';

// --- fail_todo_monomorphization_failure.carbon
library "[[@TEST_NAME]]";

import Cpp;

class B(N:! i32) {
  var a: array(i8, N);
}

// TODO: Once we allow this, we should produce an "invalid array bound" error
// during monomorphization triggered by the C++ code.
// CHECK:STDERR: fail_todo_monomorphization_failure.carbon:[[@LINE+4]]:11: error: alias initializer must be a name reference [AliasRequiresNameRef]
// CHECK:STDERR: alias T = B(-1);
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR:
alias T = B(-1);

inline Cpp '''
// CHECK:STDERR: fail_todo_monomorphization_failure.carbon:[[@LINE+4]]:9: error: no type named 'T' in namespace 'Carbon' [CppInteropParseError]
// CHECK:STDERR:    22 | Carbon::T t;
// CHECK:STDERR:       | ~~~~~~~~^
// CHECK:STDERR:
Carbon::T t;
''';

// --- fail_todo_fields.carbon
library "[[@TEST_NAME]]";

import Cpp;

class A {
  var a: i32;
  var b: i32;
}

inline Cpp '''
// CHECK:STDERR: fail_todo_fields.carbon:[[@LINE+7]]:15: error: static assertion failed due to requirement 'sizeof(Carbon::A) == 2 * sizeof(int)' [CppInteropParseError]
// CHECK:STDERR:    18 | static_assert(sizeof(Carbon::A) == 2 * sizeof(int));
// CHECK:STDERR:       |               ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_fields.carbon:[[@LINE+4]]:33: note: expression evaluates to '1 == 8' [CppInteropParseNote]
// CHECK:STDERR:    18 | static_assert(sizeof(Carbon::A) == 2 * sizeof(int));
// CHECK:STDERR:       |               ~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
static_assert(sizeof(Carbon::A) == 2 * sizeof(int));
''';

// --- alias_identity.carbon
library "[[@TEST_NAME]]";

import Cpp;

class A {}
alias B = A;

inline Cpp '''
// OK, Carbon::A and Carbon::B are the same type.
Carbon::A *pa;
Carbon::B *pb = pa;
''';
