// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/roundtrip.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/roundtrip.carbon

// --- carbon_to_cpp_to_carbon.carbon
library "[[@TEST_NAME]]";

import Cpp;

class C {}

inline Cpp '''
using CAlias = Carbon::C;
''';

fn F(p: C*) {
  let unused q: Cpp.CAlias* = p;
}

// --- cpp_to_carbon_to_cpp.carbon
library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''
class C {};
''';

alias CAlias = Cpp.C;

inline Cpp '''
auto F(C *p) -> Carbon::CAlias* {
  return p;
}
''';

// --- import.carbon

library "[[@TEST_NAME]]";

import Cpp;

// CHECK:STDERR: import.carbon:[[@LINE+4]]:1: error: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Class1;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
class Class1;

inline Cpp '''
void func(Carbon::Class1* _Nonnull);
''';

fn Caller(p: Class1*) {
  // CHECK:STDERR: import.carbon:[[@LINE+7]]:13: error: no matching function for call to 'func' [CppInteropParseError]
  // CHECK:STDERR:    13 |   Cpp.func(p);
  // CHECK:STDERR:       |             ^
  // CHECK:STDERR: import.carbon:[[@LINE-7]]:6: note: candidate function not viable: cannot convert argument of incomplete type 'Carbon::Class1 * _Nonnull' to 'Carbon::Class1 * _Nonnull' (aka 'Carbon::Class1 *') for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     9 | void func(Carbon::Class1* _Nonnull);
  // CHECK:STDERR:       |      ^    ~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.func(p);
}
