// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/namespace/roundtrip.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/namespace/roundtrip.carbon

// --- carbon_to_cpp_to_carbon.carbon
library "[[@TEST_NAME]]";

import Cpp;

namespace N;

inline Cpp '''
namespace M = Carbon::N;
''';

class C {}

fn N.F(c: C);

fn G(c: C) {
  Cpp.M.F(c);
}

// --- cpp_to_carbon_to_cpp.carbon
library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''
namespace N {}
''';

alias M = Cpp.N;

inline Cpp '''
namespace N { void F(); }

static_assert(&N::F == &Carbon::M::F);
''';

// --- cpp_carbon.carbon
library "[[@TEST_NAME]]";

import Cpp;

fn F();

fn G() {
  // TODO: Should we disallow this?
  Cpp.Carbon.F();
}

// --- fail_carbon_cpp.carbon
library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''
void F();

void G() {
  // This is not found because it looks for the identifier `Cpp`, not the keyword `Cpp`.
  // CHECK:STDERR: fail_carbon_cpp.carbon:[[@LINE+4]]:11: error: no member named 'Cpp' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:    14 |   Carbon::Cpp::F();
  // CHECK:STDERR:       |   ~~~~~~~~^
  // CHECK:STDERR:
  Carbon::Cpp::F();
}
''';

// --- fail_carbon_raw_cpp.carbon
library "[[@TEST_NAME]]";

import Cpp;

// CHECK:STDERR: fail_carbon_raw_cpp.carbon:[[@LINE+4]]:7: error: semantics TODO: `interop with non-identifier package name` [SemanticsTodo]
// CHECK:STDERR: alias r#Cpp = Cpp;
// CHECK:STDERR:       ^~~
// CHECK:STDERR:
alias r#Cpp = Cpp;

inline Cpp '''
void F();

void G() {
  // CHECK:STDERR: fail_carbon_raw_cpp.carbon:[[@LINE+4]]:11: error: no member named 'Cpp' in namespace 'Carbon' [CppInteropParseError]
  // CHECK:STDERR:    19 |   Carbon::Cpp::F();
  // CHECK:STDERR:       |   ~~~~~~~~^
  // CHECK:STDERR:
  Carbon::Cpp::F();
}
''';
