// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/namespace/export.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/namespace/export.carbon

// --- other.carbon
package Other;
namespace Nested;
namespace Nested.Again;

class Class1;
// --- namespace.carbon

library "[[@TEST_NAME]]";

import Other;
import Cpp inline '''
namespace X = Carbon::Other;
namespace Y = Carbon::Other::Nested;
namespace Z = Carbon::Other::Nested::Again;
Carbon::Other::Class1 *glbl;

template<typename T> void f1() {
  T* v;
}

void f2() {
  f1<Carbon::Other::Class1>();
}

''';

// --- alias_identity.carbon

import Cpp;

namespace M;
alias N = M;

class M.C {}

inline Cpp '''
// OK, valid redeclaration: M and N are the same namespace.
namespace A = Carbon::M;
namespace A = Carbon::N;

// Valid: M::C and N::C are the same class.
Carbon::M::C *pmc;
Carbon::N::C *pnc = pmc;
''';

// --- fail_current_package_not_in_carbon.carbon
package SelfPkg;
import Cpp;

class C {}

inline Cpp '''
Carbon::C* p;
// CHECK:STDERR: fail_current_package_not_in_carbon.carbon:[[@LINE+4]]:25: error: expected namespace name [CppInteropParseError]
// CHECK:STDERR:    12 | namespace Bad = Carbon::SelfPkg;
// CHECK:STDERR:       |                 ~~~~~~~~^
// CHECK:STDERR:
namespace Bad = Carbon::SelfPkg;
''';

// --- fail_main_package_not_in_carbon.carbon
library "[[@TEST_NAME]]";
import Cpp;

class C {}

inline Cpp '''
Carbon::C* p;
// CHECK:STDERR: fail_main_package_not_in_carbon.carbon:[[@LINE+4]]:25: error: expected namespace name [CppInteropParseError]
// CHECK:STDERR:    12 | namespace Bad = Carbon::Main;
// CHECK:STDERR:       |                 ~~~~~~~~^
// CHECK:STDERR:
namespace Bad = Carbon::Main;
''';
