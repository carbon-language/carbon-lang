// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --share-cpp-ast
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/basics/share_ast.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/basics/share_ast.carbon

// --- a.h
#ifndef A_H
#define A_H
inline int GetA() { return 42; }
#endif

// --- b.h
#ifndef B_H
#define B_H
#include "a.h"
inline int GetB() { return 100; }
#endif

// --- a.carbon
library "[[@TEST_NAME]]";
import Cpp library "a.h";

fn CallA() -> i32 {
  return Cpp.GetA();
}

// --- fail_a_does_not_leak.carbon
library "[[@TEST_NAME]]";
import Cpp;

fn CallLeakedA() -> i32 {
  // This should not be found as we have not imported "a.h".
  // CHECK:STDERR: fail_a_does_not_leak.carbon:[[@LINE+4]]:10: error: member name `GetA` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   return Cpp.GetA();
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR:
  return Cpp.GetA();
}

// --- a_again.carbon
library "[[@TEST_NAME]]";
import Cpp library "a.h";

fn CallAAgain() -> i32 {
  // This should be accepted as we have imported "a.h", even though we will just reuse the version from `a.carbon`.
  return Cpp.GetA();
}

// --- b.carbon
library "[[@TEST_NAME]]";
import Cpp library "b.h";

fn CallAIndirect() -> i32 {
  return Cpp.GetA();
}

fn CallB() -> i32 {
  return Cpp.GetB();
}

// --- pkg_a.carbon
package PkgA;
import Cpp inline '''
template <typename T> void Accept(T) {}
''';

class SameName {}
alias TypeAlias = i32;
namespace N;
class N.SameNested {}

inline Cpp '''
void CheckPkgA(Carbon::SameName, Carbon::N::SameNested) {
  Carbon::TypeAlias x = 0;
  static_assert(sizeof(x) == 4);
}
''';

fn TestPkgA(s: SameName, n: N.SameNested) {
  Cpp.Accept(s);
  Cpp.CheckPkgA(s, n);
}

// --- pkg_b.carbon
package PkgB;
import Cpp inline '''
template <typename T> void Accept(T) {}
''';

class SameName {}
alias TypeAlias = i64;
namespace N;
class N.SameNested {}

inline Cpp '''
void CheckPkgB(Carbon::SameName, Carbon::N::SameNested) {
  Carbon::TypeAlias x = 0;
  static_assert(sizeof(x) == 8);
}
''';

fn TestPkgB(s: SameName, n: N.SameNested) {
  Cpp.Accept(s);
  Cpp.CheckPkgB(s, n);
}

// --- use_both_packages.carbon
library "[[@TEST_NAME]]";
import PkgA;
import PkgB;
import Cpp inline '''
// SameName and N::SameNested in PkgA and PkgB have distinct C++ DeclContexts
// (Carbon::PkgA vs Carbon::PkgB), so overloading on them in C++ succeeds.
void OverloadByPackage(Carbon::PkgA::SameName*) {}
void OverloadByPackage(Carbon::PkgB::SameName*) {}
void OverloadNestedByPackage(Carbon::PkgA::N::SameNested*) {}
void OverloadNestedByPackage(Carbon::PkgB::N::SameNested*) {}
''';

// --- export_in_c.carbon
library "[[@TEST_NAME]]";
import Cpp;

class ExportedClass {}
namespace N;
class N.ExportedNested {}

inline Cpp '''
using UseExported = Carbon::ExportedClass;
using UseNested = Carbon::N::ExportedNested;
''';

// --- fail_carbon_export_does_not_leak.carbon
library "[[@TEST_NAME]]";
import Cpp;

namespace N;

inline Cpp '''
// CHECK:STDERR: fail_carbon_export_does_not_leak.carbon:[[@LINE+4]]:23: error: no type named 'ExportedClass' in namespace 'Carbon' [CppInteropParseError]
// CHECK:STDERR:    11 | using Leak1 = Carbon::ExportedClass;
// CHECK:STDERR:       |               ~~~~~~~~^
// CHECK:STDERR:
using Leak1 = Carbon::ExportedClass;
// CHECK:STDERR: fail_carbon_export_does_not_leak.carbon:[[@LINE+4]]:26: error: no type named 'ExportedNested' in namespace 'Carbon::N' [CppInteropParseError]
// CHECK:STDERR:    16 | using Leak2 = Carbon::N::ExportedNested;
// CHECK:STDERR:       |               ~~~~~~~~~~~^
// CHECK:STDERR:
using Leak2 = Carbon::N::ExportedNested;
// CHECK:STDERR: fail_carbon_export_does_not_leak.carbon:[[@LINE+4]]:23: error: no member named 'PkgA' in namespace 'Carbon' [CppInteropParseError]
// CHECK:STDERR:    21 | using Leak3 = Carbon::PkgA::SameName;
// CHECK:STDERR:       |               ~~~~~~~~^
// CHECK:STDERR:
using Leak3 = Carbon::PkgA::SameName;
''';
