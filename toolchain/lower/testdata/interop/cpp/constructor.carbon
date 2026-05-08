// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/constructor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/constructor.carbon

// ============================================================================
// Default constructor
// ============================================================================

// --- default.h

class C {
 public:
  C() : x_(8), y_(9) {}

 private:
  int x_;
  int y_;
};

// --- import_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "default.h";

fn F() {
  let _: Cpp.C = Cpp.C.C();
}

fn G() {
  // CHECK:STDERR: import_default.carbon:[[@LINE+8]]:18: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   let _: Cpp.C = ();
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR: import_default.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./default.h:4:3: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR:   C() : x_(8), y_(9) {}
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  let _: Cpp.C = ();
}

// --- implicit_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "default.h";

fn F() {
  var _: Cpp.C;
}

// --- defaulted_default.h

struct ImplicitlyDefaultedDefault {
  int x_ = 1;
};

struct ExplicitlyDefaultedDefault {
  ExplicitlyDefaultedDefault() = default;
  int x_ = 1;
};

// --- defaulted_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "defaulted_default.h";

fn F() {
  let _: Cpp.ImplicitlyDefaultedDefault =
      Cpp.ImplicitlyDefaultedDefault.ImplicitlyDefaultedDefault();
  var _: Cpp.ImplicitlyDefaultedDefault;

  let _: Cpp.ExplicitlyDefaultedDefault =
      Cpp.ExplicitlyDefaultedDefault.ExplicitlyDefaultedDefault();
  var _: Cpp.ExplicitlyDefaultedDefault;
}

// ============================================================================
// Copy constructor
// ============================================================================

// --- copy.h

class Copy {
 public:
  Copy();
  Copy(const Copy&);

 private:
  int n;
};

// --- call_copy.carbon

library "[[@TEST_NAME]]";

import Cpp library "copy.h";

fn Copy(c: Cpp.Copy) -> Cpp.Copy {
  return c;
}

// ============================================================================
// C1 constructor
// ============================================================================

// --- header.h

struct Base { };
struct Derived: virtual Base {
};

// --- call_c1_in_thunk.carbon

library "[[@TEST_NAME]]";

import Cpp library "header.h";

fn F() {
  let _: Cpp.Derived = Cpp.Derived.Derived();
}

// ============================================================================
// Multiple arguments
// ============================================================================

// --- multi_argument.h

class C {
 public:
  C() {}
  C(int) {}
  C(int, int) {}
  C(int, int, int, int = 0) {}
};

// --- call_multi_argument.carbon

library "[[@TEST_NAME]]";

import Cpp library "multi_argument.h";

fn Zero() {
  let _: Cpp.C = Cpp.C.C();
  // CHECK:STDERR: call_multi_argument.carbon:[[@LINE+8]]:18: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   let _: Cpp.C = ();
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR: call_multi_argument.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./multi_argument.h:4:3: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR:   C() {}
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  let _: Cpp.C = ();
}

fn One() {
  let _: Cpp.C = Cpp.C.C(1);
  let _: Cpp.C = 2;
  // CHECK:STDERR: call_multi_argument.carbon:[[@LINE+11]]:18: error: cannot implicitly convert expression of type `(Core.IntLiteral,)` to `i32` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.C = (3,);
  // CHECK:STDERR:                  ^~~~
  // CHECK:STDERR: call_multi_argument.carbon:[[@LINE+8]]:18: note: type `(Core.IntLiteral,)` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: Cpp.C = (3,);
  // CHECK:STDERR:                  ^~~~
  // CHECK:STDERR: call_multi_argument.carbon:[[@LINE-24]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./multi_argument.h:5:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   C(int) {}
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  let _: Cpp.C = (3,);
}

fn Two() {
  let _: Cpp.C = Cpp.C.C(1, 2);
  // CHECK:STDERR: call_multi_argument.carbon:[[@LINE+8]]:18: error: 1 argument passed to function expecting 2 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   let _: Cpp.C = (3, 4);
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR: call_multi_argument.carbon:[[@LINE-37]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./multi_argument.h:6:3: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR:   C(int, int) {}
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  let _: Cpp.C = (3, 4);
}

fn Three() {
  let _: Cpp.C = Cpp.C.C(1, 2, 3);
  // CHECK:STDERR: call_multi_argument.carbon:[[@LINE+8]]:18: error: 1 argument passed to function expecting 3 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   let _: Cpp.C = (4, 5, 6);
  // CHECK:STDERR:                  ^~~~~~~~~
  // CHECK:STDERR: call_multi_argument.carbon:[[@LINE-50]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./multi_argument.h:7:3: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR:   C(int, int, int, int = 0) {}
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  let _: Cpp.C = (4, 5, 6);
}

fn Four() {
  let _: Cpp.C = Cpp.C.C(1, 2, 3, 4);
  // CHECK:STDERR: call_multi_argument.carbon:[[@LINE+8]]:18: error: 1 argument passed to function expecting 4 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   let _: Cpp.C = (5, 6, 7, 8);
  // CHECK:STDERR:                  ^~~~~~~~~~~~
  // CHECK:STDERR: call_multi_argument.carbon:[[@LINE-63]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./multi_argument.h:7:3: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR:   C(int, int, int, int = 0) {}
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  let _: Cpp.C = (5, 6, 7, 8);
}

