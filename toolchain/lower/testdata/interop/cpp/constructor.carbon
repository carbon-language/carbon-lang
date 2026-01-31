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
  let _: Cpp.C = ();
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
  let _: Cpp.C = ();
  // CHECK:STDERR: call_multi_argument.carbon:[[@LINE+7]]:18: error: cannot implicitly convert expression of type `{}` to `Cpp.C` [ConversionFailure]
  // CHECK:STDERR:   let _: Cpp.C = {};
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR: call_multi_argument.carbon:[[@LINE+4]]:18: note: type `{}` does not implement interface `Core.ImplicitAs(Cpp.C)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let _: Cpp.C = {};
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR:
  let _: Cpp.C = {};
}

fn One() {
  let _: Cpp.C = Cpp.C.C(1);
  let _: Cpp.C = 2;
  let _: Cpp.C = (3,);
}

fn Two() {
  let _: Cpp.C = Cpp.C.C(1, 2);
  let _: Cpp.C = (3, 4);
}

fn Three() {
  let _: Cpp.C = Cpp.C.C(1, 2, 3);
  let _: Cpp.C = (4, 5, 6);
}

fn Four() {
  let _: Cpp.C = Cpp.C.C(1, 2, 3, 4);
  let _: Cpp.C = (5, 6, 7, 8);
}

