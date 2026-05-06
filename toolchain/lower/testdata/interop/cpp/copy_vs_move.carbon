// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// EXTRA-ARGS: --clang-arg=-fno-exceptions
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/copy_vs_move.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/copy_vs_move.carbon

// --- copyable.carbon
library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
struct Copyable {
  Copyable(const Copyable&);
  Copyable& operator=(const Copyable&);
  ~Copyable();
};
Copyable make();
void pass(Copyable);
''';

var c: Cpp.Copyable = Cpp.make();

fn PassGlobal() {
  Cpp.pass(c);
}

fn PassTemporary() {
  Cpp.pass(Cpp.make());
}

fn PassValue(c: Cpp.Copyable) {
  Cpp.pass(c);
}

fn PassRef(ref c: Cpp.Copyable) {
  Cpp.pass(c);
}

// --- copy_or_move.carbon
library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
struct CopyOrMove {
  CopyOrMove(const CopyOrMove&);
  CopyOrMove& operator=(const CopyOrMove&);
  CopyOrMove(CopyOrMove&&);
  CopyOrMove& operator=(CopyOrMove&&);
  ~CopyOrMove();
};
CopyOrMove make();
void pass(CopyOrMove);
''';

var c: Cpp.CopyOrMove = Cpp.make();

fn PassGlobal() {
  Cpp.pass(c);
}

fn PassTemporary() {
  Cpp.pass(Cpp.make());
}

fn PassValue(c: Cpp.CopyOrMove) {
  Cpp.pass(c);
}

fn PassRef(ref c: Cpp.CopyOrMove) {
  Cpp.pass(c);
}

// --- move_only.carbon
library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
namespace std {
template<typename T> T&& move(T& t) { return static_cast<T&&>(t); }
}

struct MoveOnly {
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&);
  MoveOnly& operator=(MoveOnly&&);
  ~MoveOnly();
};
MoveOnly make();
// CHECK:STDERR: move_only.carbon:[[@LINE+9]]:6: error: call to deleted constructor of 'MoveOnly' [CppInteropParseError]
// CHECK:STDERR:    26 | void pass(MoveOnly);
// CHECK:STDERR:       |      ^~~~
// CHECK:STDERR: move_only.carbon:[[@LINE-10]]:3: note: 'MoveOnly' has been explicitly marked deleted here [CppInteropParseNote]
// CHECK:STDERR:    10 |   MoveOnly(const MoveOnly&) = delete;
// CHECK:STDERR:       |   ^
// CHECK:STDERR: move_only.carbon:[[@LINE+3]]:19: note: passing argument to parameter here [CppInteropParseNote]
// CHECK:STDERR:    26 | void pass(MoveOnly);
// CHECK:STDERR:       |                   ^
void pass(MoveOnly);
''';

var c: Cpp.MoveOnly = Cpp.make();

fn PassGlobalByMove() {
  // CHECK:STDERR: move_only.carbon:[[@LINE+4]]:3: note: in thunk for C++ function used here [InCppThunk]
  // CHECK:STDERR:   Cpp.pass(Cpp.std.move(ref c));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.pass(Cpp.std.move(ref c));
}

fn PassTemporary() {
  Cpp.pass(Cpp.make());
}

fn PassValue(c: Cpp.MoveOnly) {
  Cpp.pass(c);
}

fn PassRef(ref c: Cpp.MoveOnly) {
  Cpp.pass(c);
}

