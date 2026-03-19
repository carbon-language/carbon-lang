// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// EXTRA-ARGS: --target=aarch64-unknown-linux --clang-arg=--std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/deref.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/deref.carbon

// --- fail_todo_const_deref.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class ConstDeref {
public:
  auto operator*() const -> int&;
};
''';

fn TestDerefPass(var int_ptr: Cpp.ConstDeref) -> ref i32 {
  // CHECK:STDERR: fail_todo_const_deref.carbon:[[@LINE+4]]:10: error: semantics TODO: `Unsupported definition of interface CppUnsafeDeref` [SemanticsTodo]
  // CHECK:STDERR:   return int_ptr.(Core.CppUnsafeDeref.Op)();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return int_ptr.(Core.CppUnsafeDeref.Op)();
}

// --- fail_todo_mutable_deref.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class MutableDeref {
public:
  auto operator*() -> int&;
};
''';

fn TestDerefPass(var int_ptr: Cpp.MutableDeref) -> ref i32 {
  // CHECK:STDERR: fail_todo_mutable_deref.carbon:[[@LINE+4]]:10: error: semantics TODO: `Unsupported definition of interface CppUnsafeDeref` [SemanticsTodo]
  // CHECK:STDERR:   return int_ptr.(Core.CppUnsafeDeref.Op)();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return int_ptr.(Core.CppUnsafeDeref.Op)();
}

// --- fail_todo_multi_deref.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class MultiDeref {
public:
  auto operator*() & -> int&;
  auto operator*() const& -> const int&;

  auto operator*() && -> int&&;
  auto operator*() const&& -> int const&&;
};
''';

fn TestDerefPass(var int_ptr: Cpp.MultiDeref) -> ref i32 {
  // CHECK:STDERR: fail_todo_multi_deref.carbon:[[@LINE+4]]:10: error: semantics TODO: `operator* overload sets not implemented yet` [SemanticsTodo]
  // CHECK:STDERR:   return int_ptr.(Core.CppUnsafeDeref.Op)();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return int_ptr.(Core.CppUnsafeDeref.Op)();
}

// --- fail_todo_maybe_deref.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
template<class T, class U>
concept SameImpl = __is_same(T, U);

template<class T, class U>
concept SameAs = SameImpl<T, U> && SameImpl<U, T>;

template<class T>
class MaybeDeref {
public:
  auto operator*() const -> int&
  requires SameAs<T, int>;

  auto operator*() const -> const T&;
};
''';

fn TestDerefPass(var int_ptr: Cpp.MaybeDeref(i32)) -> ref i32 {
  // CHECK:STDERR: fail_todo_maybe_deref.carbon:[[@LINE+4]]:10: error: semantics TODO: `operator* overload sets not implemented yet` [SemanticsTodo]
  // CHECK:STDERR:   return int_ptr.(Core.CppUnsafeDeref.Op)();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return int_ptr.(Core.CppUnsafeDeref.Op)();
}

// --- fail_not_a_ptr.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class NotAPtr {};
''';

fn TestDerefFail(not_ptr: Cpp.NotAPtr) {
  // CHECK:STDERR: fail_not_a_ptr.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.CppUnsafeDeref` in type `Cpp.NotAPtr` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   not_ptr.(Core.CppUnsafeDeref.Op)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  not_ptr.(Core.CppUnsafeDeref.Op)();
}
