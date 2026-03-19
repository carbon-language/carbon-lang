// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// EXTRA-ARGS: --clang-arg=-v
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/range_for.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/range_for.carbon
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:61:83: error: member name `CppUnsafeDeref` not found in `CppRangeForIterate` [MemberNameNotFoundInSpecificScope]
// CHECK:STDERR:       returned var result: Optional(R.ElementType) = Optional(R.ElementType).Some(R.CppUnsafeDeref(&cursor->0));
// CHECK:STDERR:                                                                                   ^~~~~~~~~~~~~~~~
// CHECK:STDERR:

// --- range-for.carbon

import Cpp inline '''c++
template<class T>
class Iterator {
public:
  auto operator*() const -> T&;
  auto operator++() -> Iterator&;
  friend auto operator!=(Iterator, Iterator) -> bool;
};

template<class T>
class Vector {
public:
  auto begin() -> Iterator<T>;
  auto begin() const -> Iterator<T const>;

  auto end() -> Iterator<T>;
  auto end() const -> Iterator<T const>;
};

struct Sentinel {
  template<class T>
  friend auto operator!=(Iterator<T>, Sentinel) -> bool;
};

template<class T>
struct RangeAdaptor {
  auto begin() -> Iterator<T>;
  auto begin() const -> Iterator<T const>;

  auto end() -> Sentinel;
  auto end() const -> Sentinel;
};
''';

fn Sum(v: Cpp.Vector(i32)) -> i32 {
  returned var result: i32 = 0;
  // CHECK:STDERR: range-for.carbon:[[@LINE+8]]:7: error: cannot access member of interface `Core.Iterate` in type `Cpp.Vector` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   for (var i: i32 in v) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: range-for.carbon:[[@LINE+4]]:7: error: cannot access member of interface `Core.Iterate` in type `Cpp.Vector` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   for (var i: i32 in v) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  for (var i: i32 in v) {
    result += i;
  }

  return var;
}
