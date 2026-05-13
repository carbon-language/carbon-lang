// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// EXTRA-ARGS: --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/range_for.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/range_for.carbon

// --- methods_return_same_type.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class MutableRange {
  struct Iterator {
    auto operator*() const -> int&;
    auto operator++() -> Iterator&;
    friend auto operator==(Iterator, Iterator) -> bool;
    friend auto operator!=(Iterator, Iterator) -> bool;
  };
public:
  auto begin() -> Iterator;
  auto end() -> Iterator;
};

class ConstRange {
  struct Iterator {
    auto operator*() const -> const int&;
    auto operator++() -> Iterator&;
    friend auto operator==(Iterator, Iterator) -> bool;
    friend auto operator!=(Iterator, Iterator) -> bool;
  };
public:
  auto begin() const -> Iterator;
  auto end() const -> Iterator;
};
''';

// TODO: uncomment after #7181 is merged.
// fn CppRangeForIterate[R:! Core.CppRangeForIterate where .Iterator impls Core.Destroy and .Sentinel impls Core.Destroy](r: R) {
//   //@dump-sem-ir-begin
//   r.Begin();
//   r.End();
//   //@dump-sem-ir-end
// }

fn Test(var m: Cpp.MutableRange, var c: Cpp.ConstRange) {
  // TODO: replace with Test(m) when #7181 is merged.
  m.(Core.CppRangeForIterate.Begin)();
  m.(Core.CppRangeForIterate.End)();

  // TODO: replace with Test(c) when #7181 is merged.
  c.(Core.CppRangeForIterate.Begin)();
  c.(Core.CppRangeForIterate.End)();
}

// --- methods_return_different_types.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class MutableRange {
  struct ValueType {};

  struct Iterator {
    auto operator*() const -> ValueType&;
    auto operator++() -> Iterator&;
  };

  struct Sentinel {
    friend auto operator==(Sentinel, Iterator) -> bool;
    friend auto operator!=(Sentinel, Iterator) -> bool;
  };
public:
  auto begin() -> Iterator;
  auto end() -> Sentinel;
};

class ConstRange {
  struct ValueType {};

  struct Iterator {
    auto operator*() const -> ValueType;
    auto operator++() -> Iterator&;
  };

  struct Sentinel {
    friend auto operator==(Sentinel, Iterator) -> bool;
    friend auto operator!=(Sentinel, Iterator) -> bool;
  };
public:
  auto begin() const -> Iterator;
  auto end() const -> Sentinel;
};
''';

// TODO: uncomment after #7181 is merged.
// fn Test[R:! Core.CppRangeForIterate where .Iterator impls Core.Destroy and .Sentinel impls Core.Destroy](r: R) {
//   //@dump-sem-ir-begin
//   r.Begin();
//   r.End();
//   //@dump-sem-ir-end
// }

fn TestDriver(var m: Cpp.MutableRange, var c: Cpp.ConstRange) {
  // TODO: replace with Test(m) when #7181 is merged.
  m.(Core.CppRangeForIterate.Begin)();
  m.(Core.CppRangeForIterate.End)();

  // TODO: replace with Test(c) when #7181 is merged.
  c.(Core.CppRangeForIterate.Begin)();
  c.(Core.CppRangeForIterate.End)();
}

// --- fail_todo_adl_return_same_types.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
namespace N {
class MutableRange {
  struct Iterator {
    auto operator*() const -> int&;
    auto operator++() -> Iterator&;
    friend auto operator==(Iterator, Iterator) -> bool;
    friend auto operator!=(Iterator, Iterator) -> bool;
  };
public:
  friend auto begin(MutableRange&) -> Iterator;
  friend auto end(MutableRange&) -> Iterator;
};

class ConstRange {
  struct Iterator {
    auto operator*() const -> const int&;
    auto operator++() -> Iterator&;
    friend auto operator==(Iterator, Iterator) -> bool;
    friend auto operator!=(Iterator, Iterator) -> bool;
  };
public:
  friend auto begin(const ConstRange&) -> Iterator;
  friend auto end(const ConstRange&) -> Iterator;
};
} // namespace N
''';

// TODO: uncomment after #7181 is merged.
// fn CppRangeForIterate[R:! Core.CppRangeForIterate where .Iterator impls Core.Destroy and .Sentinel impls Core.Destroy](r: R) {
//   //@dump-sem-ir-begin
//   r.Begin();
//   r.End();
//   //@dump-sem-ir-end
// }

fn Test(var m: Cpp.N.MutableRange, c: Cpp.N.ConstRange) {
  // TODO: replace with Test(m) when #7181 is merged.
  // CHECK:STDERR: fail_todo_adl_return_same_types.carbon:[[@LINE+4]]:3: error: semantics TODO: `support ADL begin/end` [SemanticsTodo]
  // CHECK:STDERR:   m.(Core.CppRangeForIterate.Begin)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  m.(Core.CppRangeForIterate.Begin)();
  m.(Core.CppRangeForIterate.End)();

  // TODO: replace with Test(c) when #7181 is merged.
  // CHECK:STDERR: fail_todo_adl_return_same_types.carbon:[[@LINE+4]]:3: error: semantics TODO: `support ADL begin/end` [SemanticsTodo]
  // CHECK:STDERR:   c.(Core.CppRangeForIterate.Begin)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  c.(Core.CppRangeForIterate.Begin)();
  c.(Core.CppRangeForIterate.End)();
}

// --- fail_todo_adl_returns_different_types.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
namespace N {
class MutableRange {
  struct ValueType {};

  struct Iterator {
    auto operator*() const -> ValueType&;
    auto operator++() -> Iterator&;
  };

  struct Sentinel {
    friend auto operator==(Sentinel, Iterator) -> bool;
    friend auto operator!=(Sentinel, Iterator) -> bool;
  };
public:
  friend auto begin(MutableRange&) -> Iterator;
  friend auto end(MutableRange&) -> Sentinel;
};

class ConstRange {
  struct ValueType {};

  struct Iterator {
    auto operator*() const -> ValueType;
    auto operator++() -> Iterator&;
  };

  struct Sentinel {
    friend auto operator==(Sentinel, Iterator) -> bool;
    friend auto operator!=(Sentinel, Iterator) -> bool;
  };
public:
  friend auto begin(const ConstRange&) -> Iterator;
  friend auto end(const ConstRange&) -> Sentinel;
};
} // namespace N
''';

// TODO: uncomment after #7181 is merged.
// fn Test[R:! Core.CppRangeForIterate where .Iterator impls Core.Destroy and .Sentinel impls Core.Destroy](r: R) {
//   //@dump-sem-ir-begin
//   r.Begin();
//   r.End();
//   //@dump-sem-ir-end
// }

fn TestDriver(var m: Cpp.N.MutableRange, c: Cpp.N.ConstRange) {
  // TODO: replace with Test(m) when #7181 is merged.
  // CHECK:STDERR: fail_todo_adl_returns_different_types.carbon:[[@LINE+4]]:3: error: semantics TODO: `support ADL begin/end` [SemanticsTodo]
  // CHECK:STDERR:   m.(Core.CppRangeForIterate.Begin)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  m.(Core.CppRangeForIterate.Begin)();
  m.(Core.CppRangeForIterate.End)();

  // TODO: replace with Test(c) when #7181 is merged.
  // CHECK:STDERR: fail_todo_adl_returns_different_types.carbon:[[@LINE+4]]:3: error: semantics TODO: `support ADL begin/end` [SemanticsTodo]
  // CHECK:STDERR:   c.(Core.CppRangeForIterate.Begin)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  c.(Core.CppRangeForIterate.Begin)();
  c.(Core.CppRangeForIterate.End)();
}

// --- fail_todo_never_valid.carbon

// TODO: rename to `fail_never_valid.carbon` after ADL is supported.

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class NoBeginEnd {};

struct NoBeginMethod {
  auto end() -> int;
};

struct NoEndMethod {
  auto begin();
};

struct NoBeginADL {
  friend auto end(NoBeginADL) -> int;
};

struct NoEndADL {
  friend auto begin(NoEndADL) -> int;
};

struct BeginMethodEndADL {
  auto begin() -> int;
  friend auto end(BeginMethodEndADL) -> int;
};

struct BeginADLEndMethod {
  friend auto begin(BeginADLEndMethod) -> int;
  auto end() -> int;
};
''';

fn Test[R:! Core.CppRangeForIterate](unused r: R) {}

fn TestDriver(no_begin_end: Cpp.NoBeginEnd,
              no_begin_method: Cpp.NoBeginMethod,
              no_end_method: Cpp.NoEndMethod,
              no_begin_adl: Cpp.NoBeginADL,
              no_end_adl: Cpp.NoEndADL,
              begin_method_end_adl: Cpp.BeginMethodEndADL,
              begin_adl_end_method: Cpp.BeginADLEndMethod)
{
  // CHECK:STDERR: fail_todo_never_valid.carbon:[[@LINE+7]]:3: error: semantics TODO: `support ADL begin/end` [SemanticsTodo]
  // CHECK:STDERR:   Test(no_begin_end);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_never_valid.carbon:[[@LINE-13]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_begin_end);

  // CHECK:STDERR: fail_todo_never_valid.carbon:[[@LINE+7]]:3: error: semantics TODO: `support ADL begin/end` [SemanticsTodo]
  // CHECK:STDERR:   Test(no_begin_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_never_valid.carbon:[[@LINE-22]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_begin_method);

  // CHECK:STDERR: fail_todo_never_valid.carbon:[[@LINE+10]]:21: error: function 'begin' with deduced return type cannot be used before it is defined [CppInteropParseError]
  // CHECK:STDERR:    74 |   Test(no_end_method);
  // CHECK:STDERR:       |                     ^
  // CHECK:STDERR: fail_todo_never_valid.carbon:[[@LINE-53]]:8: note: 'begin' declared here [CppInteropParseNote]
  // CHECK:STDERR:    14 |   auto begin();
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR: fail_todo_never_valid.carbon:[[@LINE-34]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_end_method);

  // CHECK:STDERR: fail_todo_never_valid.carbon:[[@LINE+7]]:3: error: semantics TODO: `support ADL begin/end` [SemanticsTodo]
  // CHECK:STDERR:   Test(no_begin_adl);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_never_valid.carbon:[[@LINE-43]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_begin_adl);

  // CHECK:STDERR: fail_todo_never_valid.carbon:[[@LINE+7]]:3: error: semantics TODO: `support ADL begin/end` [SemanticsTodo]
  // CHECK:STDERR:   Test(no_end_adl);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_never_valid.carbon:[[@LINE-52]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_end_adl);

  // CHECK:STDERR: fail_todo_never_valid.carbon:[[@LINE+7]]:3: error: semantics TODO: `support ADL begin/end` [SemanticsTodo]
  // CHECK:STDERR:   Test(begin_method_end_adl);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_never_valid.carbon:[[@LINE-61]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_method_end_adl);

  // CHECK:STDERR: fail_todo_never_valid.carbon:[[@LINE+7]]:3: error: semantics TODO: `support ADL begin/end` [SemanticsTodo]
  // CHECK:STDERR:   Test(begin_adl_end_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_never_valid.carbon:[[@LINE-70]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_adl_end_method);
}
