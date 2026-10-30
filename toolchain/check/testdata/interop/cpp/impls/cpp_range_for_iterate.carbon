// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// EXTRA-ARGS: --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/impls/cpp_range_for_iterate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/impls/cpp_range_for_iterate.carbon

// --- cpp_range_for_iterate.carbon
package Core library "cpp_range_for_iterate";

import library "prelude/parts/bool";

interface CppRangeForIterate {
  let Iterator: type;
  let Sentinel: type;

  // TODO: self must be ref-qualified
  fn Begin(self) -> Iterator;
  fn End(self) -> Sentinel;
}

interface IsRange {
  let Result: bool;
}

final impl forall [T: CppRangeForIterate] T as IsRange where .Result = true {}
impl forall [T: type] T as IsRange where .Result = false {}

fn AssertIsRange(unused generic R: IsRange where .Result = true) {}
fn AssertIsNotRange(unused generic S: IsRange where .Result = false) {}

// --- methods_return_same_type.carbon
library "[[@TEST_NAME]]";

import Core library "cpp_range_for_iterate";

import Cpp inline '''c++
class MutableRange {
  struct Iterator {};
public:
  auto begin() -> Iterator;
  auto end() -> Iterator;
};

class ConstRange {
  struct Iterator {};
public:
  auto begin() const -> Iterator;
  auto end() const -> Iterator;
};
''';

fn Test() {
  Core.AssertIsRange(Cpp.MutableRange);
  Core.AssertIsRange(Cpp.ConstRange);
}

// --- methods_return_different_types.carbon
library "[[@TEST_NAME]]";

import Core library "cpp_range_for_iterate";

import Cpp inline '''c++
class MutableRangeWithSentinel {
  struct Iterator {};
  struct Sentinel {};
public:
  auto begin() -> Iterator;
  auto end() -> Sentinel;
};

class ConstRangeWithSentinel {
  struct Iterator {};
  struct Sentinel {};
public:
  auto begin() const -> Iterator;
  auto end() const -> Sentinel;
};
''';

fn Test() {
  Core.AssertIsRange(Cpp.MutableRangeWithSentinel);
  Core.AssertIsRange(Cpp.ConstRangeWithSentinel);
}

// --- methods_overloaded.carbon
library "[[@TEST_NAME]]";

import Core library "cpp_range_for_iterate";

import Cpp inline '''c++
struct OverloadedRange {
  struct ConstIterator {};
  struct ConstSentinel {};
  struct Iterator {};
  struct Sentinel {};

  auto begin() -> Iterator;
  auto end() -> Sentinel;
  auto begin() const -> ConstIterator;
  auto end() const -> ConstSentinel;
};
''';

fn Test() {
  Core.AssertIsRange(Cpp.OverloadedRange);
  // We should pick the const overloads.
  let _: Core.CppRangeForIterate where
    .Iterator = Cpp.OverloadedRange.ConstIterator and
    .Sentinel = Cpp.OverloadedRange.ConstSentinel = Cpp.OverloadedRange;
}

// --- adl_return_same_types.carbon
library "[[@TEST_NAME]]";

import Core library "cpp_range_for_iterate";

import Cpp inline '''c++
namespace N {
class ConstRange {
  struct Iterator {};
public:
  friend auto begin(const ConstRange&) -> Iterator;
  friend auto end(const ConstRange&) -> Iterator;
};
} // namespace N
''';

fn Test() {
  Core.AssertIsRange(Cpp.N.ConstRange);
}

// --- fail_todo_adl_returns_same_types_mutable.carbon
// TODO: merge with adl_returns_same_types.carbon when passing.
library "[[@TEST_NAME]]";

import Core library "cpp_range_for_iterate";

import Cpp inline '''c++
namespace N {
class MutableRange {
  struct Iterator {};
public:
  // CHECK:STDERR:        ╭─┤ fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE+22]]:15
  // CHECK:STDERR: ->     │   friend auto begin(MutableRange&) -> Iterator;
  // CHECK:STDERR:        ·               ┳                  ┬
  // CHECK:STDERR:        ·               ┗━━━━━━━━━━━━━━━━━━━━━┫ error: value expression passed to
  // CHECK:STDERR:        ·                                  │  ┃        reference parameter
  // CHECK:STDERR:        ·                                  │  ┃        [ValueForRefParam]
  // CHECK:STDERR:        ·                                  ╰────┤ initializing function parameter
  // CHECK:STDERR:        ·                                       │ [InCallToFunctionParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ cpp_range_for_iterate.carbon
  // CHECK:STDERR:        │   fn Begin(self) -> Iterator;
  // CHECK:STDERR:        ·   ─────────────┬─────────────
  // CHECK:STDERR:        ·                ╰──┤ while building thunk to match the signature of this function
  // CHECK:STDERR:        ·                   │ [ThunkSignature]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ fn AssertIsRange(unused generic R: IsRange where .Result = true) {}
  // CHECK:STDERR:        ·                                 ┬──────────────────────────────
  // CHECK:STDERR:        ·                                 ╰──┤ initializing generic parameter `R` declared
  // CHECK:STDERR:        ·                                    │ here
  // CHECK:STDERR:        ·                                    │ [InitializingGenericParam]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  friend auto begin(MutableRange&) -> Iterator;
  // CHECK:STDERR:        ╭─┤ fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE+22]]:15
  // CHECK:STDERR: ->     │   friend auto end(MutableRange&) -> Iterator;
  // CHECK:STDERR:        ·               ┳                ┬
  // CHECK:STDERR:        ·               ┗━━━━━━━━━━━━━━━━━━━┫ error: value expression passed to reference
  // CHECK:STDERR:        ·                                │  ┃        parameter
  // CHECK:STDERR:        ·                                │  ┃        [ValueForRefParam]
  // CHECK:STDERR:        ·                                ╰────┤ initializing function parameter
  // CHECK:STDERR:        ·                                     │ [InCallToFunctionParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ cpp_range_for_iterate.carbon
  // CHECK:STDERR:        │   fn End(self) -> Sentinel;
  // CHECK:STDERR:        ·   ────────────┬────────────
  // CHECK:STDERR:        ·               ╰──┤ while building thunk to match the signature of this function
  // CHECK:STDERR:        ·                  │ [ThunkSignature]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ fn AssertIsRange(unused generic R: IsRange where .Result = true) {}
  // CHECK:STDERR:        ·                                 ┬──────────────────────────────
  // CHECK:STDERR:        ·                                 ╰──┤ initializing generic parameter `R` declared
  // CHECK:STDERR:        ·                                    │ here
  // CHECK:STDERR:        ·                                    │ [InitializingGenericParam]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  friend auto end(MutableRange&) -> Iterator;
};
} // namespace N
''';

fn Test() {
  Core.AssertIsRange(Cpp.N.MutableRange);
}

// --- adl_returns_different_types.carbon
library "[[@TEST_NAME]]";

import Core library "cpp_range_for_iterate";

import Cpp inline '''c++
namespace N {
class ConstRangeWithSentinel {
  struct Iterator {};
  struct Sentinel {};
public:
  friend auto begin(const ConstRangeWithSentinel&) -> Iterator;
  friend auto end(const ConstRangeWithSentinel&) -> Sentinel;
};
} // namespace N
''';

fn Test() {
  Core.AssertIsRange(Cpp.N.ConstRangeWithSentinel);
}

// --- fail_todo_adl_returns_different_types_mutable.carbon
library "[[@TEST_NAME]]";

import Core library "cpp_range_for_iterate";

import Cpp inline '''c++
namespace N {
class MutableRangeWithSentinel {
  struct Iterator {
    // CHECK:STDERR:        ╭─┤ fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+7]]:31
    // CHECK:STDERR: ->     │     auto operator*() const -> ValueType&;
    // CHECK:STDERR:        ·                               ┳
    // CHECK:STDERR:        ·                               ┗━━┫ error: unknown type name 'ValueType'
    // CHECK:STDERR:        ·                                  ┃        [CppInteropParseError]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    auto operator*() const -> ValueType&;
    auto operator++() -> Iterator&;
  };

  struct Sentinel {
    friend auto operator==(Sentinel, Iterator) -> bool;
    friend auto operator!=(Sentinel, Iterator) -> bool;
  };
public:
  // CHECK:STDERR:        ╭─┤ fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+23]]:15
  // CHECK:STDERR: ->     │   friend auto begin(MutableRangeWithSentinel&) -> Iterator;
  // CHECK:STDERR:        ·               ┳                              ┬
  // CHECK:STDERR:        ·               ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫ error: value expression passed
  // CHECK:STDERR:        ·                                              │  ┃        to reference parameter
  // CHECK:STDERR:        ·                                              │  ┃        [ValueForRefParam]
  // CHECK:STDERR:        ·                                              ╰────┤ initializing function
  // CHECK:STDERR:        ·                                                   │ parameter
  // CHECK:STDERR:        ·                                                   │ [InCallToFunctionParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ cpp_range_for_iterate.carbon
  // CHECK:STDERR:        │   fn Begin(self) -> Iterator;
  // CHECK:STDERR:        ·   ─────────────┬─────────────
  // CHECK:STDERR:        ·                ╰──┤ while building thunk to match the signature of this function
  // CHECK:STDERR:        ·                   │ [ThunkSignature]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ fn AssertIsRange(unused generic R: IsRange where .Result = true) {}
  // CHECK:STDERR:        ·                                 ┬──────────────────────────────
  // CHECK:STDERR:        ·                                 ╰──┤ initializing generic parameter `R` declared
  // CHECK:STDERR:        ·                                    │ here
  // CHECK:STDERR:        ·                                    │ [InitializingGenericParam]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  friend auto begin(MutableRangeWithSentinel&) -> Iterator;
  // CHECK:STDERR:        ╭─┤ fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+23]]:15
  // CHECK:STDERR: ->     │   friend auto end(MutableRangeWithSentinel&) -> Sentinel;
  // CHECK:STDERR:        ·               ┳                            ┬
  // CHECK:STDERR:        ·               ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫ error: value expression passed
  // CHECK:STDERR:        ·                                            │  ┃        to reference parameter
  // CHECK:STDERR:        ·                                            │  ┃        [ValueForRefParam]
  // CHECK:STDERR:        ·                                            ╰────┤ initializing function
  // CHECK:STDERR:        ·                                                 │ parameter
  // CHECK:STDERR:        ·                                                 │ [InCallToFunctionParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ cpp_range_for_iterate.carbon
  // CHECK:STDERR:        │   fn End(self) -> Sentinel;
  // CHECK:STDERR:        ·   ────────────┬────────────
  // CHECK:STDERR:        ·               ╰──┤ while building thunk to match the signature of this function
  // CHECK:STDERR:        ·                  │ [ThunkSignature]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ fn AssertIsRange(unused generic R: IsRange where .Result = true) {}
  // CHECK:STDERR:        ·                                 ┬──────────────────────────────
  // CHECK:STDERR:        ·                                 ╰──┤ initializing generic parameter `R` declared
  // CHECK:STDERR:        ·                                    │ here
  // CHECK:STDERR:        ·                                    │ [InitializingGenericParam]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  friend auto end(MutableRangeWithSentinel&) -> Sentinel;
};
}
''';

fn Test() {
  Core.AssertIsRange(Cpp.N.MutableRangeWithSentinel);
}

// --- fail_todo_adl_overloads.carbon
library "[[@TEST_NAME]]";

import Core library "cpp_range_for_iterate";

import Cpp inline '''c++
namespace N {
class MutableAndConstRange {
  struct Iterator {};
  struct ConstIterator {};
public:
  // CHECK:STDERR:        ╭─┤ fail_todo_adl_overloads.carbon:[[@LINE+22]]:15
  // CHECK:STDERR: ->     │   friend auto begin(MutableAndConstRange&) -> Iterator;
  // CHECK:STDERR:        ·               ┳                          ┬
  // CHECK:STDERR:        ·               ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫ error: value expression passed to
  // CHECK:STDERR:        ·                                          │  ┃        reference parameter
  // CHECK:STDERR:        ·                                          │  ┃        [ValueForRefParam]
  // CHECK:STDERR:        ·                                          ╰────┤ initializing function parameter
  // CHECK:STDERR:        ·                                               │ [InCallToFunctionParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ cpp_range_for_iterate.carbon
  // CHECK:STDERR:        │   fn Begin(self) -> Iterator;
  // CHECK:STDERR:        ·   ─────────────┬─────────────
  // CHECK:STDERR:        ·                ╰──┤ while building thunk to match the signature of this function
  // CHECK:STDERR:        ·                   │ [ThunkSignature]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ fn AssertIsRange(unused generic R: IsRange where .Result = true) {}
  // CHECK:STDERR:        ·                                 ┬──────────────────────────────
  // CHECK:STDERR:        ·                                 ╰──┤ initializing generic parameter `R` declared
  // CHECK:STDERR:        ·                                    │ here
  // CHECK:STDERR:        ·                                    │ [InitializingGenericParam]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  friend auto begin(MutableAndConstRange&) -> Iterator;
  // CHECK:STDERR:        ╭─┤ fail_todo_adl_overloads.carbon:[[@LINE+22]]:15
  // CHECK:STDERR: ->     │   friend auto end(MutableAndConstRange&) -> Iterator;
  // CHECK:STDERR:        ·               ┳                        ┬
  // CHECK:STDERR:        ·               ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┫ error: value expression passed to
  // CHECK:STDERR:        ·                                        │  ┃        reference parameter
  // CHECK:STDERR:        ·                                        │  ┃        [ValueForRefParam]
  // CHECK:STDERR:        ·                                        ╰────┤ initializing function parameter
  // CHECK:STDERR:        ·                                             │ [InCallToFunctionParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ cpp_range_for_iterate.carbon
  // CHECK:STDERR:        │   fn End(self) -> Sentinel;
  // CHECK:STDERR:        ·   ────────────┬────────────
  // CHECK:STDERR:        ·               ╰──┤ while building thunk to match the signature of this function
  // CHECK:STDERR:        ·                  │ [ThunkSignature]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ fn AssertIsRange(unused generic R: IsRange where .Result = true) {}
  // CHECK:STDERR:        ·                                 ┬──────────────────────────────
  // CHECK:STDERR:        ·                                 ╰──┤ initializing generic parameter `R` declared
  // CHECK:STDERR:        ·                                    │ here
  // CHECK:STDERR:        ·                                    │ [InitializingGenericParam]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  friend auto end(MutableAndConstRange&) -> Iterator;

  friend auto begin(const MutableAndConstRange&) -> ConstIterator;
  friend auto end(const MutableAndConstRange&) -> ConstIterator;
};
}
''';

fn Test() {
  Core.AssertIsRange(Cpp.N.MutableAndConstRange);
}

// --- not_ranges.carbon
library "[[@TEST_NAME]]";

import Core library "cpp_range_for_iterate";

import Cpp inline '''c++
class NoBeginEnd {};

struct NoBeginMethod {
  auto end() -> int;
};

struct NoEndMethod {
  auto begin() -> int;
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

struct BeginFieldEndMethod {
  int begin;
  auto end() -> int;
};

struct BeginMethodEndField {
  auto begin() -> int;
  int end;
};
''';

fn Test() {
  Core.AssertIsNotRange(Cpp.NoBeginEnd);
  Core.AssertIsNotRange(Cpp.NoBeginMethod);
  Core.AssertIsNotRange(Cpp.NoEndMethod);
  Core.AssertIsNotRange(Cpp.NoBeginADL);
  Core.AssertIsNotRange(Cpp.NoEndADL);
  Core.AssertIsNotRange(Cpp.BeginMethodEndADL);
  Core.AssertIsNotRange(Cpp.BeginADLEndMethod);
  Core.AssertIsNotRange(Cpp.BeginFieldEndMethod);
  Core.AssertIsNotRange(Cpp.BeginMethodEndField);
}

// --- fail_deleted.carbon
library "[[@TEST_NAME]]";

import Core library "cpp_range_for_iterate";

import Cpp inline '''c++
struct BeginMethodDeleted {
  auto begin() = delete;
  auto end() -> int;
};

struct EndMethodDeleted {
  auto begin() -> int;
  auto end() = delete;
};

struct BeginFunctionDeleted {
  friend auto begin(const BeginFunctionDeleted&) = delete;
  friend auto end(const BeginFunctionDeleted&) -> int;
};

struct EndFunctionDeleted {
  friend auto begin(const EndFunctionDeleted&) -> int;
  friend auto end(const EndFunctionDeleted&) = delete;
};
''';

fn Test() {
  // CHECK:STDERR:        ╭─┤ fail_deleted.carbon:[[@LINE+20]]:47
  // CHECK:STDERR:        │   auto begin() = delete;
  // CHECK:STDERR:        ·        ┬
  // CHECK:STDERR:        ·        ╰──┤ 'begin' has been explicitly marked deleted here
  // CHECK:STDERR:        ·           │ [CppInteropParseNote]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   Core.AssertIsNotRange(Cpp.BeginMethodDeleted);
  // CHECK:STDERR:        ·                                               ┳
  // CHECK:STDERR:        ·                                               ┗━━┫ error: attempt to use a
  // CHECK:STDERR:        ·                                                  ┃        deleted function
  // CHECK:STDERR:        ·                                                  ┃        [CppInteropParseError]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ cpp_range_for_iterate.carbon
  // CHECK:STDERR:        │ fn AssertIsNotRange(unused generic S: IsRange where .Result = false) {}
  // CHECK:STDERR:        ·                                    ┬───────────────────────────────
  // CHECK:STDERR:        ·                                    ╰──┤ initializing generic parameter `S`
  // CHECK:STDERR:        ·                                       │ declared here
  // CHECK:STDERR:        ·                                       │ [InitializingGenericParam]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Core.AssertIsNotRange(Cpp.BeginMethodDeleted);
  // CHECK:STDERR:        ╭─┤ fail_deleted.carbon:[[@LINE+19]]:45
  // CHECK:STDERR:        │   auto end() = delete;
  // CHECK:STDERR:        ·        ┬
  // CHECK:STDERR:        ·        ╰──┤ 'end' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   Core.AssertIsNotRange(Cpp.EndMethodDeleted);
  // CHECK:STDERR:        ·                                             ┳
  // CHECK:STDERR:        ·                                             ┗━━┫ error: attempt to use a deleted
  // CHECK:STDERR:        ·                                                ┃        function
  // CHECK:STDERR:        ·                                                ┃        [CppInteropParseError]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ cpp_range_for_iterate.carbon
  // CHECK:STDERR:        │ fn AssertIsNotRange(unused generic S: IsRange where .Result = false) {}
  // CHECK:STDERR:        ·                                    ┬───────────────────────────────
  // CHECK:STDERR:        ·                                    ╰──┤ initializing generic parameter `S`
  // CHECK:STDERR:        ·                                       │ declared here
  // CHECK:STDERR:        ·                                       │ [InitializingGenericParam]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Core.AssertIsNotRange(Cpp.EndMethodDeleted);
  // CHECK:STDERR:        ╭─┤ fail_deleted.carbon:[[@LINE+20]]:49
  // CHECK:STDERR:        │   friend auto begin(const BeginFunctionDeleted&) = delete;
  // CHECK:STDERR:        ·               ┬
  // CHECK:STDERR:        ·               ╰──┤ candidate function has been explicitly deleted
  // CHECK:STDERR:        ·                  │ [CppInteropParseNote]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   Core.AssertIsNotRange(Cpp.BeginFunctionDeleted);
  // CHECK:STDERR:        ·                                                 ┳
  // CHECK:STDERR:        ·                                                 ┗━━┫ error: call to deleted
  // CHECK:STDERR:        ·                                                    ┃        function 'begin'
  // CHECK:STDERR:        ·                                                    ┃        [CppInteropParseError]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ cpp_range_for_iterate.carbon
  // CHECK:STDERR:        │ fn AssertIsNotRange(unused generic S: IsRange where .Result = false) {}
  // CHECK:STDERR:        ·                                    ┬───────────────────────────────
  // CHECK:STDERR:        ·                                    ╰──┤ initializing generic parameter `S`
  // CHECK:STDERR:        ·                                       │ declared here
  // CHECK:STDERR:        ·                                       │ [InitializingGenericParam]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Core.AssertIsNotRange(Cpp.BeginFunctionDeleted);
  // CHECK:STDERR:        ╭─┤ fail_deleted.carbon:[[@LINE+20]]:47
  // CHECK:STDERR:        │   friend auto end(const EndFunctionDeleted&) = delete;
  // CHECK:STDERR:        ·               ┬
  // CHECK:STDERR:        ·               ╰──┤ candidate function has been explicitly deleted
  // CHECK:STDERR:        ·                  │ [CppInteropParseNote]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   Core.AssertIsNotRange(Cpp.EndFunctionDeleted);
  // CHECK:STDERR:        ·                                               ┳
  // CHECK:STDERR:        ·                                               ┗━━┫ error: call to deleted
  // CHECK:STDERR:        ·                                                  ┃        function 'end'
  // CHECK:STDERR:        ·                                                  ┃        [CppInteropParseError]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ cpp_range_for_iterate.carbon
  // CHECK:STDERR:        │ fn AssertIsNotRange(unused generic S: IsRange where .Result = false) {}
  // CHECK:STDERR:        ·                                    ┬───────────────────────────────
  // CHECK:STDERR:        ·                                    ╰──┤ initializing generic parameter `S`
  // CHECK:STDERR:        ·                                       │ declared here
  // CHECK:STDERR:        ·                                       │ [InitializingGenericParam]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Core.AssertIsNotRange(Cpp.EndFunctionDeleted);
}
