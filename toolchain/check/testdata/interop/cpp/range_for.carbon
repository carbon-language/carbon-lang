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

fn Test[R:! Core.Iterate where .ElementType = i32](r: R) {
  var sum: i32 = 0;

  //@dump-sem-ir-begin
  for (i: i32 in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestDriver(var m: Cpp.MutableRange, c: Cpp.ConstRange) {
  Test(m);
  Test(c);
}

// --- methods_return_different_types.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
struct ValueType {
  auto operator+=(const ValueType&) -> ValueType&;
};

class MutableRange {
  struct Iterator {
    auto operator*() const -> ValueType&;
    auto operator++() -> Iterator&;
  };

  struct Sentinel {
    friend auto operator==(Iterator, Sentinel) -> bool;
    friend auto operator!=(Iterator, Sentinel) -> bool;
  };
public:
  auto begin() -> Iterator;
  auto end() -> Sentinel;
};

class ConstRange {
  struct Iterator {
    auto operator*() const -> ValueType;
    auto operator++() -> Iterator&;
  };

  struct Sentinel {
    friend auto operator==(Iterator, Sentinel) -> bool;
    friend auto operator!=(Iterator, Sentinel) -> bool;
  };
public:
  auto begin() const -> Iterator;
  auto end() const -> Sentinel;
};
''';

fn Test[R:! Core.Iterate where .ElementType = Cpp.ValueType](var r: R) {
  var sum: Cpp.ValueType = Cpp.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.ValueType in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestDriver(var m: Cpp.MutableRange, c: Cpp.ConstRange) {
  Test(m);
  Test(c);
}

// --- adl_return_same_types.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
namespace N {
class ConstRange {
  struct Iterator {
    auto operator*() const -> const double&;
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

fn Test[R:! Core.Iterate where .ElementType = f64](r: R) {
  var sum: f64 = 0.0;

  //@dump-sem-ir-begin
  for (i: f64 in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestDriver(c: Cpp.N.ConstRange) {
  Test(c);
}

// --- fail_todo_adl_returns_same_types_mutable.carbon

// TODO: merge with adl_returns_same_types.carbon when passing.

library "[[@TEST_NAME]]";

import Cpp inline '''c++
namespace N {
class MutableRange {
  struct Iterator {
    auto operator*() const -> double&;
    auto operator++() -> Iterator&;
    friend auto operator==(Iterator, Iterator) -> bool;
    friend auto operator!=(Iterator, Iterator) -> bool;
  };
public:
  friend auto begin(MutableRange&) -> Iterator;
  friend auto end(MutableRange&) -> Iterator;
};
} // namespace N
''';

fn Test[R:! Core.Iterate where .ElementType = f64](r: R) {
  var sum: f64 = 0.0;

  //@dump-sem-ir-begin
  for (i: f64 in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestDriver(var m: Cpp.N.MutableRange) {
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE+32]]:3: error: facet type `Core.CppRangeForIterate where .(Core.CppRangeForIterate.Iterator) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator) impls Core.Copy and .(Core.CppRangeForIterate.Iterator) impls Core.CppUnsafeDeref and .(Core.CppRangeForIterate.Iterator) impls Core.EqWith(.(Core.CppRangeForIterate.Sentinel)) and .(Core.CppRangeForIterate.Iterator) impls Core.Inc and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy and .(Core.CppRangeForIterate.Sentinel) impls Core.Destroy and .(Core.CppRangeForIterate.Sentinel) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   Test(m);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:39:12: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR:   fn Begin[self: Self]() -> Iterator;
  // CHECK:STDERR:            ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE-23]]:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   friend auto begin(MutableRange&) -> Iterator;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:39:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn Begin[self: Self]() -> Iterator;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE-23]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.Iterate where .ElementType = f64](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE+16]]:3: error: facet type `Core.CppRangeForIterate where .(Core.CppRangeForIterate.Iterator) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator) impls Core.Copy and .(Core.CppRangeForIterate.Iterator) impls Core.CppUnsafeDeref and .(Core.CppRangeForIterate.Iterator) impls Core.EqWith(.(Core.CppRangeForIterate.Sentinel)) and .(Core.CppRangeForIterate.Iterator) impls Core.Inc and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy and .(Core.CppRangeForIterate.Sentinel) impls Core.Destroy and .(Core.CppRangeForIterate.Sentinel) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   Test(m);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:40:10: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR:   fn End[self: Self]() -> Sentinel;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE-38]]:32: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   friend auto end(MutableRange&) -> Iterator;
  // CHECK:STDERR:                                ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:40:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn End[self: Self]() -> Sentinel;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE-39]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.Iterate where .ElementType = f64](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(m);
}

// --- adl_returns_different_types.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
namespace N {
struct ValueType {
  auto operator+=(const ValueType& other) -> ValueType&;
};

class ConstRange {
  struct Iterator {
    auto operator*() const -> ValueType;
    auto operator++() -> Iterator&;
  };

  struct Sentinel {
    friend auto operator==(Iterator, Sentinel) -> bool;
    friend auto operator!=(Iterator, Sentinel) -> bool;
  };
public:
  friend auto begin(const ConstRange&) -> Iterator;
  friend auto end(const ConstRange&) -> Sentinel;
};
} // namespace N
''';

fn Test[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  var sum: Cpp.N.ValueType = Cpp.N.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.N.ValueType in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestDriver(c: Cpp.N.ConstRange) {
  Test(c);
}

// --- fail_todo_adl_returns_different_types_mutable.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
namespace N {
struct ValueType {
  auto operator+=(const ValueType& other) -> ValueType&;
};

class MutableRange {

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
}
''';

fn Test[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  var sum: Cpp.N.ValueType = Cpp.N.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.N.ValueType in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestDriver(var m: Cpp.N.MutableRange) {
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+46]]:3: error: facet type `Core.CppRangeForIterate where .(Core.CppRangeForIterate.Iterator) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator) impls Core.Copy and .(Core.CppRangeForIterate.Iterator) impls Core.CppUnsafeDeref and .(Core.CppRangeForIterate.Iterator) impls Core.EqWith(.(Core.CppRangeForIterate.Sentinel)) and .(Core.CppRangeForIterate.Iterator) impls Core.Inc and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy and .(Core.CppRangeForIterate.Sentinel) impls Core.Destroy and .(Core.CppRangeForIterate.Sentinel) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   Test(m);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:39:12: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR:   fn Begin[self: Self]() -> Iterator;
  // CHECK:STDERR:            ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-23]]:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   friend auto begin(MutableRange&) -> Iterator;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:39:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn Begin[self: Self]() -> Iterator;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-23]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+30]]:3: error: facet type `Core.CppRangeForIterate where .(Core.CppRangeForIterate.Iterator) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator) impls Core.Copy and .(Core.CppRangeForIterate.Iterator) impls Core.CppUnsafeDeref and .(Core.CppRangeForIterate.Iterator) impls Core.EqWith(.(Core.CppRangeForIterate.Sentinel)) and .(Core.CppRangeForIterate.Iterator) impls Core.Inc and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy and .(Core.CppRangeForIterate.Sentinel) impls Core.Destroy and .(Core.CppRangeForIterate.Sentinel) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   Test(m);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:40:10: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR:   fn End[self: Self]() -> Sentinel;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-38]]:32: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   friend auto end(MutableRange&) -> Sentinel;
  // CHECK:STDERR:                                ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:40:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn End[self: Self]() -> Sentinel;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-39]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+14]]:3: error: semantics TODO: `Rewriting operator== using operator== is not supported` [SemanticsTodo]
  // CHECK:STDERR:   Test(m);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-46]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.N.MutableRange` into type implementing `Core.Iterate where .(Core.Iterate.ElementType) = Cpp.N.ValueType as Core.Destroy & Core.Copy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(m);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-53]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(m);
}

// --- fail_todo_adl_overloads.carbon

import Cpp inline '''c++
namespace N {
struct ValueType {
  auto operator+=(const ValueType&) -> ValueType&;
};

class MutableAndConstRange {

  struct Iterator {
    auto operator*() const -> ValueType&;
    auto operator++() -> Iterator&;

    friend auto operator==(Iterator, Iterator) -> bool;
    friend auto operator!=(Iterator, Iterator) -> bool;
  };

  struct ConstIterator {
    auto operator*() const -> ValueType&;
    auto operator++() -> Iterator&;

    friend auto operator==(ConstIterator, ConstIterator) -> bool;
    friend auto operator!=(ConstIterator, ConstIterator) -> bool;
  };
  public:
    friend auto begin(MutableAndConstRange&) -> Iterator;
    friend auto end(MutableAndConstRange&) -> Iterator;

    friend auto begin(const MutableAndConstRange&) -> ConstIterator;
    friend auto end(const MutableAndConstRange&) -> ConstIterator;
};
}
''';

fn Test[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  var sum: Cpp.N.ValueType = Cpp.N.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.N.ValueType in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestDriver(var mc: Cpp.N.MutableAndConstRange) {
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE+32]]:3: error: facet type `Core.CppRangeForIterate where .(Core.CppRangeForIterate.Iterator) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator) impls Core.Copy and .(Core.CppRangeForIterate.Iterator) impls Core.CppUnsafeDeref and .(Core.CppRangeForIterate.Iterator) impls Core.EqWith(.(Core.CppRangeForIterate.Sentinel)) and .(Core.CppRangeForIterate.Iterator) impls Core.Inc and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy and .(Core.CppRangeForIterate.Sentinel) impls Core.Destroy and .(Core.CppRangeForIterate.Sentinel) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   Test(mc);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:39:12: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR:   fn Begin[self: Self]() -> Iterator;
  // CHECK:STDERR:            ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE-26]]:44: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:     friend auto begin(MutableAndConstRange&) -> Iterator;
  // CHECK:STDERR:                                            ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:39:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn Begin[self: Self]() -> Iterator;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE-23]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE+16]]:3: error: facet type `Core.CppRangeForIterate where .(Core.CppRangeForIterate.Iterator) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator) impls Core.Copy and .(Core.CppRangeForIterate.Iterator) impls Core.CppUnsafeDeref and .(Core.CppRangeForIterate.Iterator) impls Core.EqWith(.(Core.CppRangeForIterate.Sentinel)) and .(Core.CppRangeForIterate.Iterator) impls Core.Inc and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy and .(Core.CppRangeForIterate.Sentinel) impls Core.Destroy and .(Core.CppRangeForIterate.Sentinel) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   Test(mc);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:40:10: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR:   fn End[self: Self]() -> Sentinel;
  // CHECK:STDERR:          ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE-41]]:42: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:     friend auto end(MutableAndConstRange&) -> Iterator;
  // CHECK:STDERR:                                          ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:40:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn End[self: Self]() -> Sentinel;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE-39]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(mc);
}

// --- fail_never_valid.carbon

library "[[@TEST_NAME]]";

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

// We deliberately use `Core.CppRangeForIterate` here, since we want to confirm
// that types don't implement this interface.
fn Test[R:! Core.CppRangeForIterate](unused r: R) {}

fn TestDriver(var no_begin_end: Cpp.NoBeginEnd,
              var no_begin_method: Cpp.NoBeginMethod,
              var no_end_method: Cpp.NoEndMethod,
              var no_begin_adl: Cpp.NoBeginADL,
              var no_end_adl: Cpp.NoEndADL,
              var begin_method_end_adl: Cpp.BeginMethodEndADL,
              var begin_adl_end_method: Cpp.BeginADLEndMethod,
              var begin_field_end_method: Cpp.BeginFieldEndMethod,
              var begin_method_end_field: Cpp.BeginMethodEndField,
              var begin_method_deleted: Cpp.BeginMethodDeleted,
              var end_method_deleted: Cpp.EndMethodDeleted,
              var begin_function_deleted: Cpp.BeginFunctionDeleted,
              var end_function_deleted: Cpp.EndFunctionDeleted)
{
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoBeginEnd` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_begin_end);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-19]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_begin_end);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoBeginMethod` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_begin_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-28]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_begin_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoEndMethod` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_end_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-37]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_end_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoBeginADL` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_begin_adl);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-46]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_begin_adl);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoEndADL` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_end_adl);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-55]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_end_adl);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginMethodEndADL` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_method_end_adl);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-64]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_method_end_adl);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginADLEndMethod` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_adl_end_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-73]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_adl_end_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginFieldEndMethod` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_field_end_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-82]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_field_end_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginMethodEndField` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_method_end_field);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-91]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_method_end_field);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+10]]:28: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:   173 |   Test(begin_method_deleted);
  // CHECK:STDERR:       |                            ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-122]]:8: note: 'begin' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:    44 |   auto begin() = delete;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-103]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_method_deleted);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+10]]:26: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:   185 |   Test(end_method_deleted);
  // CHECK:STDERR:       |                          ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-128]]:8: note: 'end' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:    50 |   auto end() = delete;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-115]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(end_method_deleted);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+10]]:30: error: call to deleted function 'begin' [CppInteropParseError]
  // CHECK:STDERR:   197 |   Test(begin_function_deleted);
  // CHECK:STDERR:       |                              ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-136]]:15: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:    54 |   friend auto begin(const BeginFunctionDeleted&) = delete;
  // CHECK:STDERR:       |               ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-127]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_function_deleted);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+10]]:28: error: call to deleted function 'end' [CppInteropParseError]
  // CHECK:STDERR:   209 |   Test(end_function_deleted);
  // CHECK:STDERR:       |                            ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-142]]:15: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:    60 |   friend auto end(const EndFunctionDeleted&) = delete;
  // CHECK:STDERR:       |               ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-139]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(end_function_deleted);
}

// CHECK:STDOUT: --- methods_return_same_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %facet_type.b7c: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.4e7: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.d22: type = fn_type @Optional.Get, @Optional(%T.4e7) [symbolic]
// CHECK:STDOUT:   %Optional.Get.6fc: %Optional.Get.type.d22 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.498: type = fn_type @Optional.HasValue, @Optional(%T.4e7) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.bf8: %Optional.HasValue.type.498 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.cb0: %facet_type.b7c = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f8b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.46f: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8c2: %T.as_type.as.OptionalStorage.impl.Get.type.46f = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.0ae: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.e0f: %T.as_type.as.OptionalStorage.impl.Has.type.0ae = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.87a: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.898: %T.as_type.as.OptionalStorage.impl.Some.type.87a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.ceb: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.9dd: %T.as_type.as.OptionalStorage.impl.None.type.ceb = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.e31: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.732: <witness> = lookup_impl_witness %.Self.e31, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.03d: %facet_type.b7c = impl_witness_access %Iterate.lookup_impl_witness.732, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.2cc5f1.1: type = class_type @Int, @Int(%N.fe9) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc30_47.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.2: <witness> = custom_witness (%Destroy.Op.2dd605.2), @Destroy [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.2f2: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.a64: %Int.as.Copy.impl.Op.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1a4: <witness> = impl_witness imports.%Copy.impl_witness_table.0ac, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.6cc: %facet_type.b7c = facet_value %i32, (%custom_witness.e20f1f.2, %Copy.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %Iterate_where.type.6aa: type = facet_type <@Iterate where %impl.elem0.03d = %facet_value.6cc> [concrete]
// CHECK:STDOUT:   %R.5de: %Iterate_where.type.6aa = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type.549: type = facet_access_type %R.5de [symbolic]
// CHECK:STDOUT:   %pattern_type.8bb: type = pattern_type %R.as_type.549 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Int.2cc5f1.2: type = class_type @Int, @Int(%To) [symbolic]
// CHECK:STDOUT:   %facet_type.049: type = facet_type <@IntFitsIn, @IntFitsIn(%Int.2cc5f1.2) & @AnyInt> [symbolic]
// CHECK:STDOUT:   %From.094: %facet_type.049 = symbolic_binding From, 1 [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.type.5e2: type = fn_type @From.as_type.as.ImplicitAs.impl.Convert, @From.as_type.as.ImplicitAs.impl(%To, %From.094) [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.58d: %From.as_type.as.ImplicitAs.impl.Convert.type.5e2 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %.Self.10b: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.5bf: <witness> = lookup_impl_witness %.Self.10b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d47: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.bce: <witness> = lookup_impl_witness %impl.elem0.d47, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c87: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.bce, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.846: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.d47 impls @Destroy and %impl.elem0.d47 impls @Copy and %impl.elem0.d47 impls @CppUnsafeDeref and %impl.elem0.d47 impls @EqWith, @EqWith(%impl.elem1.846) and %impl.elem0.d47 impls @Inc and %impl.elem0.c87 impls @Destroy and %impl.elem0.c87 impls @Copy and %impl.elem1.846 impls @Destroy and %impl.elem1.846 impls @Copy> [concrete]
// CHECK:STDOUT:   %R.6c6: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.edd1fe.2: <witness> = lookup_impl_witness %R.6c6, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.bdcd2d.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.d5966e.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.384: type = tuple_type (%impl.elem0.d5966e.2, %impl.elem1.bdcd2d.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.0ba: <witness> = lookup_impl_witness %tuple.type.384, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.2fe: %Destroy.type = facet_value %tuple.type.384, (%Destroy.lookup_impl_witness.0ba) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.ca4: <witness> = lookup_impl_witness %impl.elem0.d5966e.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.b10: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.ca4, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.3d8: <witness> = lookup_impl_witness %impl.elem0.b10, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.bff: <witness> = lookup_impl_witness %impl.elem0.b10, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.a30: %facet_type.b7c = facet_value %impl.elem0.b10, (%Destroy.lookup_impl_witness.3d8, %Copy.lookup_impl_witness.bff) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.5b5: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%R.6c6) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.cd2: %R.as_type.as.Iterate.impl.Next.type.5b5 = struct_value () [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.9b6: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%R.6c6) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.b29: %R.as_type.as.Iterate.impl.NewCursor.type.9b6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.ccf: <witness> = lookup_impl_witness %R.5de, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.6f6: %Iterate.type = facet_value %R.as_type.549, (%Iterate.lookup_impl_witness.ccf) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.4ee: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.6f6) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.652: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.6f6) [symbolic]
// CHECK:STDOUT:   %.75c: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.4ee, %Iterate.facet.6f6 [symbolic]
// CHECK:STDOUT:   %impl.elem2.d4c: %.75c = impl_witness_access %Iterate.lookup_impl_witness.ccf, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.ad9: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.ccf, element1 [symbolic]
// CHECK:STDOUT:   %as_type.cf9: type = facet_access_type %impl.elem1.ad9 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.314: <specific function> = specific_impl_function %impl.elem2.d4c, @Iterate.WithSelf.NewCursor(%Iterate.facet.6f6) [symbolic]
// CHECK:STDOUT:   %ptr.233: type = ptr_type %as_type.cf9 [symbolic]
// CHECK:STDOUT:   %.58a: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.652, %Iterate.facet.6f6 [symbolic]
// CHECK:STDOUT:   %impl.elem3.25e: %.58a = impl_witness_access %Iterate.lookup_impl_witness.ccf, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.388: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.6cc) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.80d: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.6f7, @T.as_type.as.OptionalStorage.impl(%facet_value.6cc) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.26b: %OptionalStorage.type = facet_value %i32, (%OptionalStorage.impl_witness.80d) [concrete]
// CHECK:STDOUT:   %Optional.af5: type = class_type @Optional, @Optional(%OptionalStorage.facet.26b) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.85b: <specific function> = specific_impl_function %impl.elem3.25e, @Iterate.WithSelf.Next(%Iterate.facet.6f6) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.4a8: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.26b) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.3cd: %Optional.HasValue.type.4a8 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.f91: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.26b) [concrete]
// CHECK:STDOUT:   %Optional.Get.4e5: %Optional.Get.type.f91 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.95dcf3.2: %Destroy.type = facet_value %i32, (%custom_witness.e20f1f.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.30b: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.95dcf3.2) [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.f8d: type = struct_type {.value: %MaybeUnformed.30b, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.3cd, @Optional.HasValue(%OptionalStorage.facet.26b) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.4e5, @Optional.Get(%OptionalStorage.facet.26b) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.e37: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.594be1.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.2cc5f1.1)> [symbolic]
// CHECK:STDOUT:   %U.3d0: %ImplicitAs.type.594be1.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.484532.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N.fe9, %U.3d0) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.eec262.1: %Int.as.AddAssignWith.impl.Op.type.484532.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.484532.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N.fe9, %U.3d0) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.eec262.2: %Int.as.AddAssignWith.impl.Op.type.484532.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.e6c: type = facet_type <@IntFitsIn, @IntFitsIn(%i32) & @AnyInt> [concrete]
// CHECK:STDOUT:   %custom_witness.218: <witness> = custom_witness (), @IntFitsIn, @IntFitsIn(%i32) [concrete]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.type.5c3: type = fn_type @Int.as.AnyInt.impl.AsInt, @Int.as.AnyInt.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.164: %Int.as.AnyInt.impl.AsInt.type.5c3 = struct_value () [symbolic]
// CHECK:STDOUT:   %AnyInt.impl_witness.a35: <witness> = impl_witness imports.%AnyInt.impl_witness_table, @Int.as.AnyInt.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.1bd: %facet_type.e6c = facet_value %i32, (%custom_witness.218, %AnyInt.impl_witness.a35) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.eb2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f15, @From.as_type.as.ImplicitAs.impl(%int_32, %facet_value.1bd) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9ea: %ImplicitAs.type.360 = facet_value %i32, (%ImplicitAs.impl_witness.eb2) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.a8e: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.9ea) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.b09202.1: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.9ea) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bbd220.1: %Int.as.AddAssignWith.impl.Op.type.b09202.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.b09202.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.9ea) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bbd220.2: %Int.as.AddAssignWith.impl.Op.type.b09202.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.c0a: %AddAssignWith.type.e37 = facet_value %i32, (%AddAssignWith.impl_witness.a8e) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.94b: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet.c0a) [concrete]
// CHECK:STDOUT:   %.35b: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.94b, %AddAssignWith.facet.c0a [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.f3f3f7.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.bbd220.2, @Int.as.AddAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet.9ea) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.f3f3f7.2: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.bbd220.1, @Int.as.AddAssignWith.impl.Op.2(%int_32, %ImplicitAs.facet.9ea) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.8: type = fn_type @Destroy.Op.loc34_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.8: %Destroy.Op.type.790b3b.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.730: <witness> = lookup_impl_witness %impl.elem1.ad9, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.183: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.ad9) [symbolic]
// CHECK:STDOUT:   %.179: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.183, %impl.elem1.ad9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.636: %.179 = impl_witness_access %Destroy.lookup_impl_witness.730, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.56c: <specific function> = specific_impl_function %impl.elem0.636, @Destroy.WithSelf.Op(%impl.elem1.ad9) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.b76: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.631: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.826: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %MutableRange.Begin.type: type = fn_type @MutableRange.Begin [concrete]
// CHECK:STDOUT:   %MutableRange.Begin: %MutableRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %MutableRange.End.type: type = fn_type @MutableRange.End [concrete]
// CHECK:STDOUT:   %MutableRange.End: %MutableRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.544: <witness> = custom_witness (%Iterator.826, %Iterator.826, %MutableRange.Begin, %MutableRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.134d4f.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.f056ae.1: %Iterator.Op.type.134d4f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b4c: <witness> = custom_witness (%i32, %Iterator.Op.f056ae.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.134d4f.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.f056ae.2: %Iterator.Op.type.134d4f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ef9: <witness> = custom_witness (%Iterator.Op.f056ae.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.134d4f.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.f056ae.3: %Iterator.Op.type.134d4f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3a5: <witness> = custom_witness (%Iterator.Op.f056ae.3), @Copy [concrete]
// CHECK:STDOUT:   %Equal.type.eace26.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.8c4faf.1: %Equal.type.eace26.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.ca4b73.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.6eaf3e.1: %NotEqual.type.ca4b73.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.baa: <witness> = custom_witness (%Equal.8c4faf.1, %NotEqual.6eaf3e.1), @EqWith, @EqWith(%Iterator.826) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.134d4f.4: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.f056ae.4: %Iterator.Op.type.134d4f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f9b: <witness> = custom_witness (%Iterator.Op.f056ae.4), @Inc [concrete]
// CHECK:STDOUT:   %facet_value.31b: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.e20f1f.2, %Copy.impl_witness.1a4, %custom_witness.544, %custom_witness.ef9, %custom_witness.3a5, %custom_witness.b4c, %custom_witness.baa, %custom_witness.f9b) [concrete]
// CHECK:STDOUT:   %tuple.type.17d: type = tuple_type (%Iterator.826, %Iterator.826) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.9: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.9: %Destroy.Op.type.790b3b.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.9: <witness> = custom_witness (%Destroy.Op.2dd605.9), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.0ec: %Destroy.type = facet_value %tuple.type.17d, (%custom_witness.e20f1f.9) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.a27: <witness> = impl_witness imports.%Iterate.impl_witness_table.9f0, @R.as_type.as.Iterate.impl(%facet_value.31b) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.872: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%facet_value.31b) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.b35: %R.as_type.as.Iterate.impl.NewCursor.type.872 = struct_value () [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.4d0: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%facet_value.31b) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.dbe: %R.as_type.as.Iterate.impl.Next.type.4d0 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.989: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.a27) [concrete]
// CHECK:STDOUT:   %facet_value.b9f: %Iterate_where.type.6aa = facet_value %MutableRange, (%Iterate.impl_witness.a27) [concrete]
// CHECK:STDOUT:   %Iterator.4f0: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.83f: <witness> = custom_witness (%Iterator.4f0, %Iterator.4f0, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.9f2350.1: type = fn_type @Iterator.Op.5 [concrete]
// CHECK:STDOUT:   %Iterator.Op.7b7347.1: %Iterator.Op.type.9f2350.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.852: <witness> = custom_witness (%i32, %Iterator.Op.7b7347.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.9f2350.2: type = fn_type @Iterator.Op.6 [concrete]
// CHECK:STDOUT:   %Iterator.Op.7b7347.2: %Iterator.Op.type.9f2350.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df0: <witness> = custom_witness (%Iterator.Op.7b7347.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.9f2350.3: type = fn_type @Iterator.Op.7 [concrete]
// CHECK:STDOUT:   %Iterator.Op.7b7347.3: %Iterator.Op.type.9f2350.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f14: <witness> = custom_witness (%Iterator.Op.7b7347.3), @Copy [concrete]
// CHECK:STDOUT:   %Equal.type.eace26.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.8c4faf.2: %Equal.type.eace26.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.ca4b73.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.6eaf3e.2: %NotEqual.type.ca4b73.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f7c: <witness> = custom_witness (%Equal.8c4faf.2, %NotEqual.6eaf3e.2), @EqWith, @EqWith(%Iterator.4f0) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.9f2350.4: type = fn_type @Iterator.Op.8 [concrete]
// CHECK:STDOUT:   %Iterator.Op.7b7347.4: %Iterator.Op.type.9f2350.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c18: <witness> = custom_witness (%Iterator.Op.7b7347.4), @Inc [concrete]
// CHECK:STDOUT:   %facet_value.c7f: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.e20f1f.2, %Copy.impl_witness.1a4, %custom_witness.83f, %custom_witness.df0, %custom_witness.f14, %custom_witness.852, %custom_witness.f7c, %custom_witness.c18) [concrete]
// CHECK:STDOUT:   %tuple.type.d1a: type = tuple_type (%Iterator.4f0, %Iterator.4f0) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.10: type = fn_type @Destroy.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.10: %Destroy.Op.type.790b3b.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.10: <witness> = custom_witness (%Destroy.Op.2dd605.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.72a: %Destroy.type = facet_value %tuple.type.d1a, (%custom_witness.e20f1f.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.7a3: <witness> = impl_witness imports.%Iterate.impl_witness_table.9f0, @R.as_type.as.Iterate.impl(%facet_value.c7f) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.8bf: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%facet_value.c7f) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.130: %R.as_type.as.Iterate.impl.NewCursor.type.8bf = struct_value () [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.77d: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%facet_value.c7f) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.fb6: %R.as_type.as.Iterate.impl.Next.type.77d = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.bdf: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.7a3) [concrete]
// CHECK:STDOUT:   %facet_value.bbe: %Iterate_where.type.6aa = facet_value %ConstRange, (%Iterate.impl_witness.7a3) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.5d1: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.989) [concrete]
// CHECK:STDOUT:   %.34e: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.5d1, %Iterate.facet.989 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.specific_fn.827: <specific function> = specific_function %R.as_type.as.Iterate.impl.NewCursor.b35, @R.as_type.as.Iterate.impl.NewCursor(%facet_value.31b) [concrete]
// CHECK:STDOUT:   %complete_type.3ec: <witness> = complete_type_witness %tuple.type.17d [concrete]
// CHECK:STDOUT:   %ptr.f42: type = ptr_type %tuple.type.17d [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.68f: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.989) [concrete]
// CHECK:STDOUT:   %.42b: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.68f, %Iterate.facet.989 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.specific_fn.712: <specific function> = specific_function %R.as_type.as.Iterate.impl.Next.dbe, @R.as_type.as.Iterate.impl.Next(%facet_value.31b) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.511: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.0ec) [concrete]
// CHECK:STDOUT:   %.824: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.511, %Destroy.facet.0ec [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.bf8: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.bdf) [concrete]
// CHECK:STDOUT:   %.3d0: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.bf8, %Iterate.facet.bdf [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.specific_fn.625: <specific function> = specific_function %R.as_type.as.Iterate.impl.NewCursor.130, @R.as_type.as.Iterate.impl.NewCursor(%facet_value.c7f) [concrete]
// CHECK:STDOUT:   %complete_type.f2c: <witness> = complete_type_witness %tuple.type.d1a [concrete]
// CHECK:STDOUT:   %ptr.11b: type = ptr_type %tuple.type.d1a [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.3bc: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.bdf) [concrete]
// CHECK:STDOUT:   %.da0: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.3bc, %Iterate.facet.bdf [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.specific_fn.272: <specific function> = specific_function %R.as_type.as.Iterate.impl.Next.fb6, @R.as_type.as.Iterate.impl.Next(%facet_value.c7f) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.fa3: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.72a) [concrete]
// CHECK:STDOUT:   %.b6b: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.fa3, %Destroy.facet.72a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.0b9: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.498) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.bf8)]
// CHECK:STDOUT:   %Core.import_ref.61f: @Optional.%Optional.Get.type (%Optional.Get.type.d22) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.6fc)]
// CHECK:STDOUT:   %Core.import_ref.2b5: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.f8b)]
// CHECK:STDOUT:   %Core.import_ref.b0a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.ceb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.9dd)]
// CHECK:STDOUT:   %Core.import_ref.3fc: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.87a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.898)]
// CHECK:STDOUT:   %Core.import_ref.986: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.0ae) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.e0f)]
// CHECK:STDOUT:   %Core.import_ref.557: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.46f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8c2)]
// CHECK:STDOUT:   %Core.import_ref.999 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.6f7 = impl_witness_table (%Core.import_ref.2b5, %Core.import_ref.b0a, %Core.import_ref.3fc, %Core.import_ref.986, %Core.import_ref.557, %Core.import_ref.999), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.55f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.2f2) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.a64)]
// CHECK:STDOUT:   %Copy.impl_witness_table.0ac = impl_witness_table (%Core.import_ref.55f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.3b9: @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert.type (%From.as_type.as.ImplicitAs.impl.Convert.type.5e2) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert (constants.%From.as_type.as.ImplicitAs.impl.Convert.58d)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f15 = impl_witness_table (%Core.import_ref.3b9), @From.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.89a: %facet_type.b7c = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.a30)]
// CHECK:STDOUT:   %Core.import_ref.9ea: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.2fe)]
// CHECK:STDOUT:   %Core.import_ref.a6c: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor.type (%R.as_type.as.Iterate.impl.NewCursor.type.9b6) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor (constants.%R.as_type.as.Iterate.impl.NewCursor.b29)]
// CHECK:STDOUT:   %Core.import_ref.a02: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next.type (%R.as_type.as.Iterate.impl.Next.type.5b5) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next (constants.%R.as_type.as.Iterate.impl.Next.cd2)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.9f0 = impl_witness_table (%Core.import_ref.89a, %Core.import_ref.9ea, %Core.import_ref.a6c, %Core.import_ref.a02), @R.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.328: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.484532.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.eec262.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.328), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.d27: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.484532.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.eec262.2)]
// CHECK:STDOUT:   %Core.import_ref.47b: Core.IntLiteral = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%N (constants.%N.fe9)]
// CHECK:STDOUT:   %Core.import_ref.d13: @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt.type (%Int.as.AnyInt.impl.AsInt.type.5c3) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt (constants.%Int.as.AnyInt.impl.AsInt.164)]
// CHECK:STDOUT:   %AnyInt.impl_witness_table = impl_witness_table (%Core.import_ref.47b, %Core.import_ref.d13), @Int.as.AnyInt.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc30_10.2: %Iterate_where.type.6aa) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc30_10.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.ccf)]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %R.as_type.loc30_55.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet (constants.%Iterate.facet.6f6)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.4ee)]
// CHECK:STDOUT:   %.loc34_19.12: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet [symbolic = %.loc34_19.12 (constants.%.75c)]
// CHECK:STDOUT:   %impl.elem2.loc34_19.2: @Test.%.loc34_19.12 (%.75c) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_19.2 (constants.%impl.elem2.d4c)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4: <specific function> = specific_impl_function %impl.elem2.loc34_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.314)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.233)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.652)]
// CHECK:STDOUT:   %.loc34_19.13: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet [symbolic = %.loc34_19.13 (constants.%.58a)]
// CHECK:STDOUT:   %impl.elem3.loc34_19.2: @Test.%.loc34_19.13 (%.58a) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.25e)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5: <specific function> = specific_impl_function %impl.elem3.loc34_19.2, @Iterate.WithSelf.Next(%Iterate.facet) [symbolic = %specific_impl_fn.loc34_19.5 (constants.%specific_impl_fn.85b)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.183)]
// CHECK:STDOUT:   %.loc34_19.14: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc34_19.14 (constants.%.179)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.730)]
// CHECK:STDOUT:   %impl.elem0.loc34_19.2: @Test.%.loc34_19.14 (%.179) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_19.2 (constants.%impl.elem0.636)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6: <specific function> = specific_impl_function %impl.elem0.loc34_19.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc34_19.6 (constants.%specific_impl_fn.56c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @Test.%R.as_type.loc30_55.1 (%R.as_type.549)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.104 = value_binding_pattern i [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @Test.%R.as_type.loc30_55.1 (%R.as_type.549) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc34_19.1: @Test.%.loc34_19.12 (%.75c) = impl_witness_access constants.%Iterate.lookup_impl_witness.ccf, element2 [symbolic = %impl.elem2.loc34_19.2 (constants.%impl.elem2.d4c)]
// CHECK:STDOUT:     %bound_method.loc34_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc34_19.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.1: <specific function> = specific_impl_function %impl.elem2.loc34_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.6f6) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.314)]
// CHECK:STDOUT:     %bound_method.loc34_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.1
// CHECK:STDOUT:     %var: ref @Test.%as_type (%as_type.cf9) = var invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @Test.%as_type (%as_type.cf9) to %var = call %bound_method.loc34_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @Test.%ptr (%ptr.233) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc34_19.1: @Test.%.loc34_19.13 (%.58a) = impl_witness_access constants.%Iterate.lookup_impl_witness.ccf, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.25e)]
// CHECK:STDOUT:     %bound_method.loc34_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc34_19.1
// CHECK:STDOUT:     %.loc34_19.1: %Destroy.type = converted constants.%as_type.cf9, constants.%impl.elem1.ad9 [symbolic = %impl.elem1 (constants.%impl.elem1.ad9)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.2: <specific function> = specific_impl_function %impl.elem3.loc34_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.6f6) [symbolic = %specific_impl_fn.loc34_19.5 (constants.%specific_impl_fn.85b)]
// CHECK:STDOUT:     %bound_method.loc34_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.2
// CHECK:STDOUT:     %.loc34_19.2: ref %Optional.af5 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.af5 to %.loc34_19.2 = call %bound_method.loc34_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc34_19.3: ref %Optional.af5 = temporary %.loc34_19.2, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc34_19.4: %Optional.HasValue.type.4a8 = specific_constant imports.%Core.import_ref.0b9, @Optional(constants.%OptionalStorage.facet.26b) [concrete = constants.%Optional.HasValue.3cd]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.4a8 = name_ref HasValue, %.loc34_19.4 [concrete = constants.%Optional.HasValue.3cd]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc34_19.3, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.26b) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc34_19.5: <bound method> = bound_method %.loc34_19.3, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc34_19.5: %Optional.af5 = acquire_value %.loc34_19.3
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc34_19.5(%.loc34_19.5)
// CHECK:STDOUT:     %.loc34_19.6: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc34_19.7: bool = converted %Optional.HasValue.call, %.loc34_19.6
// CHECK:STDOUT:     if %.loc34_19.7 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc34_19.8: %Optional.Get.type.f91 = specific_constant imports.%Core.import_ref.61f, @Optional(constants.%OptionalStorage.facet.26b) [concrete = constants.%Optional.Get.4e5]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.f91 = name_ref Get, %.loc34_19.8 [concrete = constants.%Optional.Get.4e5]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc34_19.3, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.26b) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc34_19.6: <bound method> = bound_method %.loc34_19.3, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc34_19.9: %Optional.af5 = acquire_value %.loc34_19.3
// CHECK:STDOUT:     %Optional.Get.call: init %i32 = call %bound_method.loc34_19.6(%.loc34_19.9)
// CHECK:STDOUT:     %i32.loc34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc34_19.10: %i32 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc34_19.11: %i32 = converted %Optional.Get.call, %.loc34_19.10
// CHECK:STDOUT:     %i: %i32 = value_binding i, %.loc34_19.11
// CHECK:STDOUT:     %sum.ref: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %i32 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc35: %.35b = impl_witness_access constants.%AddAssignWith.impl_witness.a8e, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.bbd220.2]
// CHECK:STDOUT:     %bound_method.loc35_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc35
// CHECK:STDOUT:     %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.9ea) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.f3f3f7.1]
// CHECK:STDOUT:     %bound_method.loc35_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc35
// CHECK:STDOUT:     %.loc35: %Int.as.AddAssignWith.impl.Op.type.b09202.1 = specific_constant imports.%Core.Op.d27, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.9ea) [concrete = constants.%Int.as.AddAssignWith.impl.Op.bbd220.1]
// CHECK:STDOUT:     %Op.ref: %Int.as.AddAssignWith.impl.Op.type.b09202.1 = name_ref Op, %.loc35 [concrete = constants.%Int.as.AddAssignWith.impl.Op.bbd220.1]
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %sum.ref, %Op.ref
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.9ea) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.f3f3f7.2]
// CHECK:STDOUT:     %bound_method.loc35_9.3: <bound method> = bound_method %sum.ref, %Int.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc35_9.3(%sum.ref, %i.ref)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc34: <bound method> = bound_method %.loc34_19.3, constants.%Destroy.Op.2dd605.8
// CHECK:STDOUT:     %Destroy.Op.call.loc34: init %empty_tuple.type = call %Destroy.Op.bound.loc34(%.loc34_19.3)
// CHECK:STDOUT:     %impl.elem0.loc34_19.1: @Test.%.loc34_19.14 (%.179) = impl_witness_access constants.%Destroy.lookup_impl_witness.730, element0 [symbolic = %impl.elem0.loc34_19.2 (constants.%impl.elem0.636)]
// CHECK:STDOUT:     %bound_method.loc34_19.7: <bound method> = bound_method %var, %impl.elem0.loc34_19.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.3: <specific function> = specific_impl_function %impl.elem0.loc34_19.1, @Destroy.WithSelf.Op(constants.%impl.elem1.ad9) [symbolic = %specific_impl_fn.loc34_19.6 (constants.%specific_impl_fn.56c)]
// CHECK:STDOUT:     %bound_method.loc34_19.8: <bound method> = bound_method %var, %specific_impl_fn.loc34_19.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc34_19.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.1(%self.param: ref %MaybeUnformed.30b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.2(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.3(%self.param: ref %struct_type.value.has_value.f8d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.4(%self.param: ref %DefaultOptionalStorage.388) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.5(%self.param: ref %Optional.af5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R.5de) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%R.5de
// CHECK:STDOUT:   %R.as_type.loc30_55.1 => constants.%R.as_type.549
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8bb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.b9f) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%facet_value.b9f
// CHECK:STDOUT:   %R.as_type.loc30_55.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b76
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.a27
// CHECK:STDOUT:   %Iterate.facet => constants.%Iterate.facet.989
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.5d1
// CHECK:STDOUT:   %.loc34_19.12 => constants.%.34e
// CHECK:STDOUT:   %impl.elem2.loc34_19.2 => constants.%R.as_type.as.Iterate.impl.NewCursor.b35
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn.827
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.0ec
// CHECK:STDOUT:   %as_type => constants.%tuple.type.17d
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.3ec
// CHECK:STDOUT:   %ptr => constants.%ptr.f42
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.68f
// CHECK:STDOUT:   %.loc34_19.13 => constants.%.42b
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%R.as_type.as.Iterate.impl.Next.dbe
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5 => constants.%R.as_type.as.Iterate.impl.Next.specific_fn.712
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.511
// CHECK:STDOUT:   %.loc34_19.14 => constants.%.824
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e20f1f.9
// CHECK:STDOUT:   %impl.elem0.loc34_19.2 => constants.%Destroy.Op.2dd605.9
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6 => constants.%Destroy.Op.2dd605.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.bbe) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%facet_value.bbe
// CHECK:STDOUT:   %R.as_type.loc30_55.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.631
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.7a3
// CHECK:STDOUT:   %Iterate.facet => constants.%Iterate.facet.bdf
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.bf8
// CHECK:STDOUT:   %.loc34_19.12 => constants.%.3d0
// CHECK:STDOUT:   %impl.elem2.loc34_19.2 => constants.%R.as_type.as.Iterate.impl.NewCursor.130
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn.625
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.72a
// CHECK:STDOUT:   %as_type => constants.%tuple.type.d1a
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.f2c
// CHECK:STDOUT:   %ptr => constants.%ptr.11b
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.3bc
// CHECK:STDOUT:   %.loc34_19.13 => constants.%.da0
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%R.as_type.as.Iterate.impl.Next.fb6
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5 => constants.%R.as_type.as.Iterate.impl.Next.specific_fn.272
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.fa3
// CHECK:STDOUT:   %.loc34_19.14 => constants.%.b6b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e20f1f.10
// CHECK:STDOUT:   %impl.elem0.loc34_19.2 => constants.%Destroy.Op.2dd605.10
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6 => constants.%Destroy.Op.2dd605.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- methods_return_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.4e7: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.d22: type = fn_type @Optional.Get, @Optional(%T.4e7) [symbolic]
// CHECK:STDOUT:   %Optional.Get.6fc: %Optional.Get.type.d22 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.498: type = fn_type @Optional.HasValue, @Optional(%T.4e7) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.bf8: %Optional.HasValue.type.498 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.cb0: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f8b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.46f: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8c2: %T.as_type.as.OptionalStorage.impl.Get.type.46f = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.0ae: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.e0f: %T.as_type.as.OptionalStorage.impl.Has.type.0ae = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.87a: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.898: %T.as_type.as.OptionalStorage.impl.Some.type.87a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.ceb: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.9dd: %T.as_type.as.OptionalStorage.impl.None.type.ceb = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.e31: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.732: <witness> = lookup_impl_witness %.Self.e31, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.03d: %facet_type = impl_witness_access %Iterate.lookup_impl_witness.732, element0 [symbolic_self]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.bef: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor.type: type = fn_type @ValueType.cpp_destructor [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor: %ValueType.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.43d658.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.cc70d5.1: %ValueType.Op.type.43d658.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a98c63.1: <witness> = custom_witness (%ValueType.Op.cc70d5.1), @Destroy [concrete]
// CHECK:STDOUT:   %const.3fd: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.e0a: type = ptr_type %const.3fd [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.43d658.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.cc70d5.2: %ValueType.Op.type.43d658.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b0507b.1: <witness> = custom_witness (%ValueType.Op.cc70d5.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.ea2: %facet_type = facet_value %ValueType, (%custom_witness.a98c63.1, %custom_witness.b0507b.1) [concrete]
// CHECK:STDOUT:   %Iterate_where.type.630: type = facet_type <@Iterate where %impl.elem0.03d = %facet_value.ea2> [concrete]
// CHECK:STDOUT:   %R.2da: %Iterate_where.type.630 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type.339: type = facet_access_type %R.2da [symbolic]
// CHECK:STDOUT:   %pattern_type.df0: type = pattern_type %R.as_type.339 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %.Self.10b: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.5bf: <witness> = lookup_impl_witness %.Self.10b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d47: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.bce: <witness> = lookup_impl_witness %impl.elem0.d47, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c87: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.bce, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.846: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.d47 impls @Destroy and %impl.elem0.d47 impls @Copy and %impl.elem0.d47 impls @CppUnsafeDeref and %impl.elem0.d47 impls @EqWith, @EqWith(%impl.elem1.846) and %impl.elem0.d47 impls @Inc and %impl.elem0.c87 impls @Destroy and %impl.elem0.c87 impls @Copy and %impl.elem1.846 impls @Destroy and %impl.elem1.846 impls @Copy> [concrete]
// CHECK:STDOUT:   %R.6c6: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.edd1fe.2: <witness> = lookup_impl_witness %R.6c6, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.bdcd2d.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.d5966e.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.384: type = tuple_type (%impl.elem0.d5966e.2, %impl.elem1.bdcd2d.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.0ba: <witness> = lookup_impl_witness %tuple.type.384, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.2fe: %Destroy.type = facet_value %tuple.type.384, (%Destroy.lookup_impl_witness.0ba) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.ca4: <witness> = lookup_impl_witness %impl.elem0.d5966e.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.b10: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.ca4, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.3d8: <witness> = lookup_impl_witness %impl.elem0.b10, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.bff: <witness> = lookup_impl_witness %impl.elem0.b10, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.a30: %facet_type = facet_value %impl.elem0.b10, (%Destroy.lookup_impl_witness.3d8, %Copy.lookup_impl_witness.bff) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.5b5: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%R.6c6) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.cd2: %R.as_type.as.Iterate.impl.Next.type.5b5 = struct_value () [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.9b6: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%R.6c6) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.b29: %R.as_type.as.Iterate.impl.NewCursor.type.9b6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.695: <witness> = lookup_impl_witness %R.2da, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.8c4: %Iterate.type = facet_value %R.as_type.339, (%Iterate.lookup_impl_witness.695) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.210: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.8c4) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.ad2: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.8c4) [symbolic]
// CHECK:STDOUT:   %.9b8: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.210, %Iterate.facet.8c4 [symbolic]
// CHECK:STDOUT:   %impl.elem2.377: %.9b8 = impl_witness_access %Iterate.lookup_impl_witness.695, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.bdd: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.695, element1 [symbolic]
// CHECK:STDOUT:   %as_type.254: type = facet_access_type %impl.elem1.bdd [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ed7: <specific function> = specific_impl_function %impl.elem2.377, @Iterate.WithSelf.NewCursor(%Iterate.facet.8c4) [symbolic]
// CHECK:STDOUT:   %ptr.864: type = ptr_type %as_type.254 [symbolic]
// CHECK:STDOUT:   %.a74: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.ad2, %Iterate.facet.8c4 [symbolic]
// CHECK:STDOUT:   %impl.elem3.5d8: %.a74 = impl_witness_access %Iterate.lookup_impl_witness.695, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.448: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.ea2) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.444: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.6f7, @T.as_type.as.OptionalStorage.impl(%facet_value.ea2) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.7c3: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.444) [concrete]
// CHECK:STDOUT:   %Optional.4fe: type = class_type @Optional, @Optional(%OptionalStorage.facet.7c3) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.327: <specific function> = specific_impl_function %impl.elem3.5d8, @Iterate.WithSelf.Next(%Iterate.facet.8c4) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.fdd: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.7c3) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.fe2: %Optional.HasValue.type.fdd = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.43b: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.7c3) [concrete]
// CHECK:STDOUT:   %Optional.Get.4c3: %Optional.Get.type.43b = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.3cd: %Destroy.type = facet_value %ValueType, (%custom_witness.a98c63.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.5ea: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.3cd) [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.40c: type = struct_type {.value: %MaybeUnformed.5ea, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.fe2, @Optional.HasValue(%OptionalStorage.facet.7c3) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.4c3, @Optional.Get(%OptionalStorage.facet.7c3) [concrete]
// CHECK:STDOUT:   %ValueType.cpp_operator.type: type = fn_type @ValueType.cpp_operator [concrete]
// CHECK:STDOUT:   %ValueType.cpp_operator: %ValueType.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.6: type = fn_type @Destroy.Op.loc44_29.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.6: %Destroy.Op.type.790b3b.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.185: <witness> = lookup_impl_witness %impl.elem1.bdd, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.a36: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.bdd) [symbolic]
// CHECK:STDOUT:   %.490: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.a36, %impl.elem1.bdd [symbolic]
// CHECK:STDOUT:   %impl.elem0.d3e: %.490 = impl_witness_access %Destroy.lookup_impl_witness.185, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e07: <specific function> = specific_impl_function %impl.elem0.d3e, @Destroy.WithSelf.Op(%impl.elem1.bdd) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.b76: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.631: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %Iterator.826: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Sentinel.6eb: type = class_type @Sentinel.1 [concrete]
// CHECK:STDOUT:   %MutableRange.Begin.type: type = fn_type @MutableRange.Begin [concrete]
// CHECK:STDOUT:   %MutableRange.Begin: %MutableRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %MutableRange.End.type: type = fn_type @MutableRange.End [concrete]
// CHECK:STDOUT:   %MutableRange.End: %MutableRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.292: <witness> = custom_witness (%Iterator.826, %Sentinel.6eb, %MutableRange.Begin, %MutableRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.134d4f.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.f056ae.1: %Iterator.Op.type.134d4f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.675: <witness> = custom_witness (%ValueType, %Iterator.Op.f056ae.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.134d4f.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.f056ae.2: %Iterator.Op.type.134d4f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ef99ee.1: <witness> = custom_witness (%Iterator.Op.f056ae.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.134d4f.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.f056ae.3: %Iterator.Op.type.134d4f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3a5b7b.1: <witness> = custom_witness (%Iterator.Op.f056ae.3), @Copy [concrete]
// CHECK:STDOUT:   %Equal.type.eace26.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.8c4faf.1: %Equal.type.eace26.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.ca4b73.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.6eaf3e.1: %NotEqual.type.ca4b73.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e77: <witness> = custom_witness (%Equal.8c4faf.1, %NotEqual.6eaf3e.1), @EqWith, @EqWith(%Sentinel.6eb) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.134d4f.4: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.f056ae.4: %Iterator.Op.type.134d4f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f9b: <witness> = custom_witness (%Iterator.Op.f056ae.4), @Inc [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.134d4f.1: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.f056ae.1: %Sentinel.Op.type.134d4f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ef99ee.2: <witness> = custom_witness (%Sentinel.Op.f056ae.1), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.134d4f.2: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.f056ae.2: %Sentinel.Op.type.134d4f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3a5b7b.2: <witness> = custom_witness (%Sentinel.Op.f056ae.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.4eb: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.a98c63.1, %custom_witness.b0507b.1, %custom_witness.292, %custom_witness.ef99ee.1, %custom_witness.3a5b7b.1, %custom_witness.675, %custom_witness.e77, %custom_witness.f9b, %custom_witness.ef99ee.2, %custom_witness.3a5b7b.2) [concrete]
// CHECK:STDOUT:   %tuple.type.c00: type = tuple_type (%Iterator.826, %Sentinel.6eb) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.7: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.7: %Destroy.Op.type.790b3b.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.7: <witness> = custom_witness (%Destroy.Op.2dd605.7), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.0cd: %Destroy.type = facet_value %tuple.type.c00, (%custom_witness.e20f1f.7) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.5a3: <witness> = impl_witness imports.%Iterate.impl_witness_table.9f0, @R.as_type.as.Iterate.impl(%facet_value.4eb) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.1df: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%facet_value.4eb) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.dcd: %R.as_type.as.Iterate.impl.NewCursor.type.1df = struct_value () [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.390: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%facet_value.4eb) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.2f5: %R.as_type.as.Iterate.impl.Next.type.390 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.269: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.5a3) [concrete]
// CHECK:STDOUT:   %facet_value.035: %Iterate_where.type.630 = facet_value %MutableRange, (%Iterate.impl_witness.5a3) [concrete]
// CHECK:STDOUT:   %Iterator.4f0: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.45e: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8cb: <witness> = custom_witness (%Iterator.4f0, %Sentinel.45e, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.9f2350.1: type = fn_type @Iterator.Op.5 [concrete]
// CHECK:STDOUT:   %Iterator.Op.7b7347.1: %Iterator.Op.type.9f2350.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.762: <witness> = custom_witness (%ValueType, %Iterator.Op.7b7347.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.9f2350.2: type = fn_type @Iterator.Op.6 [concrete]
// CHECK:STDOUT:   %Iterator.Op.7b7347.2: %Iterator.Op.type.9f2350.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df0c4f.1: <witness> = custom_witness (%Iterator.Op.7b7347.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.9f2350.3: type = fn_type @Iterator.Op.7 [concrete]
// CHECK:STDOUT:   %Iterator.Op.7b7347.3: %Iterator.Op.type.9f2350.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f14c15.1: <witness> = custom_witness (%Iterator.Op.7b7347.3), @Copy [concrete]
// CHECK:STDOUT:   %Equal.type.eace26.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.8c4faf.2: %Equal.type.eace26.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.ca4b73.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.6eaf3e.2: %NotEqual.type.ca4b73.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d24: <witness> = custom_witness (%Equal.8c4faf.2, %NotEqual.6eaf3e.2), @EqWith, @EqWith(%Sentinel.45e) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.9f2350.4: type = fn_type @Iterator.Op.8 [concrete]
// CHECK:STDOUT:   %Iterator.Op.7b7347.4: %Iterator.Op.type.9f2350.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c18: <witness> = custom_witness (%Iterator.Op.7b7347.4), @Inc [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.9f2350.1: type = fn_type @Sentinel.Op.3 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.7b7347.1: %Sentinel.Op.type.9f2350.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df0c4f.2: <witness> = custom_witness (%Sentinel.Op.7b7347.1), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.9f2350.2: type = fn_type @Sentinel.Op.4 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.7b7347.2: %Sentinel.Op.type.9f2350.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f14c15.2: <witness> = custom_witness (%Sentinel.Op.7b7347.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.089: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.a98c63.1, %custom_witness.b0507b.1, %custom_witness.8cb, %custom_witness.df0c4f.1, %custom_witness.f14c15.1, %custom_witness.762, %custom_witness.d24, %custom_witness.c18, %custom_witness.df0c4f.2, %custom_witness.f14c15.2) [concrete]
// CHECK:STDOUT:   %tuple.type.6b1: type = tuple_type (%Iterator.4f0, %Sentinel.45e) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.8: type = fn_type @Destroy.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.8: %Destroy.Op.type.790b3b.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.8: <witness> = custom_witness (%Destroy.Op.2dd605.8), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.018: %Destroy.type = facet_value %tuple.type.6b1, (%custom_witness.e20f1f.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.c99: <witness> = impl_witness imports.%Iterate.impl_witness_table.9f0, @R.as_type.as.Iterate.impl(%facet_value.089) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.d53: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%facet_value.089) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.017: %R.as_type.as.Iterate.impl.NewCursor.type.d53 = struct_value () [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.4b3: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%facet_value.089) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.c8f: %R.as_type.as.Iterate.impl.Next.type.4b3 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.780: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.c99) [concrete]
// CHECK:STDOUT:   %facet_value.2f3: %Iterate_where.type.630 = facet_value %ConstRange, (%Iterate.impl_witness.c99) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.635: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.269) [concrete]
// CHECK:STDOUT:   %.3fa: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.635, %Iterate.facet.269 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.specific_fn.55e: <specific function> = specific_function %R.as_type.as.Iterate.impl.NewCursor.dcd, @R.as_type.as.Iterate.impl.NewCursor(%facet_value.4eb) [concrete]
// CHECK:STDOUT:   %complete_type.dee: <witness> = complete_type_witness %tuple.type.c00 [concrete]
// CHECK:STDOUT:   %ptr.993: type = ptr_type %tuple.type.c00 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.ba6: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.269) [concrete]
// CHECK:STDOUT:   %.dcc: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.ba6, %Iterate.facet.269 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.specific_fn.370: <specific function> = specific_function %R.as_type.as.Iterate.impl.Next.2f5, @R.as_type.as.Iterate.impl.Next(%facet_value.4eb) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.6fd: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.0cd) [concrete]
// CHECK:STDOUT:   %.ad7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.6fd, %Destroy.facet.0cd [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b98: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.780) [concrete]
// CHECK:STDOUT:   %.25f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b98, %Iterate.facet.780 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.specific_fn.26d: <specific function> = specific_function %R.as_type.as.Iterate.impl.NewCursor.017, @R.as_type.as.Iterate.impl.NewCursor(%facet_value.089) [concrete]
// CHECK:STDOUT:   %complete_type.74d: <witness> = complete_type_witness %tuple.type.6b1 [concrete]
// CHECK:STDOUT:   %ptr.f19: type = ptr_type %tuple.type.6b1 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.8d9: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.780) [concrete]
// CHECK:STDOUT:   %.99c: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.8d9, %Iterate.facet.780 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.specific_fn.ffd: <specific function> = specific_function %R.as_type.as.Iterate.impl.Next.c8f, @R.as_type.as.Iterate.impl.Next(%facet_value.089) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.fe4: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.018) [concrete]
// CHECK:STDOUT:   %.57d: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.fe4, %Destroy.facet.018 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .ValueType = %ValueType.decl
// CHECK:STDOUT:     .MutableRange = %MutableRange.decl
// CHECK:STDOUT:     .ConstRange = %ConstRange.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.0b9: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.498) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.bf8)]
// CHECK:STDOUT:   %Core.import_ref.61f: @Optional.%Optional.Get.type (%Optional.Get.type.d22) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.6fc)]
// CHECK:STDOUT:   %Core.import_ref.2b5: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.f8b)]
// CHECK:STDOUT:   %Core.import_ref.b0a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.ceb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.9dd)]
// CHECK:STDOUT:   %Core.import_ref.3fc: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.87a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.898)]
// CHECK:STDOUT:   %Core.import_ref.986: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.0ae) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.e0f)]
// CHECK:STDOUT:   %Core.import_ref.557: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.46f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8c2)]
// CHECK:STDOUT:   %Core.import_ref.999 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.6f7 = impl_witness_table (%Core.import_ref.2b5, %Core.import_ref.b0a, %Core.import_ref.3fc, %Core.import_ref.986, %Core.import_ref.557, %Core.import_ref.999), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %ValueType.decl: type = class_decl @ValueType [concrete = constants.%ValueType] {} {}
// CHECK:STDOUT:   %ValueType.cpp_destructor.decl: %ValueType.cpp_destructor.type = fn_decl @ValueType.cpp_destructor [concrete = constants.%ValueType.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.89a: %facet_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.a30)]
// CHECK:STDOUT:   %Core.import_ref.9ea: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.2fe)]
// CHECK:STDOUT:   %Core.import_ref.a6c: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor.type (%R.as_type.as.Iterate.impl.NewCursor.type.9b6) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor (constants.%R.as_type.as.Iterate.impl.NewCursor.b29)]
// CHECK:STDOUT:   %Core.import_ref.a02: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next.type (%R.as_type.as.Iterate.impl.Next.type.5b5) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next (constants.%R.as_type.as.Iterate.impl.Next.cd2)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.9f0 = impl_witness_table (%Core.import_ref.89a, %Core.import_ref.9ea, %Core.import_ref.a6c, %Core.import_ref.a02), @R.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %ValueType.cpp_operator.decl: %ValueType.cpp_operator.type = fn_decl @ValueType.cpp_operator [concrete = constants.%ValueType.cpp_operator] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MutableRange.decl: type = class_decl @MutableRange [concrete = constants.%MutableRange] {} {}
// CHECK:STDOUT:   %ConstRange.decl: type = class_decl @ConstRange [concrete = constants.%ConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc40_10.2: %Iterate_where.type.630) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc40_10.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.695)]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %R.as_type.loc40_69.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet (constants.%Iterate.facet.8c4)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.210)]
// CHECK:STDOUT:   %.loc44_29.13: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet [symbolic = %.loc44_29.13 (constants.%.9b8)]
// CHECK:STDOUT:   %impl.elem2.loc44_29.2: @Test.%.loc44_29.13 (%.9b8) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_29.2 (constants.%impl.elem2.377)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.4: <specific function> = specific_impl_function %impl.elem2.loc44_29.2, @Iterate.WithSelf.NewCursor(%Iterate.facet) [symbolic = %specific_impl_fn.loc44_29.4 (constants.%specific_impl_fn.ed7)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.864)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.ad2)]
// CHECK:STDOUT:   %.loc44_29.14: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet [symbolic = %.loc44_29.14 (constants.%.a74)]
// CHECK:STDOUT:   %impl.elem3.loc44_29.2: @Test.%.loc44_29.14 (%.a74) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc44_29.2 (constants.%impl.elem3.5d8)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.5: <specific function> = specific_impl_function %impl.elem3.loc44_29.2, @Iterate.WithSelf.Next(%Iterate.facet) [symbolic = %specific_impl_fn.loc44_29.5 (constants.%specific_impl_fn.327)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.a36)]
// CHECK:STDOUT:   %.loc44_29.15: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc44_29.15 (constants.%.490)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.185)]
// CHECK:STDOUT:   %impl.elem0.loc44_29.2: @Test.%.loc44_29.15 (%.490) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc44_29.2 (constants.%impl.elem0.d3e)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.6: <specific function> = specific_impl_function %impl.elem0.loc44_29.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc44_29.6 (constants.%specific_impl_fn.e07)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc40_69.1 (%R.as_type.339)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.bef = value_binding_pattern i [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @Test.%R.as_type.loc40_69.1 (%R.as_type.339) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc44_29.1: @Test.%.loc44_29.13 (%.9b8) = impl_witness_access constants.%Iterate.lookup_impl_witness.695, element2 [symbolic = %impl.elem2.loc44_29.2 (constants.%impl.elem2.377)]
// CHECK:STDOUT:     %bound_method.loc44_29.1: <bound method> = bound_method %r.ref, %impl.elem2.loc44_29.1
// CHECK:STDOUT:     %specific_impl_fn.loc44_29.1: <specific function> = specific_impl_function %impl.elem2.loc44_29.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.8c4) [symbolic = %specific_impl_fn.loc44_29.4 (constants.%specific_impl_fn.ed7)]
// CHECK:STDOUT:     %bound_method.loc44_29.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_29.1
// CHECK:STDOUT:     %var: ref @Test.%as_type (%as_type.254) = var invalid
// CHECK:STDOUT:     %.loc44_28.1: @Test.%R.as_type.loc40_69.1 (%R.as_type.339) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @Test.%as_type (%as_type.254) to %var = call %bound_method.loc44_29.2(%.loc44_28.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc44: @Test.%ptr (%ptr.864) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc44_29.1: @Test.%.loc44_29.14 (%.a74) = impl_witness_access constants.%Iterate.lookup_impl_witness.695, element3 [symbolic = %impl.elem3.loc44_29.2 (constants.%impl.elem3.5d8)]
// CHECK:STDOUT:     %bound_method.loc44_29.3: <bound method> = bound_method %r.ref, %impl.elem3.loc44_29.1
// CHECK:STDOUT:     %.loc44_29.1: %Destroy.type = converted constants.%as_type.254, constants.%impl.elem1.bdd [symbolic = %impl.elem1 (constants.%impl.elem1.bdd)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_29.2: <specific function> = specific_impl_function %impl.elem3.loc44_29.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.8c4) [symbolic = %specific_impl_fn.loc44_29.5 (constants.%specific_impl_fn.327)]
// CHECK:STDOUT:     %bound_method.loc44_29.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_29.2
// CHECK:STDOUT:     %.loc44_29.2: ref %Optional.4fe = temporary_storage
// CHECK:STDOUT:     %.loc44_28.2: @Test.%R.as_type.loc40_69.1 (%R.as_type.339) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.4fe to %.loc44_29.2 = call %bound_method.loc44_29.4(%.loc44_28.2, %addr.loc44)
// CHECK:STDOUT:     %.loc44_29.3: ref %Optional.4fe = temporary %.loc44_29.2, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc44_29.4: %Optional.HasValue.type.fdd = specific_constant imports.%Core.import_ref.0b9, @Optional(constants.%OptionalStorage.facet.7c3) [concrete = constants.%Optional.HasValue.fe2]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.fdd = name_ref HasValue, %.loc44_29.4 [concrete = constants.%Optional.HasValue.fe2]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc44_29.3, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.7c3) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc44_29.5: <bound method> = bound_method %.loc44_29.3, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc44_29.5: %Optional.4fe = acquire_value %.loc44_29.3
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc44_29.5(%.loc44_29.5)
// CHECK:STDOUT:     %.loc44_29.6: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc44_29.7: bool = converted %Optional.HasValue.call, %.loc44_29.6
// CHECK:STDOUT:     if %.loc44_29.7 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc44_29.8: %Optional.Get.type.43b = specific_constant imports.%Core.import_ref.61f, @Optional(constants.%OptionalStorage.facet.7c3) [concrete = constants.%Optional.Get.4c3]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.43b = name_ref Get, %.loc44_29.8 [concrete = constants.%Optional.Get.4c3]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc44_29.3, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.7c3) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc44_29.6: <bound method> = bound_method %.loc44_29.3, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc44_29.9: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc44_29.10: %Optional.4fe = acquire_value %.loc44_29.3
// CHECK:STDOUT:     %Optional.Get.call: init %ValueType to %.loc44_29.9 = call %bound_method.loc44_29.6(%.loc44_29.10)
// CHECK:STDOUT:     %.loc44_14: type = splice_block %ValueType.ref.loc44 [concrete = constants.%ValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc44: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %ValueType.ref.loc44: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc44_29.11: ref %ValueType = temporary %.loc44_29.9, %Optional.Get.call
// CHECK:STDOUT:     %.loc44_29.12: %ValueType = acquire_value %.loc44_29.11
// CHECK:STDOUT:     %i: %ValueType = value_binding i, %.loc44_29.12
// CHECK:STDOUT:     %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:     %ValueType.cpp_operator.bound: <bound method> = bound_method %sum.ref, imports.%ValueType.cpp_operator.decl
// CHECK:STDOUT:     %.loc45_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:     %addr.loc45: %ptr.6db = addr_of %.loc45_12
// CHECK:STDOUT:     %.loc45_9.1: %ptr.e0a = as_compatible %addr.loc45
// CHECK:STDOUT:     %.loc45_9.2: %ptr.e0a = converted %addr.loc45, %.loc45_9.1
// CHECK:STDOUT:     %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc45_9.2)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %ValueType.Op.bound.loc44: <bound method> = bound_method %.loc44_29.11, constants.%ValueType.Op.cc70d5.1
// CHECK:STDOUT:     %Op.ref.loc44: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:     %ValueType.cpp_destructor.bound.loc44: <bound method> = bound_method %.loc44_29.11, %Op.ref.loc44
// CHECK:STDOUT:     %ValueType.cpp_destructor.call.loc44: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc44(%.loc44_29.11)
// CHECK:STDOUT:     %Destroy.Op.bound: <bound method> = bound_method %.loc44_29.3, constants.%Destroy.Op.2dd605.6
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc44_29.3)
// CHECK:STDOUT:     %impl.elem0.loc44_29.1: @Test.%.loc44_29.15 (%.490) = impl_witness_access constants.%Destroy.lookup_impl_witness.185, element0 [symbolic = %impl.elem0.loc44_29.2 (constants.%impl.elem0.d3e)]
// CHECK:STDOUT:     %bound_method.loc44_29.7: <bound method> = bound_method %var, %impl.elem0.loc44_29.1
// CHECK:STDOUT:     %specific_impl_fn.loc44_29.3: <specific function> = specific_impl_function %impl.elem0.loc44_29.1, @Destroy.WithSelf.Op(constants.%impl.elem1.bdd) [symbolic = %specific_impl_fn.loc44_29.6 (constants.%specific_impl_fn.e07)]
// CHECK:STDOUT:     %bound_method.loc44_29.8: <bound method> = bound_method %var, %specific_impl_fn.loc44_29.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc44_29.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_29.1(%self.param: ref %MaybeUnformed.5ea) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_29.2(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_29.3(%self.param: ref %struct_type.value.has_value.40c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_29.4(%self.param: ref %DefaultOptionalStorage.448) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_29.5(%self.param: ref %Optional.4fe) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R.2da) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%R.2da
// CHECK:STDOUT:   %R.as_type.loc40_69.1 => constants.%R.as_type.339
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.df0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.035) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%facet_value.035
// CHECK:STDOUT:   %R.as_type.loc40_69.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b76
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.5a3
// CHECK:STDOUT:   %Iterate.facet => constants.%Iterate.facet.269
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.635
// CHECK:STDOUT:   %.loc44_29.13 => constants.%.3fa
// CHECK:STDOUT:   %impl.elem2.loc44_29.2 => constants.%R.as_type.as.Iterate.impl.NewCursor.dcd
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.4 => constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn.55e
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.0cd
// CHECK:STDOUT:   %as_type => constants.%tuple.type.c00
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.dee
// CHECK:STDOUT:   %ptr => constants.%ptr.993
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.ba6
// CHECK:STDOUT:   %.loc44_29.14 => constants.%.dcc
// CHECK:STDOUT:   %impl.elem3.loc44_29.2 => constants.%R.as_type.as.Iterate.impl.Next.2f5
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.5 => constants.%R.as_type.as.Iterate.impl.Next.specific_fn.370
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.6fd
// CHECK:STDOUT:   %.loc44_29.15 => constants.%.ad7
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e20f1f.7
// CHECK:STDOUT:   %impl.elem0.loc44_29.2 => constants.%Destroy.Op.2dd605.7
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.6 => constants.%Destroy.Op.2dd605.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.2f3) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%facet_value.2f3
// CHECK:STDOUT:   %R.as_type.loc40_69.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.631
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.c99
// CHECK:STDOUT:   %Iterate.facet => constants.%Iterate.facet.780
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.b98
// CHECK:STDOUT:   %.loc44_29.13 => constants.%.25f
// CHECK:STDOUT:   %impl.elem2.loc44_29.2 => constants.%R.as_type.as.Iterate.impl.NewCursor.017
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.4 => constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn.26d
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.018
// CHECK:STDOUT:   %as_type => constants.%tuple.type.6b1
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.74d
// CHECK:STDOUT:   %ptr => constants.%ptr.f19
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.8d9
// CHECK:STDOUT:   %.loc44_29.14 => constants.%.99c
// CHECK:STDOUT:   %impl.elem3.loc44_29.2 => constants.%R.as_type.as.Iterate.impl.Next.c8f
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.5 => constants.%R.as_type.as.Iterate.impl.Next.specific_fn.ffd
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.fe4
// CHECK:STDOUT:   %.loc44_29.15 => constants.%.57d
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e20f1f.8
// CHECK:STDOUT:   %impl.elem0.loc44_29.2 => constants.%Destroy.Op.2dd605.8
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.6 => constants.%Destroy.Op.2dd605.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_return_same_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.4e7: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.d22: type = fn_type @Optional.Get, @Optional(%T.4e7) [symbolic]
// CHECK:STDOUT:   %Optional.Get.6fc: %Optional.Get.type.d22 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.498: type = fn_type @Optional.HasValue, @Optional(%T.4e7) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.bf8: %Optional.HasValue.type.498 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.cb0: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f8b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.46f: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8c2: %T.as_type.as.OptionalStorage.impl.Get.type.46f = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.0ae: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.e0f: %T.as_type.as.OptionalStorage.impl.Has.type.0ae = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.87a: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.898: %T.as_type.as.OptionalStorage.impl.Some.type.87a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.ceb: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.9dd: %T.as_type.as.OptionalStorage.impl.None.type.ceb = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.e31: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.732: <witness> = lookup_impl_witness %.Self.e31, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.03d: %facet_type = impl_witness_access %Iterate.lookup_impl_witness.732, element0 [symbolic_self]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f64.b55: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.f3c: type = pattern_type %f64.b55 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc20_47.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.2: <witness> = custom_witness (%Destroy.Op.2dd605.2), @Destroy [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.125: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bfc: %Float.as.Copy.impl.Op.type.125 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.811: <witness> = impl_witness imports.%Copy.impl_witness_table.384, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.b35: %facet_type = facet_value %f64.b55, (%custom_witness.e20f1f.2, %Copy.impl_witness.811) [concrete]
// CHECK:STDOUT:   %Iterate_where.type.fd5: type = facet_type <@Iterate where %impl.elem0.03d = %facet_value.b35> [concrete]
// CHECK:STDOUT:   %R.706: %Iterate_where.type.fd5 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type.8de: type = facet_access_type %R.706 [symbolic]
// CHECK:STDOUT:   %pattern_type.4d6: type = pattern_type %R.as_type.8de [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %.Self.10b: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.5bf: <witness> = lookup_impl_witness %.Self.10b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d47: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.bce: <witness> = lookup_impl_witness %impl.elem0.d47, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c87: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.bce, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.846: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.d47 impls @Destroy and %impl.elem0.d47 impls @Copy and %impl.elem0.d47 impls @CppUnsafeDeref and %impl.elem0.d47 impls @EqWith, @EqWith(%impl.elem1.846) and %impl.elem0.d47 impls @Inc and %impl.elem0.c87 impls @Destroy and %impl.elem0.c87 impls @Copy and %impl.elem1.846 impls @Destroy and %impl.elem1.846 impls @Copy> [concrete]
// CHECK:STDOUT:   %R.6c6: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.edd1fe.2: <witness> = lookup_impl_witness %R.6c6, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.bdcd2d.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.d5966e.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.384: type = tuple_type (%impl.elem0.d5966e.2, %impl.elem1.bdcd2d.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.0ba: <witness> = lookup_impl_witness %tuple.type.384, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.2fe: %Destroy.type = facet_value %tuple.type.384, (%Destroy.lookup_impl_witness.0ba) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.ca4: <witness> = lookup_impl_witness %impl.elem0.d5966e.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.b10: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.ca4, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.3d8: <witness> = lookup_impl_witness %impl.elem0.b10, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.bff: <witness> = lookup_impl_witness %impl.elem0.b10, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.a30: %facet_type = facet_value %impl.elem0.b10, (%Destroy.lookup_impl_witness.3d8, %Copy.lookup_impl_witness.bff) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.5b5: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%R.6c6) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.cd2: %R.as_type.as.Iterate.impl.Next.type.5b5 = struct_value () [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.9b6: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%R.6c6) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.b29: %R.as_type.as.Iterate.impl.NewCursor.type.9b6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.97c: <witness> = lookup_impl_witness %R.706, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.564: %Iterate.type = facet_value %R.as_type.8de, (%Iterate.lookup_impl_witness.97c) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.95b: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.564) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.31d: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.564) [symbolic]
// CHECK:STDOUT:   %.937: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.95b, %Iterate.facet.564 [symbolic]
// CHECK:STDOUT:   %impl.elem2.fdc: %.937 = impl_witness_access %Iterate.lookup_impl_witness.97c, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.46d: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.97c, element1 [symbolic]
// CHECK:STDOUT:   %as_type.e0d: type = facet_access_type %impl.elem1.46d [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a2c: <specific function> = specific_impl_function %impl.elem2.fdc, @Iterate.WithSelf.NewCursor(%Iterate.facet.564) [symbolic]
// CHECK:STDOUT:   %ptr.678: type = ptr_type %as_type.e0d [symbolic]
// CHECK:STDOUT:   %.b70: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.31d, %Iterate.facet.564 [symbolic]
// CHECK:STDOUT:   %impl.elem3.5cd: %.b70 = impl_witness_access %Iterate.lookup_impl_witness.97c, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d02: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.b35) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.4b5: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.6f7, @T.as_type.as.OptionalStorage.impl(%facet_value.b35) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.fd4: %OptionalStorage.type = facet_value %f64.b55, (%OptionalStorage.impl_witness.4b5) [concrete]
// CHECK:STDOUT:   %Optional.249: type = class_type @Optional, @Optional(%OptionalStorage.facet.fd4) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.e20: <specific function> = specific_impl_function %impl.elem3.5cd, @Iterate.WithSelf.Next(%Iterate.facet.564) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e04: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.fd4) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.69e: %Optional.HasValue.type.e04 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.941: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.fd4) [concrete]
// CHECK:STDOUT:   %Optional.Get.8ce: %Optional.Get.type.941 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.169: %Destroy.type = facet_value %f64.b55, (%custom_witness.e20f1f.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.bf7: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.169) [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.ec9: type = struct_type {.value: %MaybeUnformed.bf7, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.69e, @Optional.HasValue(%OptionalStorage.facet.fd4) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.8ce, @Optional.Get(%OptionalStorage.facet.fd4) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.f8a: type = facet_type <@AddAssignWith, @AddAssignWith(%f64.b55)> [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.4cd: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.ec5: %Float.as.AddAssignWith.impl.Op.type.4cd = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.4c6: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.db3: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.ba4: %Float.as.AddAssignWith.impl.Op.type.db3 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.f8a = facet_value %f64.b55, (%AddAssignWith.impl_witness.4c6) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.aab: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%f64.b55, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.8cc: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.aab, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Float.as.AddAssignWith.impl.Op.ba4, @Float.as.AddAssignWith.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.8: type = fn_type @Destroy.Op.loc24_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.8: %Destroy.Op.type.790b3b.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.56d: <witness> = lookup_impl_witness %impl.elem1.46d, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.19e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.46d) [symbolic]
// CHECK:STDOUT:   %.fb1: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.19e, %impl.elem1.46d [symbolic]
// CHECK:STDOUT:   %impl.elem0.648: %.fb1 = impl_witness_access %Destroy.lookup_impl_witness.56d, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.05c: <specific function> = specific_impl_function %impl.elem0.648, @Destroy.WithSelf.Op(%impl.elem1.46d) [symbolic]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.88d: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.4c3: <witness> = custom_witness (%Iterator, %Iterator, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.587642.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.dc644c.1: %Iterator.Op.type.587642.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e59: <witness> = custom_witness (%f64.b55, %Iterator.Op.dc644c.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.587642.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.dc644c.2: %Iterator.Op.type.587642.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.166: <witness> = custom_witness (%Iterator.Op.dc644c.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.587642.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.dc644c.3: %Iterator.Op.type.587642.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b36: <witness> = custom_witness (%Iterator.Op.dc644c.3), @Copy [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.832: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.587642.4: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.dc644c.4: %Iterator.Op.type.587642.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8ca: <witness> = custom_witness (%Iterator.Op.dc644c.4), @Inc [concrete]
// CHECK:STDOUT:   %facet_value.17b: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.e20f1f.2, %Copy.impl_witness.811, %custom_witness.4c3, %custom_witness.166, %custom_witness.b36, %custom_witness.e59, %custom_witness.832, %custom_witness.8ca) [concrete]
// CHECK:STDOUT:   %tuple.type.d7d: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.9: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.9: %Destroy.Op.type.790b3b.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.9: <witness> = custom_witness (%Destroy.Op.2dd605.9), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.57a: %Destroy.type = facet_value %tuple.type.d7d, (%custom_witness.e20f1f.9) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.d7d: <witness> = impl_witness imports.%Iterate.impl_witness_table.9f0, @R.as_type.as.Iterate.impl(%facet_value.17b) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.26a: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%facet_value.17b) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.51f: %R.as_type.as.Iterate.impl.NewCursor.type.26a = struct_value () [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.d6c: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%facet_value.17b) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.753: %R.as_type.as.Iterate.impl.Next.type.d6c = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.946: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.d7d) [concrete]
// CHECK:STDOUT:   %facet_value.7f4: %Iterate_where.type.fd5 = facet_value %ConstRange, (%Iterate.impl_witness.d7d) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.aba: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.946) [concrete]
// CHECK:STDOUT:   %.35d: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.aba, %Iterate.facet.946 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %R.as_type.as.Iterate.impl.NewCursor.51f, @R.as_type.as.Iterate.impl.NewCursor(%facet_value.17b) [concrete]
// CHECK:STDOUT:   %complete_type.a15: <witness> = complete_type_witness %tuple.type.d7d [concrete]
// CHECK:STDOUT:   %ptr.365: type = ptr_type %tuple.type.d7d [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.eee: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.946) [concrete]
// CHECK:STDOUT:   %.828: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.eee, %Iterate.facet.946 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %R.as_type.as.Iterate.impl.Next.753, @R.as_type.as.Iterate.impl.Next(%facet_value.17b) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.286: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.57a) [concrete]
// CHECK:STDOUT:   %.2ef: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.286, %Destroy.facet.57a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.0b9: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.498) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.bf8)]
// CHECK:STDOUT:   %Core.import_ref.61f: @Optional.%Optional.Get.type (%Optional.Get.type.d22) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.6fc)]
// CHECK:STDOUT:   %Core.import_ref.2b5: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.f8b)]
// CHECK:STDOUT:   %Core.import_ref.b0a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.ceb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.9dd)]
// CHECK:STDOUT:   %Core.import_ref.3fc: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.87a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.898)]
// CHECK:STDOUT:   %Core.import_ref.986: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.0ae) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.e0f)]
// CHECK:STDOUT:   %Core.import_ref.557: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.46f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8c2)]
// CHECK:STDOUT:   %Core.import_ref.999 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.6f7 = impl_witness_table (%Core.import_ref.2b5, %Core.import_ref.b0a, %Core.import_ref.3fc, %Core.import_ref.986, %Core.import_ref.557, %Core.import_ref.999), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.461: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.125) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.bfc)]
// CHECK:STDOUT:   %Copy.impl_witness_table.384 = impl_witness_table (%Core.import_ref.461), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.89a: %facet_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.a30)]
// CHECK:STDOUT:   %Core.import_ref.9ea: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.2fe)]
// CHECK:STDOUT:   %Core.import_ref.a6c: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor.type (%R.as_type.as.Iterate.impl.NewCursor.type.9b6) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor (constants.%R.as_type.as.Iterate.impl.NewCursor.b29)]
// CHECK:STDOUT:   %Core.import_ref.a02: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next.type (%R.as_type.as.Iterate.impl.Next.type.5b5) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next (constants.%R.as_type.as.Iterate.impl.Next.cd2)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.9f0 = impl_witness_table (%Core.import_ref.89a, %Core.import_ref.9ea, %Core.import_ref.a6c, %Core.import_ref.a02), @R.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c81: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.4cd) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.ec5)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.c81), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc20_10.2: %Iterate_where.type.fd5) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc20_10.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.97c)]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %R.as_type.loc20_55.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet (constants.%Iterate.facet.564)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.95b)]
// CHECK:STDOUT:   %.loc24_19.12: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet [symbolic = %.loc24_19.12 (constants.%.937)]
// CHECK:STDOUT:   %impl.elem2.loc24_19.2: @Test.%.loc24_19.12 (%.937) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc24_19.2 (constants.%impl.elem2.fdc)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4: <specific function> = specific_impl_function %impl.elem2.loc24_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.a2c)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.678)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.31d)]
// CHECK:STDOUT:   %.loc24_19.13: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet [symbolic = %.loc24_19.13 (constants.%.b70)]
// CHECK:STDOUT:   %impl.elem3.loc24_19.2: @Test.%.loc24_19.13 (%.b70) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.5cd)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.5: <specific function> = specific_impl_function %impl.elem3.loc24_19.2, @Iterate.WithSelf.Next(%Iterate.facet) [symbolic = %specific_impl_fn.loc24_19.5 (constants.%specific_impl_fn.e20)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.19e)]
// CHECK:STDOUT:   %.loc24_19.14: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc24_19.14 (constants.%.fb1)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.56d)]
// CHECK:STDOUT:   %impl.elem0.loc24_19.2: @Test.%.loc24_19.14 (%.fb1) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc24_19.2 (constants.%impl.elem0.648)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.6: <specific function> = specific_impl_function %impl.elem0.loc24_19.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc24_19.6 (constants.%specific_impl_fn.05c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @Test.%R.as_type.loc20_55.1 (%R.as_type.8de)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.f3c = value_binding_pattern i [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @Test.%R.as_type.loc20_55.1 (%R.as_type.8de) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc24_19.1: @Test.%.loc24_19.12 (%.937) = impl_witness_access constants.%Iterate.lookup_impl_witness.97c, element2 [symbolic = %impl.elem2.loc24_19.2 (constants.%impl.elem2.fdc)]
// CHECK:STDOUT:     %bound_method.loc24_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc24_19.1
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.1: <specific function> = specific_impl_function %impl.elem2.loc24_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.564) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.a2c)]
// CHECK:STDOUT:     %bound_method.loc24_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.1
// CHECK:STDOUT:     %var: ref @Test.%as_type (%as_type.e0d) = var invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @Test.%as_type (%as_type.e0d) to %var = call %bound_method.loc24_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @Test.%ptr (%ptr.678) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc24_19.1: @Test.%.loc24_19.13 (%.b70) = impl_witness_access constants.%Iterate.lookup_impl_witness.97c, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.5cd)]
// CHECK:STDOUT:     %bound_method.loc24_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc24_19.1
// CHECK:STDOUT:     %.loc24_19.1: %Destroy.type = converted constants.%as_type.e0d, constants.%impl.elem1.46d [symbolic = %impl.elem1 (constants.%impl.elem1.46d)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.2: <specific function> = specific_impl_function %impl.elem3.loc24_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.564) [symbolic = %specific_impl_fn.loc24_19.5 (constants.%specific_impl_fn.e20)]
// CHECK:STDOUT:     %bound_method.loc24_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.2
// CHECK:STDOUT:     %.loc24_19.2: ref %Optional.249 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.249 to %.loc24_19.2 = call %bound_method.loc24_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc24_19.3: ref %Optional.249 = temporary %.loc24_19.2, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc24_19.4: %Optional.HasValue.type.e04 = specific_constant imports.%Core.import_ref.0b9, @Optional(constants.%OptionalStorage.facet.fd4) [concrete = constants.%Optional.HasValue.69e]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.e04 = name_ref HasValue, %.loc24_19.4 [concrete = constants.%Optional.HasValue.69e]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc24_19.3, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.fd4) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc24_19.5: <bound method> = bound_method %.loc24_19.3, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc24_19.5: %Optional.249 = acquire_value %.loc24_19.3
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc24_19.5(%.loc24_19.5)
// CHECK:STDOUT:     %.loc24_19.6: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc24_19.7: bool = converted %Optional.HasValue.call, %.loc24_19.6
// CHECK:STDOUT:     if %.loc24_19.7 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc24_19.8: %Optional.Get.type.941 = specific_constant imports.%Core.import_ref.61f, @Optional(constants.%OptionalStorage.facet.fd4) [concrete = constants.%Optional.Get.8ce]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.941 = name_ref Get, %.loc24_19.8 [concrete = constants.%Optional.Get.8ce]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc24_19.3, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.fd4) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc24_19.6: <bound method> = bound_method %.loc24_19.3, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc24_19.9: %Optional.249 = acquire_value %.loc24_19.3
// CHECK:STDOUT:     %Optional.Get.call: init %f64.b55 = call %bound_method.loc24_19.6(%.loc24_19.9)
// CHECK:STDOUT:     %f64.loc24: type = type_literal constants.%f64.b55 [concrete = constants.%f64.b55]
// CHECK:STDOUT:     %.loc24_19.10: %f64.b55 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc24_19.11: %f64.b55 = converted %Optional.Get.call, %.loc24_19.10
// CHECK:STDOUT:     %i: %f64.b55 = value_binding i, %.loc24_19.11
// CHECK:STDOUT:     %sum.ref: ref %f64.b55 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %f64.b55 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc25: %.8cc = impl_witness_access constants.%AddAssignWith.impl_witness.4c6, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.ba4]
// CHECK:STDOUT:     %bound_method.loc25_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc25
// CHECK:STDOUT:     %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc25_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc25
// CHECK:STDOUT:     %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc25_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc24: <bound method> = bound_method %.loc24_19.3, constants.%Destroy.Op.2dd605.8
// CHECK:STDOUT:     %Destroy.Op.call.loc24: init %empty_tuple.type = call %Destroy.Op.bound.loc24(%.loc24_19.3)
// CHECK:STDOUT:     %impl.elem0.loc24_19.1: @Test.%.loc24_19.14 (%.fb1) = impl_witness_access constants.%Destroy.lookup_impl_witness.56d, element0 [symbolic = %impl.elem0.loc24_19.2 (constants.%impl.elem0.648)]
// CHECK:STDOUT:     %bound_method.loc24_19.7: <bound method> = bound_method %var, %impl.elem0.loc24_19.1
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.3: <specific function> = specific_impl_function %impl.elem0.loc24_19.1, @Destroy.WithSelf.Op(constants.%impl.elem1.46d) [symbolic = %specific_impl_fn.loc24_19.6 (constants.%specific_impl_fn.05c)]
// CHECK:STDOUT:     %bound_method.loc24_19.8: <bound method> = bound_method %var, %specific_impl_fn.loc24_19.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc24_19.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.1(%self.param: ref %MaybeUnformed.bf7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.2(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.3(%self.param: ref %struct_type.value.has_value.ec9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.4(%self.param: ref %DefaultOptionalStorage.d02) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.5(%self.param: ref %Optional.249) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R.706) {
// CHECK:STDOUT:   %R.loc20_10.1 => constants.%R.706
// CHECK:STDOUT:   %R.as_type.loc20_55.1 => constants.%R.as_type.8de
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4d6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.7f4) {
// CHECK:STDOUT:   %R.loc20_10.1 => constants.%facet_value.7f4
// CHECK:STDOUT:   %R.as_type.loc20_55.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.88d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.d7d
// CHECK:STDOUT:   %Iterate.facet => constants.%Iterate.facet.946
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.aba
// CHECK:STDOUT:   %.loc24_19.12 => constants.%.35d
// CHECK:STDOUT:   %impl.elem2.loc24_19.2 => constants.%R.as_type.as.Iterate.impl.NewCursor.51f
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4 => constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.57a
// CHECK:STDOUT:   %as_type => constants.%tuple.type.d7d
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.a15
// CHECK:STDOUT:   %ptr => constants.%ptr.365
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.eee
// CHECK:STDOUT:   %.loc24_19.13 => constants.%.828
// CHECK:STDOUT:   %impl.elem3.loc24_19.2 => constants.%R.as_type.as.Iterate.impl.Next.753
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.5 => constants.%R.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.286
// CHECK:STDOUT:   %.loc24_19.14 => constants.%.2ef
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e20f1f.9
// CHECK:STDOUT:   %impl.elem0.loc24_19.2 => constants.%Destroy.Op.2dd605.9
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.6 => constants.%Destroy.Op.2dd605.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_returns_same_types_mutable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.4e7: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.d22: type = fn_type @Optional.Get, @Optional(%T.4e7) [symbolic]
// CHECK:STDOUT:   %Optional.Get.6fc: %Optional.Get.type.d22 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.498: type = fn_type @Optional.HasValue, @Optional(%T.4e7) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.bf8: %Optional.HasValue.type.498 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.cb0: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f8b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.46f: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8c2: %T.as_type.as.OptionalStorage.impl.Get.type.46f = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.0ae: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.e0f: %T.as_type.as.OptionalStorage.impl.Has.type.0ae = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.87a: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.898: %T.as_type.as.OptionalStorage.impl.Some.type.87a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.ceb: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.9dd: %T.as_type.as.OptionalStorage.impl.None.type.ceb = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.e31: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.732: <witness> = lookup_impl_witness %.Self.e31, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.03d: %facet_type = impl_witness_access %Iterate.lookup_impl_witness.732, element0 [symbolic_self]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f64.b55: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.f3c: type = pattern_type %f64.b55 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc22_47.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.2: <witness> = custom_witness (%Destroy.Op.2dd605.2), @Destroy [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.125: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bfc: %Float.as.Copy.impl.Op.type.125 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.811: <witness> = impl_witness imports.%Copy.impl_witness_table.384, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.b35: %facet_type = facet_value %f64.b55, (%custom_witness.e20f1f.2, %Copy.impl_witness.811) [concrete]
// CHECK:STDOUT:   %Iterate_where.type.fd5: type = facet_type <@Iterate where %impl.elem0.03d = %facet_value.b35> [concrete]
// CHECK:STDOUT:   %R.706: %Iterate_where.type.fd5 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type.8de: type = facet_access_type %R.706 [symbolic]
// CHECK:STDOUT:   %pattern_type.4d6: type = pattern_type %R.as_type.8de [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %.Self.10b: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.5bf: <witness> = lookup_impl_witness %.Self.10b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d47: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.bce: <witness> = lookup_impl_witness %impl.elem0.d47, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c87: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.bce, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.846: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.d47 impls @Destroy and %impl.elem0.d47 impls @Copy and %impl.elem0.d47 impls @CppUnsafeDeref and %impl.elem0.d47 impls @EqWith, @EqWith(%impl.elem1.846) and %impl.elem0.d47 impls @Inc and %impl.elem0.c87 impls @Destroy and %impl.elem0.c87 impls @Copy and %impl.elem1.846 impls @Destroy and %impl.elem1.846 impls @Copy> [concrete]
// CHECK:STDOUT:   %R.6c6: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.edd1fe.2: <witness> = lookup_impl_witness %R.6c6, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.bdcd2d.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.d5966e.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.384: type = tuple_type (%impl.elem0.d5966e.2, %impl.elem1.bdcd2d.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.0ba: <witness> = lookup_impl_witness %tuple.type.384, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.2fe: %Destroy.type = facet_value %tuple.type.384, (%Destroy.lookup_impl_witness.0ba) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.ca4: <witness> = lookup_impl_witness %impl.elem0.d5966e.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.b10: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.ca4, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.3d8: <witness> = lookup_impl_witness %impl.elem0.b10, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.bff: <witness> = lookup_impl_witness %impl.elem0.b10, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.a30: %facet_type = facet_value %impl.elem0.b10, (%Destroy.lookup_impl_witness.3d8, %Copy.lookup_impl_witness.bff) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.5b5: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%R.6c6) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.cd2: %R.as_type.as.Iterate.impl.Next.type.5b5 = struct_value () [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.9b6: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%R.6c6) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.b29: %R.as_type.as.Iterate.impl.NewCursor.type.9b6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.97c: <witness> = lookup_impl_witness %R.706, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.564: %Iterate.type = facet_value %R.as_type.8de, (%Iterate.lookup_impl_witness.97c) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.95b: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.564) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.31d: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.564) [symbolic]
// CHECK:STDOUT:   %.937: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.95b, %Iterate.facet.564 [symbolic]
// CHECK:STDOUT:   %impl.elem2.fdc: %.937 = impl_witness_access %Iterate.lookup_impl_witness.97c, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.46d: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.97c, element1 [symbolic]
// CHECK:STDOUT:   %as_type.e0d: type = facet_access_type %impl.elem1.46d [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a2c: <specific function> = specific_impl_function %impl.elem2.fdc, @Iterate.WithSelf.NewCursor(%Iterate.facet.564) [symbolic]
// CHECK:STDOUT:   %ptr.678: type = ptr_type %as_type.e0d [symbolic]
// CHECK:STDOUT:   %.b70: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.31d, %Iterate.facet.564 [symbolic]
// CHECK:STDOUT:   %impl.elem3.5cd: %.b70 = impl_witness_access %Iterate.lookup_impl_witness.97c, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d02: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.b35) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.4b5: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.6f7, @T.as_type.as.OptionalStorage.impl(%facet_value.b35) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.fd4: %OptionalStorage.type = facet_value %f64.b55, (%OptionalStorage.impl_witness.4b5) [concrete]
// CHECK:STDOUT:   %Optional.249: type = class_type @Optional, @Optional(%OptionalStorage.facet.fd4) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.e20: <specific function> = specific_impl_function %impl.elem3.5cd, @Iterate.WithSelf.Next(%Iterate.facet.564) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e04: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.fd4) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.69e: %Optional.HasValue.type.e04 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.941: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.fd4) [concrete]
// CHECK:STDOUT:   %Optional.Get.8ce: %Optional.Get.type.941 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.169: %Destroy.type = facet_value %f64.b55, (%custom_witness.e20f1f.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.bf7: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.169) [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.ec9: type = struct_type {.value: %MaybeUnformed.bf7, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.69e, @Optional.HasValue(%OptionalStorage.facet.fd4) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.8ce, @Optional.Get(%OptionalStorage.facet.fd4) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.f8a: type = facet_type <@AddAssignWith, @AddAssignWith(%f64.b55)> [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.4cd: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.ec5: %Float.as.AddAssignWith.impl.Op.type.4cd = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.4c6: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.db3: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.ba4: %Float.as.AddAssignWith.impl.Op.type.db3 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.f8a = facet_value %f64.b55, (%AddAssignWith.impl_witness.4c6) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.aab: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%f64.b55, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.8cc: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.aab, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Float.as.AddAssignWith.impl.Op.ba4, @Float.as.AddAssignWith.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.8: type = fn_type @Destroy.Op.loc26_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.8: %Destroy.Op.type.790b3b.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.56d: <witness> = lookup_impl_witness %impl.elem1.46d, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.19e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.46d) [symbolic]
// CHECK:STDOUT:   %.fb1: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.19e, %impl.elem1.46d [symbolic]
// CHECK:STDOUT:   %impl.elem0.648: %.fb1 = impl_witness_access %Destroy.lookup_impl_witness.56d, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.05c: <specific function> = specific_impl_function %impl.elem0.648, @Destroy.WithSelf.Op(%impl.elem1.46d) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.80f: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.25d: <witness> = custom_witness (%Iterator, %Iterator, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.910736.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.cbb479.1: %Iterator.Op.type.910736.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5e7: <witness> = custom_witness (%f64.b55, %Iterator.Op.cbb479.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.910736.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.cbb479.2: %Iterator.Op.type.910736.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe6: <witness> = custom_witness (%Iterator.Op.cbb479.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.910736.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.cbb479.3: %Iterator.Op.type.910736.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b9a: <witness> = custom_witness (%Iterator.Op.cbb479.3), @Copy [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ab9: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.910736.4: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.cbb479.4: %Iterator.Op.type.910736.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.88b: <witness> = custom_witness (%Iterator.Op.cbb479.4), @Inc [concrete]
// CHECK:STDOUT:   %facet_value.fbd: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.e20f1f.2, %Copy.impl_witness.811, %custom_witness.25d, %custom_witness.fe6, %custom_witness.b9a, %custom_witness.5e7, %custom_witness.ab9, %custom_witness.88b) [concrete]
// CHECK:STDOUT:   %tuple.type.e5c: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.9: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.9: %Destroy.Op.type.790b3b.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.9: <witness> = custom_witness (%Destroy.Op.2dd605.9), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.a02: %Destroy.type = facet_value %tuple.type.e5c, (%custom_witness.e20f1f.9) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.b58: <witness> = impl_witness imports.%Iterate.impl_witness_table.9f0, @R.as_type.as.Iterate.impl(%facet_value.fbd) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.efa: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%facet_value.fbd) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.74e: %R.as_type.as.Iterate.impl.NewCursor.type.efa = struct_value () [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.ddb: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%facet_value.fbd) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.adb: %R.as_type.as.Iterate.impl.Next.type.ddb = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.7ea: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.b58) [concrete]
// CHECK:STDOUT:   %facet_value.459: %Iterate_where.type.fd5 = facet_value %MutableRange, (%Iterate.impl_witness.b58) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.7f2: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.7ea) [concrete]
// CHECK:STDOUT:   %.b77: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.7f2, %Iterate.facet.7ea [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %R.as_type.as.Iterate.impl.NewCursor.74e, @R.as_type.as.Iterate.impl.NewCursor(%facet_value.fbd) [concrete]
// CHECK:STDOUT:   %complete_type.0e3: <witness> = complete_type_witness %tuple.type.e5c [concrete]
// CHECK:STDOUT:   %ptr.505: type = ptr_type %tuple.type.e5c [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.8ae: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.7ea) [concrete]
// CHECK:STDOUT:   %.8c6: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.8ae, %Iterate.facet.7ea [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %R.as_type.as.Iterate.impl.Next.adb, @R.as_type.as.Iterate.impl.Next(%facet_value.fbd) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.fe8: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.a02) [concrete]
// CHECK:STDOUT:   %.9af: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.fe8, %Destroy.facet.a02 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.0b9: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.498) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.bf8)]
// CHECK:STDOUT:   %Core.import_ref.61f: @Optional.%Optional.Get.type (%Optional.Get.type.d22) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.6fc)]
// CHECK:STDOUT:   %Core.import_ref.2b5: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.f8b)]
// CHECK:STDOUT:   %Core.import_ref.b0a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.ceb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.9dd)]
// CHECK:STDOUT:   %Core.import_ref.3fc: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.87a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.898)]
// CHECK:STDOUT:   %Core.import_ref.986: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.0ae) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.e0f)]
// CHECK:STDOUT:   %Core.import_ref.557: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.46f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8c2)]
// CHECK:STDOUT:   %Core.import_ref.999 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.6f7 = impl_witness_table (%Core.import_ref.2b5, %Core.import_ref.b0a, %Core.import_ref.3fc, %Core.import_ref.986, %Core.import_ref.557, %Core.import_ref.999), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.461: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.125) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.bfc)]
// CHECK:STDOUT:   %Copy.impl_witness_table.384 = impl_witness_table (%Core.import_ref.461), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.89a: %facet_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.a30)]
// CHECK:STDOUT:   %Core.import_ref.9ea: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.2fe)]
// CHECK:STDOUT:   %Core.import_ref.a6c: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor.type (%R.as_type.as.Iterate.impl.NewCursor.type.9b6) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor (constants.%R.as_type.as.Iterate.impl.NewCursor.b29)]
// CHECK:STDOUT:   %Core.import_ref.a02: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next.type (%R.as_type.as.Iterate.impl.Next.type.5b5) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next (constants.%R.as_type.as.Iterate.impl.Next.cd2)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.9f0 = impl_witness_table (%Core.import_ref.89a, %Core.import_ref.9ea, %Core.import_ref.a6c, %Core.import_ref.a02), @R.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c81: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.4cd) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.ec5)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.c81), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc22_10.2: %Iterate_where.type.fd5) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc22_10.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.97c)]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %R.as_type.loc22_55.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet (constants.%Iterate.facet.564)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.95b)]
// CHECK:STDOUT:   %.loc26_19.12: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet [symbolic = %.loc26_19.12 (constants.%.937)]
// CHECK:STDOUT:   %impl.elem2.loc26_19.2: @Test.%.loc26_19.12 (%.937) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc26_19.2 (constants.%impl.elem2.fdc)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.4: <specific function> = specific_impl_function %impl.elem2.loc26_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet) [symbolic = %specific_impl_fn.loc26_19.4 (constants.%specific_impl_fn.a2c)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.678)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.31d)]
// CHECK:STDOUT:   %.loc26_19.13: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet [symbolic = %.loc26_19.13 (constants.%.b70)]
// CHECK:STDOUT:   %impl.elem3.loc26_19.2: @Test.%.loc26_19.13 (%.b70) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc26_19.2 (constants.%impl.elem3.5cd)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.5: <specific function> = specific_impl_function %impl.elem3.loc26_19.2, @Iterate.WithSelf.Next(%Iterate.facet) [symbolic = %specific_impl_fn.loc26_19.5 (constants.%specific_impl_fn.e20)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.19e)]
// CHECK:STDOUT:   %.loc26_19.14: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc26_19.14 (constants.%.fb1)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.56d)]
// CHECK:STDOUT:   %impl.elem0.loc26_19.2: @Test.%.loc26_19.14 (%.fb1) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_19.2 (constants.%impl.elem0.648)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.6: <specific function> = specific_impl_function %impl.elem0.loc26_19.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc26_19.6 (constants.%specific_impl_fn.05c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @Test.%R.as_type.loc22_55.1 (%R.as_type.8de)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.f3c = value_binding_pattern i [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @Test.%R.as_type.loc22_55.1 (%R.as_type.8de) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc26_19.1: @Test.%.loc26_19.12 (%.937) = impl_witness_access constants.%Iterate.lookup_impl_witness.97c, element2 [symbolic = %impl.elem2.loc26_19.2 (constants.%impl.elem2.fdc)]
// CHECK:STDOUT:     %bound_method.loc26_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc26_19.1
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.1: <specific function> = specific_impl_function %impl.elem2.loc26_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.564) [symbolic = %specific_impl_fn.loc26_19.4 (constants.%specific_impl_fn.a2c)]
// CHECK:STDOUT:     %bound_method.loc26_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc26_19.1
// CHECK:STDOUT:     %var: ref @Test.%as_type (%as_type.e0d) = var invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @Test.%as_type (%as_type.e0d) to %var = call %bound_method.loc26_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @Test.%ptr (%ptr.678) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc26_19.1: @Test.%.loc26_19.13 (%.b70) = impl_witness_access constants.%Iterate.lookup_impl_witness.97c, element3 [symbolic = %impl.elem3.loc26_19.2 (constants.%impl.elem3.5cd)]
// CHECK:STDOUT:     %bound_method.loc26_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc26_19.1
// CHECK:STDOUT:     %.loc26_19.1: %Destroy.type = converted constants.%as_type.e0d, constants.%impl.elem1.46d [symbolic = %impl.elem1 (constants.%impl.elem1.46d)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.2: <specific function> = specific_impl_function %impl.elem3.loc26_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.564) [symbolic = %specific_impl_fn.loc26_19.5 (constants.%specific_impl_fn.e20)]
// CHECK:STDOUT:     %bound_method.loc26_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc26_19.2
// CHECK:STDOUT:     %.loc26_19.2: ref %Optional.249 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.249 to %.loc26_19.2 = call %bound_method.loc26_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc26_19.3: ref %Optional.249 = temporary %.loc26_19.2, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc26_19.4: %Optional.HasValue.type.e04 = specific_constant imports.%Core.import_ref.0b9, @Optional(constants.%OptionalStorage.facet.fd4) [concrete = constants.%Optional.HasValue.69e]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.e04 = name_ref HasValue, %.loc26_19.4 [concrete = constants.%Optional.HasValue.69e]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc26_19.3, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.fd4) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc26_19.5: <bound method> = bound_method %.loc26_19.3, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc26_19.5: %Optional.249 = acquire_value %.loc26_19.3
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc26_19.5(%.loc26_19.5)
// CHECK:STDOUT:     %.loc26_19.6: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc26_19.7: bool = converted %Optional.HasValue.call, %.loc26_19.6
// CHECK:STDOUT:     if %.loc26_19.7 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc26_19.8: %Optional.Get.type.941 = specific_constant imports.%Core.import_ref.61f, @Optional(constants.%OptionalStorage.facet.fd4) [concrete = constants.%Optional.Get.8ce]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.941 = name_ref Get, %.loc26_19.8 [concrete = constants.%Optional.Get.8ce]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc26_19.3, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.fd4) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc26_19.6: <bound method> = bound_method %.loc26_19.3, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc26_19.9: %Optional.249 = acquire_value %.loc26_19.3
// CHECK:STDOUT:     %Optional.Get.call: init %f64.b55 = call %bound_method.loc26_19.6(%.loc26_19.9)
// CHECK:STDOUT:     %f64.loc26: type = type_literal constants.%f64.b55 [concrete = constants.%f64.b55]
// CHECK:STDOUT:     %.loc26_19.10: %f64.b55 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc26_19.11: %f64.b55 = converted %Optional.Get.call, %.loc26_19.10
// CHECK:STDOUT:     %i: %f64.b55 = value_binding i, %.loc26_19.11
// CHECK:STDOUT:     %sum.ref: ref %f64.b55 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %f64.b55 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc27: %.8cc = impl_witness_access constants.%AddAssignWith.impl_witness.4c6, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.ba4]
// CHECK:STDOUT:     %bound_method.loc27_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc27
// CHECK:STDOUT:     %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc27_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc27
// CHECK:STDOUT:     %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc27_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc26: <bound method> = bound_method %.loc26_19.3, constants.%Destroy.Op.2dd605.8
// CHECK:STDOUT:     %Destroy.Op.call.loc26: init %empty_tuple.type = call %Destroy.Op.bound.loc26(%.loc26_19.3)
// CHECK:STDOUT:     %impl.elem0.loc26_19.1: @Test.%.loc26_19.14 (%.fb1) = impl_witness_access constants.%Destroy.lookup_impl_witness.56d, element0 [symbolic = %impl.elem0.loc26_19.2 (constants.%impl.elem0.648)]
// CHECK:STDOUT:     %bound_method.loc26_19.7: <bound method> = bound_method %var, %impl.elem0.loc26_19.1
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.3: <specific function> = specific_impl_function %impl.elem0.loc26_19.1, @Destroy.WithSelf.Op(constants.%impl.elem1.46d) [symbolic = %specific_impl_fn.loc26_19.6 (constants.%specific_impl_fn.05c)]
// CHECK:STDOUT:     %bound_method.loc26_19.8: <bound method> = bound_method %var, %specific_impl_fn.loc26_19.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc26_19.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.1(%self.param: ref %MaybeUnformed.bf7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.2(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.3(%self.param: ref %struct_type.value.has_value.ec9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.4(%self.param: ref %DefaultOptionalStorage.d02) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.5(%self.param: ref %Optional.249) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R.706) {
// CHECK:STDOUT:   %R.loc22_10.1 => constants.%R.706
// CHECK:STDOUT:   %R.as_type.loc22_55.1 => constants.%R.as_type.8de
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4d6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.459) {
// CHECK:STDOUT:   %R.loc22_10.1 => constants.%facet_value.459
// CHECK:STDOUT:   %R.as_type.loc22_55.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.80f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc22 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.b58
// CHECK:STDOUT:   %Iterate.facet => constants.%Iterate.facet.7ea
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.7f2
// CHECK:STDOUT:   %.loc26_19.12 => constants.%.b77
// CHECK:STDOUT:   %impl.elem2.loc26_19.2 => constants.%R.as_type.as.Iterate.impl.NewCursor.74e
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.4 => constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.a02
// CHECK:STDOUT:   %as_type => constants.%tuple.type.e5c
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.0e3
// CHECK:STDOUT:   %ptr => constants.%ptr.505
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.8ae
// CHECK:STDOUT:   %.loc26_19.13 => constants.%.8c6
// CHECK:STDOUT:   %impl.elem3.loc26_19.2 => constants.%R.as_type.as.Iterate.impl.Next.adb
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.5 => constants.%R.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.fe8
// CHECK:STDOUT:   %.loc26_19.14 => constants.%.9af
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e20f1f.9
// CHECK:STDOUT:   %impl.elem0.loc26_19.2 => constants.%Destroy.Op.2dd605.9
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.6 => constants.%Destroy.Op.2dd605.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_returns_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.4e7: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.d22: type = fn_type @Optional.Get, @Optional(%T.4e7) [symbolic]
// CHECK:STDOUT:   %Optional.Get.6fc: %Optional.Get.type.d22 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.498: type = fn_type @Optional.HasValue, @Optional(%T.4e7) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.bf8: %Optional.HasValue.type.498 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.cb0: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f8b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.46f: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8c2: %T.as_type.as.OptionalStorage.impl.Get.type.46f = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.0ae: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.e0f: %T.as_type.as.OptionalStorage.impl.Has.type.0ae = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.87a: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.898: %T.as_type.as.OptionalStorage.impl.Some.type.87a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.ceb: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.9dd: %T.as_type.as.OptionalStorage.impl.None.type.ceb = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.e31: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.732: <witness> = lookup_impl_witness %.Self.e31, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.03d: %facet_type = impl_witness_access %Iterate.lookup_impl_witness.732, element0 [symbolic_self]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.94b: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor.type: type = fn_type @ValueType.cpp_destructor [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor: %ValueType.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.bd116f.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.1edc3c.1: %ValueType.Op.type.bd116f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.031: <witness> = custom_witness (%ValueType.Op.1edc3c.1), @Destroy [concrete]
// CHECK:STDOUT:   %const.e15: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.140: type = ptr_type %const.e15 [concrete]
// CHECK:STDOUT:   %ptr.fed: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.bd116f.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.1edc3c.2: %ValueType.Op.type.bd116f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d86: <witness> = custom_witness (%ValueType.Op.1edc3c.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.95f: %facet_type = facet_value %ValueType, (%custom_witness.031, %custom_witness.d86) [concrete]
// CHECK:STDOUT:   %Iterate_where.type.6d9: type = facet_type <@Iterate where %impl.elem0.03d = %facet_value.95f> [concrete]
// CHECK:STDOUT:   %R.bd8: %Iterate_where.type.6d9 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type.e1d: type = facet_access_type %R.bd8 [symbolic]
// CHECK:STDOUT:   %pattern_type.b93: type = pattern_type %R.as_type.e1d [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %.Self.10b: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.5bf: <witness> = lookup_impl_witness %.Self.10b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d47: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.bce: <witness> = lookup_impl_witness %impl.elem0.d47, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c87: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.bce, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.846: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.d47 impls @Destroy and %impl.elem0.d47 impls @Copy and %impl.elem0.d47 impls @CppUnsafeDeref and %impl.elem0.d47 impls @EqWith, @EqWith(%impl.elem1.846) and %impl.elem0.d47 impls @Inc and %impl.elem0.c87 impls @Destroy and %impl.elem0.c87 impls @Copy and %impl.elem1.846 impls @Destroy and %impl.elem1.846 impls @Copy> [concrete]
// CHECK:STDOUT:   %R.6c6: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.edd1fe.2: <witness> = lookup_impl_witness %R.6c6, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.bdcd2d.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.d5966e.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.384: type = tuple_type (%impl.elem0.d5966e.2, %impl.elem1.bdcd2d.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.0ba: <witness> = lookup_impl_witness %tuple.type.384, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.2fe: %Destroy.type = facet_value %tuple.type.384, (%Destroy.lookup_impl_witness.0ba) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.ca4: <witness> = lookup_impl_witness %impl.elem0.d5966e.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.b10: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.ca4, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.3d8: <witness> = lookup_impl_witness %impl.elem0.b10, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.bff: <witness> = lookup_impl_witness %impl.elem0.b10, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.a30: %facet_type = facet_value %impl.elem0.b10, (%Destroy.lookup_impl_witness.3d8, %Copy.lookup_impl_witness.bff) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.5b5: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%R.6c6) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.cd2: %R.as_type.as.Iterate.impl.Next.type.5b5 = struct_value () [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.9b6: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%R.6c6) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.b29: %R.as_type.as.Iterate.impl.NewCursor.type.9b6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.135: <witness> = lookup_impl_witness %R.bd8, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.dca: %Iterate.type = facet_value %R.as_type.e1d, (%Iterate.lookup_impl_witness.135) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.842: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.dca) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.a4d: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.dca) [symbolic]
// CHECK:STDOUT:   %.89c: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.842, %Iterate.facet.dca [symbolic]
// CHECK:STDOUT:   %impl.elem2.488: %.89c = impl_witness_access %Iterate.lookup_impl_witness.135, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.b38: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.135, element1 [symbolic]
// CHECK:STDOUT:   %as_type.470: type = facet_access_type %impl.elem1.b38 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5a0: <specific function> = specific_impl_function %impl.elem2.488, @Iterate.WithSelf.NewCursor(%Iterate.facet.dca) [symbolic]
// CHECK:STDOUT:   %ptr.ce1: type = ptr_type %as_type.470 [symbolic]
// CHECK:STDOUT:   %.503: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.a4d, %Iterate.facet.dca [symbolic]
// CHECK:STDOUT:   %impl.elem3.3bc: %.503 = impl_witness_access %Iterate.lookup_impl_witness.135, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.046: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.95f) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.1ee: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.6f7, @T.as_type.as.OptionalStorage.impl(%facet_value.95f) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.7ac: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.1ee) [concrete]
// CHECK:STDOUT:   %Optional.fb3: type = class_type @Optional, @Optional(%OptionalStorage.facet.7ac) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.7c6: <specific function> = specific_impl_function %impl.elem3.3bc, @Iterate.WithSelf.Next(%Iterate.facet.dca) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.475: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.7ac) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.c00: %Optional.HasValue.type.475 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.dd5: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.7ac) [concrete]
// CHECK:STDOUT:   %Optional.Get.335: %Optional.Get.type.dd5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.1a4: %Destroy.type = facet_value %ValueType, (%custom_witness.031) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.d14: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.1a4) [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.330: type = struct_type {.value: %MaybeUnformed.d14, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.c00, @Optional.HasValue(%OptionalStorage.facet.7ac) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.335, @Optional.Get(%OptionalStorage.facet.7ac) [concrete]
// CHECK:STDOUT:   %ValueType.cpp_operator.type: type = fn_type @ValueType.cpp_operator [concrete]
// CHECK:STDOUT:   %ValueType.cpp_operator: %ValueType.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.6: type = fn_type @Destroy.Op.loc31_31.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.6: %Destroy.Op.type.790b3b.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.55f: <witness> = lookup_impl_witness %impl.elem1.b38, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.567: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.b38) [symbolic]
// CHECK:STDOUT:   %.c3e: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.567, %impl.elem1.b38 [symbolic]
// CHECK:STDOUT:   %impl.elem0.38e: %.c3e = impl_witness_access %Destroy.lookup_impl_witness.55f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.3a8: <specific function> = specific_impl_function %impl.elem0.38e, @Destroy.WithSelf.Op(%impl.elem1.b38) [symbolic]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.88d: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Sentinel: type = class_type @Sentinel.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2ff: <witness> = custom_witness (%Iterator, %Sentinel, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.587642.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.dc644c.1: %Iterator.Op.type.587642.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.23f: <witness> = custom_witness (%ValueType, %Iterator.Op.dc644c.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.587642.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.dc644c.2: %Iterator.Op.type.587642.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.166783.1: <witness> = custom_witness (%Iterator.Op.dc644c.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.587642.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.dc644c.3: %Iterator.Op.type.587642.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b36db2.1: <witness> = custom_witness (%Iterator.Op.dc644c.3), @Copy [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2ed: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Sentinel) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.587642.4: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.dc644c.4: %Iterator.Op.type.587642.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8ca: <witness> = custom_witness (%Iterator.Op.dc644c.4), @Inc [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.587642.1: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.dc644c.1: %Sentinel.Op.type.587642.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.166783.2: <witness> = custom_witness (%Sentinel.Op.dc644c.1), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.587642.2: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.dc644c.2: %Sentinel.Op.type.587642.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b36db2.2: <witness> = custom_witness (%Sentinel.Op.dc644c.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.166: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.031, %custom_witness.d86, %custom_witness.2ff, %custom_witness.166783.1, %custom_witness.b36db2.1, %custom_witness.23f, %custom_witness.2ed, %custom_witness.8ca, %custom_witness.166783.2, %custom_witness.b36db2.2) [concrete]
// CHECK:STDOUT:   %tuple.type.a15: type = tuple_type (%Iterator, %Sentinel) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.7: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.7: %Destroy.Op.type.790b3b.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.7: <witness> = custom_witness (%Destroy.Op.2dd605.7), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.c5c: %Destroy.type = facet_value %tuple.type.a15, (%custom_witness.e20f1f.7) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.5a2: <witness> = impl_witness imports.%Iterate.impl_witness_table.9f0, @R.as_type.as.Iterate.impl(%facet_value.166) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.49f: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%facet_value.166) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.1c2: %R.as_type.as.Iterate.impl.NewCursor.type.49f = struct_value () [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.fc8: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%facet_value.166) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.7bf: %R.as_type.as.Iterate.impl.Next.type.fc8 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.bdd: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.5a2) [concrete]
// CHECK:STDOUT:   %facet_value.5f8: %Iterate_where.type.6d9 = facet_value %ConstRange, (%Iterate.impl_witness.5a2) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.e45: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.bdd) [concrete]
// CHECK:STDOUT:   %.cc1: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.e45, %Iterate.facet.bdd [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %R.as_type.as.Iterate.impl.NewCursor.1c2, @R.as_type.as.Iterate.impl.NewCursor(%facet_value.166) [concrete]
// CHECK:STDOUT:   %complete_type.443: <witness> = complete_type_witness %tuple.type.a15 [concrete]
// CHECK:STDOUT:   %ptr.261: type = ptr_type %tuple.type.a15 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.3db: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.bdd) [concrete]
// CHECK:STDOUT:   %.c8f: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.3db, %Iterate.facet.bdd [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %R.as_type.as.Iterate.impl.Next.7bf, @R.as_type.as.Iterate.impl.Next(%facet_value.166) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.6c1: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.c5c) [concrete]
// CHECK:STDOUT:   %.3f3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.6c1, %Destroy.facet.c5c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.0b9: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.498) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.bf8)]
// CHECK:STDOUT:   %Core.import_ref.61f: @Optional.%Optional.Get.type (%Optional.Get.type.d22) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.6fc)]
// CHECK:STDOUT:   %Core.import_ref.2b5: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.f8b)]
// CHECK:STDOUT:   %Core.import_ref.b0a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.ceb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.9dd)]
// CHECK:STDOUT:   %Core.import_ref.3fc: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.87a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.898)]
// CHECK:STDOUT:   %Core.import_ref.986: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.0ae) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.e0f)]
// CHECK:STDOUT:   %Core.import_ref.557: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.46f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8c2)]
// CHECK:STDOUT:   %Core.import_ref.999 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.6f7 = impl_witness_table (%Core.import_ref.2b5, %Core.import_ref.b0a, %Core.import_ref.3fc, %Core.import_ref.986, %Core.import_ref.557, %Core.import_ref.999), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ValueType = %ValueType.decl
// CHECK:STDOUT:     .ConstRange = %ConstRange.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ValueType.decl: type = class_decl @ValueType [concrete = constants.%ValueType] {} {}
// CHECK:STDOUT:   %ValueType.cpp_destructor.decl: %ValueType.cpp_destructor.type = fn_decl @ValueType.cpp_destructor [concrete = constants.%ValueType.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.89a: %facet_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.a30)]
// CHECK:STDOUT:   %Core.import_ref.9ea: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.2fe)]
// CHECK:STDOUT:   %Core.import_ref.a6c: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor.type (%R.as_type.as.Iterate.impl.NewCursor.type.9b6) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor (constants.%R.as_type.as.Iterate.impl.NewCursor.b29)]
// CHECK:STDOUT:   %Core.import_ref.a02: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next.type (%R.as_type.as.Iterate.impl.Next.type.5b5) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next (constants.%R.as_type.as.Iterate.impl.Next.cd2)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.9f0 = impl_witness_table (%Core.import_ref.89a, %Core.import_ref.9ea, %Core.import_ref.a6c, %Core.import_ref.a02), @R.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %ValueType.cpp_operator.decl: %ValueType.cpp_operator.type = fn_decl @ValueType.cpp_operator [concrete = constants.%ValueType.cpp_operator] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstRange.decl: type = class_decl @ConstRange [concrete = constants.%ConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc27_10.2: %Iterate_where.type.6d9) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_10.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.135)]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %R.as_type.loc27_67.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet (constants.%Iterate.facet.dca)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.842)]
// CHECK:STDOUT:   %.loc31_31.13: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet [symbolic = %.loc31_31.13 (constants.%.89c)]
// CHECK:STDOUT:   %impl.elem2.loc31_31.2: @Test.%.loc31_31.13 (%.89c) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_31.2 (constants.%impl.elem2.488)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.4: <specific function> = specific_impl_function %impl.elem2.loc31_31.2, @Iterate.WithSelf.NewCursor(%Iterate.facet) [symbolic = %specific_impl_fn.loc31_31.4 (constants.%specific_impl_fn.5a0)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.ce1)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.a4d)]
// CHECK:STDOUT:   %.loc31_31.14: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet [symbolic = %.loc31_31.14 (constants.%.503)]
// CHECK:STDOUT:   %impl.elem3.loc31_31.2: @Test.%.loc31_31.14 (%.503) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_31.2 (constants.%impl.elem3.3bc)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.5: <specific function> = specific_impl_function %impl.elem3.loc31_31.2, @Iterate.WithSelf.Next(%Iterate.facet) [symbolic = %specific_impl_fn.loc31_31.5 (constants.%specific_impl_fn.7c6)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.567)]
// CHECK:STDOUT:   %.loc31_31.15: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc31_31.15 (constants.%.c3e)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.55f)]
// CHECK:STDOUT:   %impl.elem0.loc31_31.2: @Test.%.loc31_31.15 (%.c3e) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc31_31.2 (constants.%impl.elem0.38e)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.6: <specific function> = specific_impl_function %impl.elem0.loc31_31.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc31_31.6 (constants.%specific_impl_fn.3a8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @Test.%R.as_type.loc27_67.1 (%R.as_type.e1d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.94b = value_binding_pattern i [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @Test.%R.as_type.loc27_67.1 (%R.as_type.e1d) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_31.1: @Test.%.loc31_31.13 (%.89c) = impl_witness_access constants.%Iterate.lookup_impl_witness.135, element2 [symbolic = %impl.elem2.loc31_31.2 (constants.%impl.elem2.488)]
// CHECK:STDOUT:     %bound_method.loc31_31.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_31.1
// CHECK:STDOUT:     %specific_impl_fn.loc31_31.1: <specific function> = specific_impl_function %impl.elem2.loc31_31.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.dca) [symbolic = %specific_impl_fn.loc31_31.4 (constants.%specific_impl_fn.5a0)]
// CHECK:STDOUT:     %bound_method.loc31_31.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_31.1
// CHECK:STDOUT:     %var: ref @Test.%as_type (%as_type.470) = var invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @Test.%as_type (%as_type.470) to %var = call %bound_method.loc31_31.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc31: @Test.%ptr (%ptr.ce1) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc31_31.1: @Test.%.loc31_31.14 (%.503) = impl_witness_access constants.%Iterate.lookup_impl_witness.135, element3 [symbolic = %impl.elem3.loc31_31.2 (constants.%impl.elem3.3bc)]
// CHECK:STDOUT:     %bound_method.loc31_31.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_31.1
// CHECK:STDOUT:     %.loc31_31.1: %Destroy.type = converted constants.%as_type.470, constants.%impl.elem1.b38 [symbolic = %impl.elem1 (constants.%impl.elem1.b38)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_31.2: <specific function> = specific_impl_function %impl.elem3.loc31_31.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.dca) [symbolic = %specific_impl_fn.loc31_31.5 (constants.%specific_impl_fn.7c6)]
// CHECK:STDOUT:     %bound_method.loc31_31.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_31.2
// CHECK:STDOUT:     %.loc31_31.2: ref %Optional.fb3 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.fb3 to %.loc31_31.2 = call %bound_method.loc31_31.4(%r.ref, %addr.loc31)
// CHECK:STDOUT:     %.loc31_31.3: ref %Optional.fb3 = temporary %.loc31_31.2, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_31.4: %Optional.HasValue.type.475 = specific_constant imports.%Core.import_ref.0b9, @Optional(constants.%OptionalStorage.facet.7ac) [concrete = constants.%Optional.HasValue.c00]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.475 = name_ref HasValue, %.loc31_31.4 [concrete = constants.%Optional.HasValue.c00]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_31.3, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.7ac) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_31.5: <bound method> = bound_method %.loc31_31.3, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_31.5: %Optional.fb3 = acquire_value %.loc31_31.3
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_31.5(%.loc31_31.5)
// CHECK:STDOUT:     %.loc31_31.6: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_31.7: bool = converted %Optional.HasValue.call, %.loc31_31.6
// CHECK:STDOUT:     if %.loc31_31.7 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_31.8: %Optional.Get.type.dd5 = specific_constant imports.%Core.import_ref.61f, @Optional(constants.%OptionalStorage.facet.7ac) [concrete = constants.%Optional.Get.335]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.dd5 = name_ref Get, %.loc31_31.8 [concrete = constants.%Optional.Get.335]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_31.3, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.7ac) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_31.6: <bound method> = bound_method %.loc31_31.3, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_31.9: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_31.10: %Optional.fb3 = acquire_value %.loc31_31.3
// CHECK:STDOUT:     %Optional.Get.call: init %ValueType to %.loc31_31.9 = call %bound_method.loc31_31.6(%.loc31_31.10)
// CHECK:STDOUT:     %.loc31_16: type = splice_block %ValueType.ref.loc31 [concrete = constants.%ValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %N.ref.loc31: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:       %ValueType.ref.loc31: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc31_31.11: ref %ValueType = temporary %.loc31_31.9, %Optional.Get.call
// CHECK:STDOUT:     %.loc31_31.12: %ValueType = acquire_value %.loc31_31.11
// CHECK:STDOUT:     %i: %ValueType = value_binding i, %.loc31_31.12
// CHECK:STDOUT:     %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:     %ValueType.cpp_operator.bound: <bound method> = bound_method %sum.ref, imports.%ValueType.cpp_operator.decl
// CHECK:STDOUT:     %.loc32_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:     %addr.loc32: %ptr.fed = addr_of %.loc32_12
// CHECK:STDOUT:     %.loc32_9.1: %ptr.140 = as_compatible %addr.loc32
// CHECK:STDOUT:     %.loc32_9.2: %ptr.140 = converted %addr.loc32, %.loc32_9.1
// CHECK:STDOUT:     %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc32_9.2)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %ValueType.Op.bound.loc31: <bound method> = bound_method %.loc31_31.11, constants.%ValueType.Op.1edc3c.1
// CHECK:STDOUT:     %Op.ref.loc31: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:     %ValueType.cpp_destructor.bound.loc31: <bound method> = bound_method %.loc31_31.11, %Op.ref.loc31
// CHECK:STDOUT:     %ValueType.cpp_destructor.call.loc31: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc31(%.loc31_31.11)
// CHECK:STDOUT:     %Destroy.Op.bound: <bound method> = bound_method %.loc31_31.3, constants.%Destroy.Op.2dd605.6
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc31_31.3)
// CHECK:STDOUT:     %impl.elem0.loc31_31.1: @Test.%.loc31_31.15 (%.c3e) = impl_witness_access constants.%Destroy.lookup_impl_witness.55f, element0 [symbolic = %impl.elem0.loc31_31.2 (constants.%impl.elem0.38e)]
// CHECK:STDOUT:     %bound_method.loc31_31.7: <bound method> = bound_method %var, %impl.elem0.loc31_31.1
// CHECK:STDOUT:     %specific_impl_fn.loc31_31.3: <specific function> = specific_impl_function %impl.elem0.loc31_31.1, @Destroy.WithSelf.Op(constants.%impl.elem1.b38) [symbolic = %specific_impl_fn.loc31_31.6 (constants.%specific_impl_fn.3a8)]
// CHECK:STDOUT:     %bound_method.loc31_31.8: <bound method> = bound_method %var, %specific_impl_fn.loc31_31.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc31_31.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.1(%self.param: ref %MaybeUnformed.d14) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.2(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.3(%self.param: ref %struct_type.value.has_value.330) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.4(%self.param: ref %DefaultOptionalStorage.046) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.5(%self.param: ref %Optional.fb3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R.bd8) {
// CHECK:STDOUT:   %R.loc27_10.1 => constants.%R.bd8
// CHECK:STDOUT:   %R.as_type.loc27_67.1 => constants.%R.as_type.e1d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.5f8) {
// CHECK:STDOUT:   %R.loc27_10.1 => constants.%facet_value.5f8
// CHECK:STDOUT:   %R.as_type.loc27_67.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.88d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc27 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.5a2
// CHECK:STDOUT:   %Iterate.facet => constants.%Iterate.facet.bdd
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.e45
// CHECK:STDOUT:   %.loc31_31.13 => constants.%.cc1
// CHECK:STDOUT:   %impl.elem2.loc31_31.2 => constants.%R.as_type.as.Iterate.impl.NewCursor.1c2
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.4 => constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.c5c
// CHECK:STDOUT:   %as_type => constants.%tuple.type.a15
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.443
// CHECK:STDOUT:   %ptr => constants.%ptr.261
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.3db
// CHECK:STDOUT:   %.loc31_31.14 => constants.%.c8f
// CHECK:STDOUT:   %impl.elem3.loc31_31.2 => constants.%R.as_type.as.Iterate.impl.Next.7bf
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.5 => constants.%R.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.6c1
// CHECK:STDOUT:   %.loc31_31.15 => constants.%.3f3
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e20f1f.7
// CHECK:STDOUT:   %impl.elem0.loc31_31.2 => constants.%Destroy.Op.2dd605.7
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.6 => constants.%Destroy.Op.2dd605.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_returns_different_types_mutable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.4e7: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.d22: type = fn_type @Optional.Get, @Optional(%T.4e7) [symbolic]
// CHECK:STDOUT:   %Optional.Get.6fc: %Optional.Get.type.d22 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.498: type = fn_type @Optional.HasValue, @Optional(%T.4e7) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.bf8: %Optional.HasValue.type.498 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.cb0: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f8b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.46f: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8c2: %T.as_type.as.OptionalStorage.impl.Get.type.46f = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.0ae: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.e0f: %T.as_type.as.OptionalStorage.impl.Has.type.0ae = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.e31: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.732: <witness> = lookup_impl_witness %.Self.e31, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.03d: %facet_type = impl_witness_access %Iterate.lookup_impl_witness.732, element0 [symbolic_self]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %pattern_type.94b: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor.type: type = fn_type @ValueType.cpp_destructor [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor: %ValueType.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.bd116f.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.1edc3c.1: %ValueType.Op.type.bd116f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.031: <witness> = custom_witness (%ValueType.Op.1edc3c.1), @Destroy [concrete]
// CHECK:STDOUT:   %const.e15: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.140: type = ptr_type %const.e15 [concrete]
// CHECK:STDOUT:   %ptr.fed: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.bd116f.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.1edc3c.2: %ValueType.Op.type.bd116f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d86: <witness> = custom_witness (%ValueType.Op.1edc3c.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.95f: %facet_type = facet_value %ValueType, (%custom_witness.031, %custom_witness.d86) [concrete]
// CHECK:STDOUT:   %Iterate_where.type.6d9: type = facet_type <@Iterate where %impl.elem0.03d = %facet_value.95f> [concrete]
// CHECK:STDOUT:   %R.bd8: %Iterate_where.type.6d9 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type.e1d: type = facet_access_type %R.bd8 [symbolic]
// CHECK:STDOUT:   %pattern_type.b93: type = pattern_type %R.as_type.e1d [symbolic]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.135: <witness> = lookup_impl_witness %R.bd8, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %R.as_type.e1d, (%Iterate.lookup_impl_witness.135) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.842: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.a4d: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [symbolic]
// CHECK:STDOUT:   %.89c: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.842, %Iterate.facet [symbolic]
// CHECK:STDOUT:   %impl.elem2.488: %.89c = impl_witness_access %Iterate.lookup_impl_witness.135, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.b38: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.135, element1 [symbolic]
// CHECK:STDOUT:   %as_type.470: type = facet_access_type %impl.elem1.b38 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5a0: <specific function> = specific_impl_function %impl.elem2.488, @Iterate.WithSelf.NewCursor(%Iterate.facet) [symbolic]
// CHECK:STDOUT:   %ptr.ce1: type = ptr_type %as_type.470 [symbolic]
// CHECK:STDOUT:   %.503: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.a4d, %Iterate.facet [symbolic]
// CHECK:STDOUT:   %impl.elem3.3bc: %.503 = impl_witness_access %Iterate.lookup_impl_witness.135, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.046: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.95f) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.342: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.041, @T.as_type.as.OptionalStorage.impl(%facet_value.95f) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.52a: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.342) [concrete]
// CHECK:STDOUT:   %Optional.68d: type = class_type @Optional, @Optional(%OptionalStorage.facet.52a) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.7c6: <specific function> = specific_impl_function %impl.elem3.3bc, @Iterate.WithSelf.Next(%Iterate.facet) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.426: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.52a) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.9f3: %Optional.HasValue.type.426 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.1e2: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.52a) [concrete]
// CHECK:STDOUT:   %Optional.Get.64e: %Optional.Get.type.1e2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.1a4: %Destroy.type = facet_value %ValueType, (%custom_witness.031) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.d14: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.1a4) [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.330: type = struct_type {.value: %MaybeUnformed.d14, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.9f3, @Optional.HasValue(%OptionalStorage.facet.52a) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.64e, @Optional.Get(%OptionalStorage.facet.52a) [concrete]
// CHECK:STDOUT:   %ValueType.cpp_operator.type: type = fn_type @ValueType.cpp_operator [concrete]
// CHECK:STDOUT:   %ValueType.cpp_operator: %ValueType.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.6: type = fn_type @Destroy.Op.loc32_31.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.6: %Destroy.Op.type.790b3b.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.55f: <witness> = lookup_impl_witness %impl.elem1.b38, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.567: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.b38) [symbolic]
// CHECK:STDOUT:   %.c3e: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.567, %impl.elem1.b38 [symbolic]
// CHECK:STDOUT:   %impl.elem0.38e: %.c3e = impl_witness_access %Destroy.lookup_impl_witness.55f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.3a8: <specific function> = specific_impl_function %impl.elem0.38e, @Destroy.WithSelf.Op(%impl.elem1.b38) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.0b9: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.498) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.bf8)]
// CHECK:STDOUT:   %Core.import_ref.61f: @Optional.%Optional.Get.type (%Optional.Get.type.d22) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.6fc)]
// CHECK:STDOUT:   %Core.import_ref.2b5: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.f8b)]
// CHECK:STDOUT:   %Core.import_ref.b5a = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Core.import_ref.977 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Core.import_ref.986: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.0ae) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.e0f)]
// CHECK:STDOUT:   %Core.import_ref.557: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.46f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8c2)]
// CHECK:STDOUT:   %Core.import_ref.999 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.041 = impl_witness_table (%Core.import_ref.2b5, %Core.import_ref.b5a, %Core.import_ref.977, %Core.import_ref.986, %Core.import_ref.557, %Core.import_ref.999), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ValueType = %ValueType.decl
// CHECK:STDOUT:     .MutableRange = %MutableRange.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ValueType.decl: type = class_decl @ValueType [concrete = constants.%ValueType] {} {}
// CHECK:STDOUT:   %ValueType.cpp_destructor.decl: %ValueType.cpp_destructor.type = fn_decl @ValueType.cpp_destructor [concrete = constants.%ValueType.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ValueType.cpp_operator.decl: %ValueType.cpp_operator.type = fn_decl @ValueType.cpp_operator [concrete = constants.%ValueType.cpp_operator] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MutableRange.decl: type = class_decl @MutableRange [concrete = constants.%MutableRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc28_10.2: %Iterate_where.type.6d9) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc28_10.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.135)]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %R.as_type.loc28_67.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet (constants.%Iterate.facet)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.842)]
// CHECK:STDOUT:   %.loc32_31.13: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet [symbolic = %.loc32_31.13 (constants.%.89c)]
// CHECK:STDOUT:   %impl.elem2.loc32_31.2: @Test.%.loc32_31.13 (%.89c) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc32_31.2 (constants.%impl.elem2.488)]
// CHECK:STDOUT:   %specific_impl_fn.loc32_31.4: <specific function> = specific_impl_function %impl.elem2.loc32_31.2, @Iterate.WithSelf.NewCursor(%Iterate.facet) [symbolic = %specific_impl_fn.loc32_31.4 (constants.%specific_impl_fn.5a0)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.ce1)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.a4d)]
// CHECK:STDOUT:   %.loc32_31.14: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet [symbolic = %.loc32_31.14 (constants.%.503)]
// CHECK:STDOUT:   %impl.elem3.loc32_31.2: @Test.%.loc32_31.14 (%.503) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc32_31.2 (constants.%impl.elem3.3bc)]
// CHECK:STDOUT:   %specific_impl_fn.loc32_31.5: <specific function> = specific_impl_function %impl.elem3.loc32_31.2, @Iterate.WithSelf.Next(%Iterate.facet) [symbolic = %specific_impl_fn.loc32_31.5 (constants.%specific_impl_fn.7c6)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.567)]
// CHECK:STDOUT:   %.loc32_31.15: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc32_31.15 (constants.%.c3e)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.55f)]
// CHECK:STDOUT:   %impl.elem0.loc32_31.2: @Test.%.loc32_31.15 (%.c3e) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc32_31.2 (constants.%impl.elem0.38e)]
// CHECK:STDOUT:   %specific_impl_fn.loc32_31.6: <specific function> = specific_impl_function %impl.elem0.loc32_31.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc32_31.6 (constants.%specific_impl_fn.3a8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @Test.%R.as_type.loc28_67.1 (%R.as_type.e1d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.94b = value_binding_pattern i [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @Test.%R.as_type.loc28_67.1 (%R.as_type.e1d) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc32_31.1: @Test.%.loc32_31.13 (%.89c) = impl_witness_access constants.%Iterate.lookup_impl_witness.135, element2 [symbolic = %impl.elem2.loc32_31.2 (constants.%impl.elem2.488)]
// CHECK:STDOUT:     %bound_method.loc32_31.1: <bound method> = bound_method %r.ref, %impl.elem2.loc32_31.1
// CHECK:STDOUT:     %specific_impl_fn.loc32_31.1: <specific function> = specific_impl_function %impl.elem2.loc32_31.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet) [symbolic = %specific_impl_fn.loc32_31.4 (constants.%specific_impl_fn.5a0)]
// CHECK:STDOUT:     %bound_method.loc32_31.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc32_31.1
// CHECK:STDOUT:     %var: ref @Test.%as_type (%as_type.470) = var invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @Test.%as_type (%as_type.470) to %var = call %bound_method.loc32_31.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc32: @Test.%ptr (%ptr.ce1) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc32_31.1: @Test.%.loc32_31.14 (%.503) = impl_witness_access constants.%Iterate.lookup_impl_witness.135, element3 [symbolic = %impl.elem3.loc32_31.2 (constants.%impl.elem3.3bc)]
// CHECK:STDOUT:     %bound_method.loc32_31.3: <bound method> = bound_method %r.ref, %impl.elem3.loc32_31.1
// CHECK:STDOUT:     %.loc32_31.1: %Destroy.type = converted constants.%as_type.470, constants.%impl.elem1.b38 [symbolic = %impl.elem1 (constants.%impl.elem1.b38)]
// CHECK:STDOUT:     %specific_impl_fn.loc32_31.2: <specific function> = specific_impl_function %impl.elem3.loc32_31.1, @Iterate.WithSelf.Next(constants.%Iterate.facet) [symbolic = %specific_impl_fn.loc32_31.5 (constants.%specific_impl_fn.7c6)]
// CHECK:STDOUT:     %bound_method.loc32_31.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc32_31.2
// CHECK:STDOUT:     %.loc32_31.2: ref %Optional.68d = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.68d to %.loc32_31.2 = call %bound_method.loc32_31.4(%r.ref, %addr.loc32)
// CHECK:STDOUT:     %.loc32_31.3: ref %Optional.68d = temporary %.loc32_31.2, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc32_31.4: %Optional.HasValue.type.426 = specific_constant imports.%Core.import_ref.0b9, @Optional(constants.%OptionalStorage.facet.52a) [concrete = constants.%Optional.HasValue.9f3]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.426 = name_ref HasValue, %.loc32_31.4 [concrete = constants.%Optional.HasValue.9f3]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc32_31.3, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.52a) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc32_31.5: <bound method> = bound_method %.loc32_31.3, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc32_31.5: %Optional.68d = acquire_value %.loc32_31.3
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc32_31.5(%.loc32_31.5)
// CHECK:STDOUT:     %.loc32_31.6: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc32_31.7: bool = converted %Optional.HasValue.call, %.loc32_31.6
// CHECK:STDOUT:     if %.loc32_31.7 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc32_31.8: %Optional.Get.type.1e2 = specific_constant imports.%Core.import_ref.61f, @Optional(constants.%OptionalStorage.facet.52a) [concrete = constants.%Optional.Get.64e]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.1e2 = name_ref Get, %.loc32_31.8 [concrete = constants.%Optional.Get.64e]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc32_31.3, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.52a) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc32_31.6: <bound method> = bound_method %.loc32_31.3, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc32_31.9: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc32_31.10: %Optional.68d = acquire_value %.loc32_31.3
// CHECK:STDOUT:     %Optional.Get.call: init %ValueType to %.loc32_31.9 = call %bound_method.loc32_31.6(%.loc32_31.10)
// CHECK:STDOUT:     %.loc32_16: type = splice_block %ValueType.ref.loc32 [concrete = constants.%ValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc32: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %N.ref.loc32: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:       %ValueType.ref.loc32: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc32_31.11: ref %ValueType = temporary %.loc32_31.9, %Optional.Get.call
// CHECK:STDOUT:     %.loc32_31.12: %ValueType = acquire_value %.loc32_31.11
// CHECK:STDOUT:     %i: %ValueType = value_binding i, %.loc32_31.12
// CHECK:STDOUT:     %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:     %ValueType.cpp_operator.bound: <bound method> = bound_method %sum.ref, imports.%ValueType.cpp_operator.decl
// CHECK:STDOUT:     %.loc33_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:     %addr.loc33: %ptr.fed = addr_of %.loc33_12
// CHECK:STDOUT:     %.loc33_9.1: %ptr.140 = as_compatible %addr.loc33
// CHECK:STDOUT:     %.loc33_9.2: %ptr.140 = converted %addr.loc33, %.loc33_9.1
// CHECK:STDOUT:     %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc33_9.2)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %ValueType.Op.bound.loc32: <bound method> = bound_method %.loc32_31.11, constants.%ValueType.Op.1edc3c.1
// CHECK:STDOUT:     %Op.ref.loc32: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:     %ValueType.cpp_destructor.bound.loc32: <bound method> = bound_method %.loc32_31.11, %Op.ref.loc32
// CHECK:STDOUT:     %ValueType.cpp_destructor.call.loc32: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc32(%.loc32_31.11)
// CHECK:STDOUT:     %Destroy.Op.bound: <bound method> = bound_method %.loc32_31.3, constants.%Destroy.Op.2dd605.6
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc32_31.3)
// CHECK:STDOUT:     %impl.elem0.loc32_31.1: @Test.%.loc32_31.15 (%.c3e) = impl_witness_access constants.%Destroy.lookup_impl_witness.55f, element0 [symbolic = %impl.elem0.loc32_31.2 (constants.%impl.elem0.38e)]
// CHECK:STDOUT:     %bound_method.loc32_31.7: <bound method> = bound_method %var, %impl.elem0.loc32_31.1
// CHECK:STDOUT:     %specific_impl_fn.loc32_31.3: <specific function> = specific_impl_function %impl.elem0.loc32_31.1, @Destroy.WithSelf.Op(constants.%impl.elem1.b38) [symbolic = %specific_impl_fn.loc32_31.6 (constants.%specific_impl_fn.3a8)]
// CHECK:STDOUT:     %bound_method.loc32_31.8: <bound method> = bound_method %var, %specific_impl_fn.loc32_31.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc32_31.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_31.1(%self.param: ref %MaybeUnformed.d14) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_31.2(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_31.3(%self.param: ref %struct_type.value.has_value.330) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_31.4(%self.param: ref %DefaultOptionalStorage.046) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc32_31.5(%self.param: ref %Optional.68d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R.bd8) {
// CHECK:STDOUT:   %R.loc28_10.1 => constants.%R.bd8
// CHECK:STDOUT:   %R.as_type.loc28_67.1 => constants.%R.as_type.e1d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_overloads.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.4e7: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.d22: type = fn_type @Optional.Get, @Optional(%T.4e7) [symbolic]
// CHECK:STDOUT:   %Optional.Get.6fc: %Optional.Get.type.d22 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.498: type = fn_type @Optional.HasValue, @Optional(%T.4e7) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.bf8: %Optional.HasValue.type.498 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.cb0: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f8b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.46f: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8c2: %T.as_type.as.OptionalStorage.impl.Get.type.46f = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.0ae: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.e0f: %T.as_type.as.OptionalStorage.impl.Has.type.0ae = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.87a: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.898: %T.as_type.as.OptionalStorage.impl.Some.type.87a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.ceb: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.cb0) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.9dd: %T.as_type.as.OptionalStorage.impl.None.type.ceb = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.e31: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.732: <witness> = lookup_impl_witness %.Self.e31, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.03d: %facet_type = impl_witness_access %Iterate.lookup_impl_witness.732, element0 [symbolic_self]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.94b: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor.type: type = fn_type @ValueType.cpp_destructor [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor: %ValueType.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.bd116f.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.1edc3c.1: %ValueType.Op.type.bd116f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.031: <witness> = custom_witness (%ValueType.Op.1edc3c.1), @Destroy [concrete]
// CHECK:STDOUT:   %const.e15: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.140: type = ptr_type %const.e15 [concrete]
// CHECK:STDOUT:   %ptr.fed: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.bd116f.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.1edc3c.2: %ValueType.Op.type.bd116f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d86: <witness> = custom_witness (%ValueType.Op.1edc3c.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.95f: %facet_type = facet_value %ValueType, (%custom_witness.031, %custom_witness.d86) [concrete]
// CHECK:STDOUT:   %Iterate_where.type.6d9: type = facet_type <@Iterate where %impl.elem0.03d = %facet_value.95f> [concrete]
// CHECK:STDOUT:   %R.bd8: %Iterate_where.type.6d9 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type.e1d: type = facet_access_type %R.bd8 [symbolic]
// CHECK:STDOUT:   %pattern_type.b93: type = pattern_type %R.as_type.e1d [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %.Self.10b: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.5bf: <witness> = lookup_impl_witness %.Self.10b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d47: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.bce: <witness> = lookup_impl_witness %impl.elem0.d47, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c87: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.bce, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.846: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.d47 impls @Destroy and %impl.elem0.d47 impls @Copy and %impl.elem0.d47 impls @CppUnsafeDeref and %impl.elem0.d47 impls @EqWith, @EqWith(%impl.elem1.846) and %impl.elem0.d47 impls @Inc and %impl.elem0.c87 impls @Destroy and %impl.elem0.c87 impls @Copy and %impl.elem1.846 impls @Destroy and %impl.elem1.846 impls @Copy> [concrete]
// CHECK:STDOUT:   %R.6c6: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.edd1fe.2: <witness> = lookup_impl_witness %R.6c6, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.bdcd2d.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.d5966e.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.384: type = tuple_type (%impl.elem0.d5966e.2, %impl.elem1.bdcd2d.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.0ba: <witness> = lookup_impl_witness %tuple.type.384, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.2fe: %Destroy.type = facet_value %tuple.type.384, (%Destroy.lookup_impl_witness.0ba) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.ca4: <witness> = lookup_impl_witness %impl.elem0.d5966e.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.b10: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.ca4, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.3d8: <witness> = lookup_impl_witness %impl.elem0.b10, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.bff: <witness> = lookup_impl_witness %impl.elem0.b10, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.a30: %facet_type = facet_value %impl.elem0.b10, (%Destroy.lookup_impl_witness.3d8, %Copy.lookup_impl_witness.bff) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.5b5: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%R.6c6) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.cd2: %R.as_type.as.Iterate.impl.Next.type.5b5 = struct_value () [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.9b6: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%R.6c6) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.b29: %R.as_type.as.Iterate.impl.NewCursor.type.9b6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.135: <witness> = lookup_impl_witness %R.bd8, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.dca: %Iterate.type = facet_value %R.as_type.e1d, (%Iterate.lookup_impl_witness.135) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.842: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.dca) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.a4d: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.dca) [symbolic]
// CHECK:STDOUT:   %.89c: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.842, %Iterate.facet.dca [symbolic]
// CHECK:STDOUT:   %impl.elem2.488: %.89c = impl_witness_access %Iterate.lookup_impl_witness.135, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.b38: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.135, element1 [symbolic]
// CHECK:STDOUT:   %as_type.470: type = facet_access_type %impl.elem1.b38 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5a0: <specific function> = specific_impl_function %impl.elem2.488, @Iterate.WithSelf.NewCursor(%Iterate.facet.dca) [symbolic]
// CHECK:STDOUT:   %ptr.ce1: type = ptr_type %as_type.470 [symbolic]
// CHECK:STDOUT:   %.503: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.a4d, %Iterate.facet.dca [symbolic]
// CHECK:STDOUT:   %impl.elem3.3bc: %.503 = impl_witness_access %Iterate.lookup_impl_witness.135, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.046: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.95f) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.1ee: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.6f7, @T.as_type.as.OptionalStorage.impl(%facet_value.95f) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.7ac: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.1ee) [concrete]
// CHECK:STDOUT:   %Optional.fb3: type = class_type @Optional, @Optional(%OptionalStorage.facet.7ac) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.7c6: <specific function> = specific_impl_function %impl.elem3.3bc, @Iterate.WithSelf.Next(%Iterate.facet.dca) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.475: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.7ac) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.c00: %Optional.HasValue.type.475 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.dd5: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.7ac) [concrete]
// CHECK:STDOUT:   %Optional.Get.335: %Optional.Get.type.dd5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.1a4: %Destroy.type = facet_value %ValueType, (%custom_witness.031) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.d14: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.1a4) [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.330: type = struct_type {.value: %MaybeUnformed.d14, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.c00, @Optional.HasValue(%OptionalStorage.facet.7ac) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.335, @Optional.Get(%OptionalStorage.facet.7ac) [concrete]
// CHECK:STDOUT:   %ValueType.cpp_operator.type: type = fn_type @ValueType.cpp_operator [concrete]
// CHECK:STDOUT:   %ValueType.cpp_operator: %ValueType.cpp_operator.type = struct_value () [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.6: type = fn_type @Destroy.Op.loc39_31.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.6: %Destroy.Op.type.790b3b.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.55f: <witness> = lookup_impl_witness %impl.elem1.b38, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.567: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.b38) [symbolic]
// CHECK:STDOUT:   %.c3e: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.567, %impl.elem1.b38 [symbolic]
// CHECK:STDOUT:   %impl.elem0.38e: %.c3e = impl_witness_access %Destroy.lookup_impl_witness.55f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.3a8: <specific function> = specific_impl_function %impl.elem0.38e, @Destroy.WithSelf.Op(%impl.elem1.b38) [symbolic]
// CHECK:STDOUT:   %MutableAndConstRange: type = class_type @MutableAndConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.e6c: type = pattern_type %MutableAndConstRange [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.af9: <witness> = custom_witness (%Iterator, %Iterator, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.152308.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.b8fe13.1: %Iterator.Op.type.152308.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.016: <witness> = custom_witness (%ValueType, %Iterator.Op.b8fe13.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.152308.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.b8fe13.2: %Iterator.Op.type.152308.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.481: <witness> = custom_witness (%Iterator.Op.b8fe13.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.152308.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.b8fe13.3: %Iterator.Op.type.152308.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.307: <witness> = custom_witness (%Iterator.Op.b8fe13.3), @Copy [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.571: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.152308.4: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.b8fe13.4: %Iterator.Op.type.152308.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6de: <witness> = custom_witness (%Iterator.Op.b8fe13.4), @Inc [concrete]
// CHECK:STDOUT:   %facet_value.7d8: %CppRangeForIterate_where.type = facet_value %MutableAndConstRange, (%custom_witness.031, %custom_witness.d86, %custom_witness.af9, %custom_witness.481, %custom_witness.307, %custom_witness.016, %custom_witness.571, %custom_witness.6de) [concrete]
// CHECK:STDOUT:   %tuple.type.084: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.7: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.7: %Destroy.Op.type.790b3b.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.7: <witness> = custom_witness (%Destroy.Op.2dd605.7), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.da4: %Destroy.type = facet_value %tuple.type.084, (%custom_witness.e20f1f.7) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.902: <witness> = impl_witness imports.%Iterate.impl_witness_table.9f0, @R.as_type.as.Iterate.impl(%facet_value.7d8) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.377: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%facet_value.7d8) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.310: %R.as_type.as.Iterate.impl.NewCursor.type.377 = struct_value () [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.3d2: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%facet_value.7d8) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.06d: %R.as_type.as.Iterate.impl.Next.type.3d2 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.227: %Iterate.type = facet_value %MutableAndConstRange, (%Iterate.impl_witness.902) [concrete]
// CHECK:STDOUT:   %facet_value.8e0: %Iterate_where.type.6d9 = facet_value %MutableAndConstRange, (%Iterate.impl_witness.902) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.087: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.227) [concrete]
// CHECK:STDOUT:   %.4d4: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.087, %Iterate.facet.227 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %R.as_type.as.Iterate.impl.NewCursor.310, @R.as_type.as.Iterate.impl.NewCursor(%facet_value.7d8) [concrete]
// CHECK:STDOUT:   %complete_type.7b5: <witness> = complete_type_witness %tuple.type.084 [concrete]
// CHECK:STDOUT:   %ptr.877: type = ptr_type %tuple.type.084 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.f81: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.227) [concrete]
// CHECK:STDOUT:   %.728: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.f81, %Iterate.facet.227 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %R.as_type.as.Iterate.impl.Next.06d, @R.as_type.as.Iterate.impl.Next(%facet_value.7d8) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.1ee: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.da4) [concrete]
// CHECK:STDOUT:   %.e5c: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.1ee, %Destroy.facet.da4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.0b9: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.498) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.bf8)]
// CHECK:STDOUT:   %Core.import_ref.61f: @Optional.%Optional.Get.type (%Optional.Get.type.d22) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.6fc)]
// CHECK:STDOUT:   %Core.import_ref.2b5: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.f8b)]
// CHECK:STDOUT:   %Core.import_ref.b0a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.ceb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.9dd)]
// CHECK:STDOUT:   %Core.import_ref.3fc: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.87a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.898)]
// CHECK:STDOUT:   %Core.import_ref.986: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.0ae) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.e0f)]
// CHECK:STDOUT:   %Core.import_ref.557: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.46f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8c2)]
// CHECK:STDOUT:   %Core.import_ref.999 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.6f7 = impl_witness_table (%Core.import_ref.2b5, %Core.import_ref.b0a, %Core.import_ref.3fc, %Core.import_ref.986, %Core.import_ref.557, %Core.import_ref.999), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .ValueType = %ValueType.decl
// CHECK:STDOUT:     .MutableAndConstRange = %MutableAndConstRange.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ValueType.decl: type = class_decl @ValueType [concrete = constants.%ValueType] {} {}
// CHECK:STDOUT:   %ValueType.cpp_destructor.decl: %ValueType.cpp_destructor.type = fn_decl @ValueType.cpp_destructor [concrete = constants.%ValueType.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.89a: %facet_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.a30)]
// CHECK:STDOUT:   %Core.import_ref.9ea: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.2fe)]
// CHECK:STDOUT:   %Core.import_ref.a6c: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor.type (%R.as_type.as.Iterate.impl.NewCursor.type.9b6) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor (constants.%R.as_type.as.Iterate.impl.NewCursor.b29)]
// CHECK:STDOUT:   %Core.import_ref.a02: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next.type (%R.as_type.as.Iterate.impl.Next.type.5b5) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next (constants.%R.as_type.as.Iterate.impl.Next.cd2)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.9f0 = impl_witness_table (%Core.import_ref.89a, %Core.import_ref.9ea, %Core.import_ref.a6c, %Core.import_ref.a02), @R.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %ValueType.cpp_operator.decl: %ValueType.cpp_operator.type = fn_decl @ValueType.cpp_operator [concrete = constants.%ValueType.cpp_operator] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MutableAndConstRange.decl: type = class_decl @MutableAndConstRange [concrete = constants.%MutableAndConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc35_10.2: %Iterate_where.type.6d9) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc35_10.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.135)]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %R.as_type.loc35_67.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet (constants.%Iterate.facet.dca)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.842)]
// CHECK:STDOUT:   %.loc39_31.13: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet [symbolic = %.loc39_31.13 (constants.%.89c)]
// CHECK:STDOUT:   %impl.elem2.loc39_31.2: @Test.%.loc39_31.13 (%.89c) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc39_31.2 (constants.%impl.elem2.488)]
// CHECK:STDOUT:   %specific_impl_fn.loc39_31.4: <specific function> = specific_impl_function %impl.elem2.loc39_31.2, @Iterate.WithSelf.NewCursor(%Iterate.facet) [symbolic = %specific_impl_fn.loc39_31.4 (constants.%specific_impl_fn.5a0)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.ce1)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.a4d)]
// CHECK:STDOUT:   %.loc39_31.14: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet [symbolic = %.loc39_31.14 (constants.%.503)]
// CHECK:STDOUT:   %impl.elem3.loc39_31.2: @Test.%.loc39_31.14 (%.503) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc39_31.2 (constants.%impl.elem3.3bc)]
// CHECK:STDOUT:   %specific_impl_fn.loc39_31.5: <specific function> = specific_impl_function %impl.elem3.loc39_31.2, @Iterate.WithSelf.Next(%Iterate.facet) [symbolic = %specific_impl_fn.loc39_31.5 (constants.%specific_impl_fn.7c6)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.567)]
// CHECK:STDOUT:   %.loc39_31.15: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc39_31.15 (constants.%.c3e)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.55f)]
// CHECK:STDOUT:   %impl.elem0.loc39_31.2: @Test.%.loc39_31.15 (%.c3e) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc39_31.2 (constants.%impl.elem0.38e)]
// CHECK:STDOUT:   %specific_impl_fn.loc39_31.6: <specific function> = specific_impl_function %impl.elem0.loc39_31.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc39_31.6 (constants.%specific_impl_fn.3a8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @Test.%R.as_type.loc35_67.1 (%R.as_type.e1d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.94b = value_binding_pattern i [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @Test.%R.as_type.loc35_67.1 (%R.as_type.e1d) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc39_31.1: @Test.%.loc39_31.13 (%.89c) = impl_witness_access constants.%Iterate.lookup_impl_witness.135, element2 [symbolic = %impl.elem2.loc39_31.2 (constants.%impl.elem2.488)]
// CHECK:STDOUT:     %bound_method.loc39_31.1: <bound method> = bound_method %r.ref, %impl.elem2.loc39_31.1
// CHECK:STDOUT:     %specific_impl_fn.loc39_31.1: <specific function> = specific_impl_function %impl.elem2.loc39_31.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.dca) [symbolic = %specific_impl_fn.loc39_31.4 (constants.%specific_impl_fn.5a0)]
// CHECK:STDOUT:     %bound_method.loc39_31.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc39_31.1
// CHECK:STDOUT:     %var: ref @Test.%as_type (%as_type.470) = var invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @Test.%as_type (%as_type.470) to %var = call %bound_method.loc39_31.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc39: @Test.%ptr (%ptr.ce1) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc39_31.1: @Test.%.loc39_31.14 (%.503) = impl_witness_access constants.%Iterate.lookup_impl_witness.135, element3 [symbolic = %impl.elem3.loc39_31.2 (constants.%impl.elem3.3bc)]
// CHECK:STDOUT:     %bound_method.loc39_31.3: <bound method> = bound_method %r.ref, %impl.elem3.loc39_31.1
// CHECK:STDOUT:     %.loc39_31.1: %Destroy.type = converted constants.%as_type.470, constants.%impl.elem1.b38 [symbolic = %impl.elem1 (constants.%impl.elem1.b38)]
// CHECK:STDOUT:     %specific_impl_fn.loc39_31.2: <specific function> = specific_impl_function %impl.elem3.loc39_31.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.dca) [symbolic = %specific_impl_fn.loc39_31.5 (constants.%specific_impl_fn.7c6)]
// CHECK:STDOUT:     %bound_method.loc39_31.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc39_31.2
// CHECK:STDOUT:     %.loc39_31.2: ref %Optional.fb3 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.fb3 to %.loc39_31.2 = call %bound_method.loc39_31.4(%r.ref, %addr.loc39)
// CHECK:STDOUT:     %.loc39_31.3: ref %Optional.fb3 = temporary %.loc39_31.2, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc39_31.4: %Optional.HasValue.type.475 = specific_constant imports.%Core.import_ref.0b9, @Optional(constants.%OptionalStorage.facet.7ac) [concrete = constants.%Optional.HasValue.c00]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.475 = name_ref HasValue, %.loc39_31.4 [concrete = constants.%Optional.HasValue.c00]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc39_31.3, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.7ac) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc39_31.5: <bound method> = bound_method %.loc39_31.3, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc39_31.5: %Optional.fb3 = acquire_value %.loc39_31.3
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc39_31.5(%.loc39_31.5)
// CHECK:STDOUT:     %.loc39_31.6: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc39_31.7: bool = converted %Optional.HasValue.call, %.loc39_31.6
// CHECK:STDOUT:     if %.loc39_31.7 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc39_31.8: %Optional.Get.type.dd5 = specific_constant imports.%Core.import_ref.61f, @Optional(constants.%OptionalStorage.facet.7ac) [concrete = constants.%Optional.Get.335]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.dd5 = name_ref Get, %.loc39_31.8 [concrete = constants.%Optional.Get.335]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc39_31.3, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.7ac) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc39_31.6: <bound method> = bound_method %.loc39_31.3, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc39_31.9: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc39_31.10: %Optional.fb3 = acquire_value %.loc39_31.3
// CHECK:STDOUT:     %Optional.Get.call: init %ValueType to %.loc39_31.9 = call %bound_method.loc39_31.6(%.loc39_31.10)
// CHECK:STDOUT:     %.loc39_16: type = splice_block %ValueType.ref.loc39 [concrete = constants.%ValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc39: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %N.ref.loc39: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:       %ValueType.ref.loc39: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc39_31.11: ref %ValueType = temporary %.loc39_31.9, %Optional.Get.call
// CHECK:STDOUT:     %.loc39_31.12: %ValueType = acquire_value %.loc39_31.11
// CHECK:STDOUT:     %i: %ValueType = value_binding i, %.loc39_31.12
// CHECK:STDOUT:     %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:     %ValueType.cpp_operator.bound: <bound method> = bound_method %sum.ref, imports.%ValueType.cpp_operator.decl
// CHECK:STDOUT:     %.loc40_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:     %addr.loc40: %ptr.fed = addr_of %.loc40_12
// CHECK:STDOUT:     %.loc40_9.1: %ptr.140 = as_compatible %addr.loc40
// CHECK:STDOUT:     %.loc40_9.2: %ptr.140 = converted %addr.loc40, %.loc40_9.1
// CHECK:STDOUT:     %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc40_9.2)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %ValueType.Op.bound.loc39: <bound method> = bound_method %.loc39_31.11, constants.%ValueType.Op.1edc3c.1
// CHECK:STDOUT:     %Op.ref.loc39: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:     %ValueType.cpp_destructor.bound.loc39: <bound method> = bound_method %.loc39_31.11, %Op.ref.loc39
// CHECK:STDOUT:     %ValueType.cpp_destructor.call.loc39: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc39(%.loc39_31.11)
// CHECK:STDOUT:     %Destroy.Op.bound: <bound method> = bound_method %.loc39_31.3, constants.%Destroy.Op.2dd605.6
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc39_31.3)
// CHECK:STDOUT:     %impl.elem0.loc39_31.1: @Test.%.loc39_31.15 (%.c3e) = impl_witness_access constants.%Destroy.lookup_impl_witness.55f, element0 [symbolic = %impl.elem0.loc39_31.2 (constants.%impl.elem0.38e)]
// CHECK:STDOUT:     %bound_method.loc39_31.7: <bound method> = bound_method %var, %impl.elem0.loc39_31.1
// CHECK:STDOUT:     %specific_impl_fn.loc39_31.3: <specific function> = specific_impl_function %impl.elem0.loc39_31.1, @Destroy.WithSelf.Op(constants.%impl.elem1.b38) [symbolic = %specific_impl_fn.loc39_31.6 (constants.%specific_impl_fn.3a8)]
// CHECK:STDOUT:     %bound_method.loc39_31.8: <bound method> = bound_method %var, %specific_impl_fn.loc39_31.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc39_31.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc39_31.1(%self.param: ref %MaybeUnformed.d14) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc39_31.2(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc39_31.3(%self.param: ref %struct_type.value.has_value.330) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc39_31.4(%self.param: ref %DefaultOptionalStorage.046) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc39_31.5(%self.param: ref %Optional.fb3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R.bd8) {
// CHECK:STDOUT:   %R.loc35_10.1 => constants.%R.bd8
// CHECK:STDOUT:   %R.as_type.loc35_67.1 => constants.%R.as_type.e1d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.8e0) {
// CHECK:STDOUT:   %R.loc35_10.1 => constants.%facet_value.8e0
// CHECK:STDOUT:   %R.as_type.loc35_67.1 => constants.%MutableAndConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e6c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc35 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.902
// CHECK:STDOUT:   %Iterate.facet => constants.%Iterate.facet.227
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.087
// CHECK:STDOUT:   %.loc39_31.13 => constants.%.4d4
// CHECK:STDOUT:   %impl.elem2.loc39_31.2 => constants.%R.as_type.as.Iterate.impl.NewCursor.310
// CHECK:STDOUT:   %specific_impl_fn.loc39_31.4 => constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.da4
// CHECK:STDOUT:   %as_type => constants.%tuple.type.084
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.7b5
// CHECK:STDOUT:   %ptr => constants.%ptr.877
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.f81
// CHECK:STDOUT:   %.loc39_31.14 => constants.%.728
// CHECK:STDOUT:   %impl.elem3.loc39_31.2 => constants.%R.as_type.as.Iterate.impl.Next.06d
// CHECK:STDOUT:   %specific_impl_fn.loc39_31.5 => constants.%R.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.1ee
// CHECK:STDOUT:   %.loc39_31.15 => constants.%.e5c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e20f1f.7
// CHECK:STDOUT:   %impl.elem0.loc39_31.2 => constants.%Destroy.Op.2dd605.7
// CHECK:STDOUT:   %specific_impl_fn.loc39_31.6 => constants.%Destroy.Op.2dd605.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
